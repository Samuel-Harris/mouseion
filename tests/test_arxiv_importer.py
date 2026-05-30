from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mouseion.config import Settings
from mouseion.domain.models import ReadDocumentInput, SearchInput
from mouseion.errors import EmbeddingError
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.ingestor import Ingestor
from mouseion.ingest.repo import RepoService
from mouseion.services.document_reader import DocumentReader
from mouseion.services.exporter import Exporter
from mouseion.services.search import SearchService
from mouseion.services.service import MouseionService
from mouseion.storage.db import SQLiteStore
from utilities.arxiv.import_arxiv_metadata import (
    ImportOptions,
    async_main,
    category_filter_codes,
    import_arxiv_metadata,
    load_category_catalog,
    raw_categories_match_filter,
)


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0] * 767 + [1.0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 767 + [1.0] for _ in texts]


class RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return await super().embed_many(texts)


class FailingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise EmbeddingError("embedding backend unavailable")


class ReadinessFailingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.ready_calls = 0
        self.embed_calls = 0

    async def ensure_ready(self) -> None:
        self.ready_calls += 1
        raise EmbeddingError("embedding backend unavailable")

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return await super().embed_many(texts)


@pytest.fixture
def categories_json(tmp_path: Path) -> Path:
    path = tmp_path / "categories.json"
    path.write_text(
        json.dumps(
            [
                {
                    "Computer Science": [
                        {
                            "cs.AI": {
                                "name": "Artificial Intelligence",
                                "description": "AI papers",
                            }
                        },
                        {
                            "cs.CG": {
                                "name": "Computational Geometry",
                                "description": "Geometry papers",
                            }
                        },
                    ]
                },
                {
                    "Statistics": [
                        {
                            "stat.ML": {
                                "name": "Machine Learning",
                                "description": "Statistical machine learning",
                            }
                        }
                    ]
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def arxiv_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "arxiv.jsonl"
    records = [
        {
            "id": "1234.0001",
            "submitter": "Ada Lovelace",
            "authors": "Ada Lovelace and Grace Hopper",
            "title": " Neural Symbolic Planning\n ",
            "comments": "12 pages",
            "journal-ref": "Journal Ref",
            "doi": "10.1000/example",
            "report-no": "REPORT-1",
            "categories": "cs.AI stat.ML",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "  This paper contains a deliberately unique phrase about zeta planning.\n",
            "versions": [{"version": "v1", "created": "Fri, 1 May 2026 00:00:00 GMT"}],
            "update_date": "2026-05-01",
            "authors_parsed": [["Lovelace", "Ada", ""], ["Hopper", "Grace", ""]],
        },
        {
            "id": "1234.0002",
            "submitter": "Louis Theran",
            "authors": "Louis Theran",
            "title": "Sparse Graphs",
            "categories": "cs.CG unknown.XY",
            "abstract": "A sparse graph abstract with another unique term.",
            "versions": [],
            "update_date": "2026-05-02",
            "authors_parsed": [["Theran", "Louis", ""]],
        },
        {
            "id": "1234.0003",
            "title": "Physics Paper",
            "categories": "hep-ph",
            "abstract": "This one is filtered unless all records are imported.",
        },
    ]
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    return path


@pytest.fixture
async def imported_service(
    tmp_path: Path, categories_json: Path, arxiv_jsonl: Path
) -> AsyncIterator[tuple[SQLiteStore, MouseionService, Settings]]:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    await import_arxiv_metadata(
        ImportOptions(
            input=arxiv_jsonl,
            categories_json=categories_json,
            groups=("computer-science",),
            limit=2,
            batch_size=2,
            progress_every=0,
        ),
        settings=settings,
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )
    store = SQLiteStore(settings)
    await store.open()
    embedder = FakeEmbedder()
    searcher = SearchService(store, embedder, settings.rrf_k)  # type: ignore[arg-type]
    service = MouseionService(
        store,
        Ingestor(settings),
        Chunker(settings),
        embedder,  # type: ignore[arg-type]
        searcher,
        RepoService(settings),
        Exporter(settings, store),
        DocumentReader(store, searcher),
    )
    try:
        yield store, service, settings
    finally:
        await store.close()


async def test_dry_run_reports_counts_without_document_writes(
    tmp_path: Path, categories_json: Path, arxiv_jsonl: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")

    rerun_embedder = RecordingEmbedder()
    stats = await import_arxiv_metadata(
        ImportOptions(
            input=arxiv_jsonl,
            categories_json=categories_json,
            groups=("computer-science",),
            dry_run=True,
            progress_every=0,
        ),
        settings=settings,
        embedder=rerun_embedder,  # type: ignore[arg-type]
    )

    assert stats.selected == 2
    assert stats.inserted == 2
    assert stats.updated == 0
    assert not settings.sqlite_path.exists()


async def test_cli_asks_user_to_download_missing_input(
    tmp_path: Path, categories_json: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_input = tmp_path / "missing-arxiv.json"

    status = await async_main(
        [
            "--input",
            str(missing_input),
            "--categories-json",
            str(categories_json),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert f"arXiv metadata JSONL file not found: {missing_input}" in captured.err
    assert (
        "https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download" in captured.err
    )
    assert "Please download" in captured.err


async def test_import_fails_fast_when_embedding_backend_is_unavailable(
    tmp_path: Path, categories_json: Path, arxiv_jsonl: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    embedder = FailingEmbedder()

    with pytest.raises(EmbeddingError, match="embedding backend unavailable"):
        await import_arxiv_metadata(
            ImportOptions(
                input=arxiv_jsonl,
                categories_json=categories_json,
                groups=("computer-science",),
                batch_size=1,
                progress_every=0,
            ),
            settings=settings,
            embedder=embedder,  # type: ignore[arg-type]
        )

    assert embedder.calls == 1


async def test_import_checks_embedding_backend_before_scanning_records(
    tmp_path: Path, categories_json: Path, arxiv_jsonl: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    embedder = ReadinessFailingEmbedder()

    with pytest.raises(EmbeddingError, match="embedding backend unavailable"):
        await import_arxiv_metadata(
            ImportOptions(
                input=arxiv_jsonl,
                categories_json=categories_json,
                groups=("computer-science",),
                progress_every=0,
            ),
            settings=settings,
            embedder=embedder,  # type: ignore[arg-type]
        )

    assert embedder.ready_calls == 1
    assert embedder.embed_calls == 0
    assert not settings.sqlite_path.exists()


async def test_cli_reports_embedding_backend_error_once(
    tmp_path: Path, categories_json: Path, arxiv_jsonl: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")

    status = await async_main(
        [
            "--input",
            str(arxiv_jsonl),
            "--categories-json",
            str(categories_json),
            "--groups",
            "computer-science",
            "--batch-size",
            "1",
            "--progress-every",
            "0",
        ],
        settings=settings,
        embedder=FailingEmbedder(),  # type: ignore[arg-type]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert captured.err.count("embedding backend unavailable") == 1


async def test_import_creates_searchable_documents_with_arxiv_metadata_and_tags(
    imported_service: tuple[SQLiteStore, MouseionService, Settings],
) -> None:
    store, service, _ = imported_service

    result = await service.search(SearchInput(query="zeta planning", top_k=3))
    document = result["results"][0]["document"]

    assert document["source"] == "arxiv:1234.0001"
    assert document["type"] == "document"
    assert document["title"] == "Neural Symbolic Planning"
    assert document["metadata"]["pdf_url"] == "https://export.arxiv.org/pdf/1234.0001"
    assert document["metadata"]["html_url"] == "https://export.arxiv.org/html/1234.0001"
    assert "journal_ref" not in document["metadata"]
    assert "authors_parsed" not in document["metadata"]
    full_document = await service.read_document(ReadDocumentInput(document_id=document["id"]))
    assert full_document["document"]["metadata"]["journal_ref"] == "Journal Ref"
    assert full_document["document"]["metadata"]["authors_parsed"] == [
        ["Lovelace", "Ada", ""],
        ["Hopper", "Grace", ""],
    ]
    assert document["tags"] == [
        "arxiv",
        "arxiv:category:cs.ai",
        "arxiv:category:stat.ml",
        "arxiv:group:computer-science",
        "arxiv:group:statistics",
    ]

    chunks = await store.execute(
        "SELECT content FROM chunks WHERE document_id = ?", (document["id"],)
    )
    assert "Neural Symbolic Planning" in chunks.first()["content"]  # type: ignore[index]
    assert "zeta planning" in chunks.first()["content"]  # type: ignore[index]

    author_result = await service.search(SearchInput(query="Grace Hopper", top_k=3))
    source_result = await service.search(SearchInput(query="1234.0001", top_k=3))
    category_result = await service.search(SearchInput(query="stat.ML", top_k=3))

    assert author_result["results"][0]["document"]["source"] == "arxiv:1234.0001"
    assert source_result["results"][0]["document"]["source"] == "arxiv:1234.0001"
    assert category_result["results"][0]["document"]["source"] == "arxiv:1234.0001"


async def test_import_batches_embeddings_and_writes_one_chunk_per_paper(
    tmp_path: Path, categories_json: Path, arxiv_jsonl: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    embedder = RecordingEmbedder()

    await import_arxiv_metadata(
        ImportOptions(
            input=arxiv_jsonl,
            categories_json=categories_json,
            groups=("computer-science",),
            limit=2,
            batch_size=2,
            progress_every=0,
        ),
        settings=settings,
        embedder=embedder,  # type: ignore[arg-type]
    )

    store = SQLiteStore(settings)
    await store.open()
    try:
        chunks = await store.execute("SELECT count(*) AS total FROM chunks")
    finally:
        await store.close()

    assert embedder.calls == [
        [
            "Neural Symbolic Planning\n\n"
            "This paper contains a deliberately unique phrase about zeta planning.",
            "Sparse Graphs\n\nA sparse graph abstract with another unique term.",
        ]
    ]
    assert chunks.first()["total"] == 2  # type: ignore[index]


async def test_rerun_skips_unchanged_existing_document_without_duplicates(
    imported_service: tuple[SQLiteStore, MouseionService, Settings],
    categories_json: Path,
    arxiv_jsonl: Path,
) -> None:
    store, _, settings = imported_service
    first_count_row = (
        await store.execute("SELECT count(*) AS total FROM documents WHERE source LIKE 'arxiv:%'")
    ).first()
    assert first_count_row is not None
    first_count = first_count_row["total"]
    rerun_embedder = RecordingEmbedder()

    dry_run = await import_arxiv_metadata(
        ImportOptions(
            input=arxiv_jsonl,
            categories_json=categories_json,
            groups=("computer-science",),
            limit=2,
            dry_run=True,
            progress_every=0,
        ),
        settings=settings,
        embedder=rerun_embedder,  # type: ignore[arg-type]
    )
    stats = await import_arxiv_metadata(
        ImportOptions(
            input=arxiv_jsonl,
            categories_json=categories_json,
            groups=("computer-science",),
            limit=2,
            batch_size=1,
            progress_every=0,
        ),
        settings=settings,
        embedder=rerun_embedder,  # type: ignore[arg-type]
    )
    second_count_row = (
        await store.execute("SELECT count(*) AS total FROM documents WHERE source LIKE 'arxiv:%'")
    ).first()
    assert second_count_row is not None
    second_count = second_count_row["total"]

    assert first_count == 2
    assert dry_run.inserted == 0
    assert dry_run.updated == 0
    assert dry_run.skipped == 2
    assert second_count == 2
    assert stats.inserted == 0
    assert stats.updated == 0
    assert stats.skipped == 2
    assert rerun_embedder.calls == []


async def test_rerun_updates_changed_existing_document(
    imported_service: tuple[SQLiteStore, MouseionService, Settings],
    categories_json: Path,
    arxiv_jsonl: Path,
) -> None:
    store, _, settings = imported_service
    arxiv_jsonl.write_text(
        json.dumps(
            {
                "id": "1234.0001",
                "submitter": "Ada Lovelace",
                "authors": "Ada Lovelace and Grace Hopper",
                "title": "Neural Symbolic Planning",
                "categories": "cs.AI stat.ML",
                "abstract": "A changed abstract with a fresh update phrase.",
                "versions": [{"version": "v2", "created": "Sat, 2 May 2026 00:00:00 GMT"}],
                "update_date": "2026-05-02",
                "authors_parsed": [["Lovelace", "Ada", ""], ["Hopper", "Grace", ""]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rerun_embedder = RecordingEmbedder()

    stats = await import_arxiv_metadata(
        ImportOptions(
            input=arxiv_jsonl,
            categories_json=categories_json,
            groups=("computer-science",),
            batch_size=10,
            progress_every=0,
        ),
        settings=settings,
        embedder=rerun_embedder,  # type: ignore[arg-type]
    )
    updated_chunk = await store.execute(
        """
        SELECT c.content
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.source = ?
        """,
        ("arxiv:1234.0001",),
    )

    assert stats.inserted == 0
    assert stats.updated == 1
    assert stats.skipped == 0
    assert len(rerun_embedder.calls) == 1
    assert "fresh update phrase" in updated_chunk.first()["content"]  # type: ignore[index]


async def test_group_and_category_filters_use_union_semantics(
    tmp_path: Path, categories_json: Path, arxiv_jsonl: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")

    stats = await import_arxiv_metadata(
        ImportOptions(
            input=arxiv_jsonl,
            categories_json=categories_json,
            groups=("statistics",),
            categories=("cs.CG",),
            dry_run=True,
            progress_every=0,
        ),
        settings=settings,
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )

    assert stats.selected == 2


def test_category_filter_codes_expand_requested_groups(categories_json: Path) -> None:
    catalog = load_category_catalog(categories_json)

    codes = category_filter_codes(
        catalog,
        requested_groups={"computer-science"},
        requested_categories={"stat.ml"},
    )

    assert codes == {"cs.ai", "cs.cg", "stat.ml"}


def test_raw_category_prefilter_matches_exact_codes_case_insensitively() -> None:
    line = b'{"id":"1234.0001","categories":"cs.AI stat.ML","title":"Paper"}'

    assert raw_categories_match_filter(line, {"cs.ai"})
    assert raw_categories_match_filter(line, {"stat.ml"})
    assert not raw_categories_match_filter(line, {"cs.a"})


def test_raw_category_prefilter_handles_escaped_values() -> None:
    line = b'{"id":"1234.0001","categories":"cs.AI\\u0020stat.ML","title":"Paper"}'

    assert raw_categories_match_filter(line, {"stat.ml"})


async def test_unknown_categories_are_imported_and_reported(
    imported_service: tuple[SQLiteStore, MouseionService, Settings],
) -> None:
    _, service, _ = imported_service

    result = await service.search(SearchInput(query="another unique term", top_k=3))
    document = result["results"][0]["document"]

    assert document["source"] == "arxiv:1234.0002"
    assert "arxiv:category:unknown.xy" in document["tags"]
    assert document["metadata"]["unresolved_categories"] == ["unknown.XY"]
