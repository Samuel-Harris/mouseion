from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from mouseion.config import Settings
from mouseion.domain.models import (
    AddFileInput,
    AddMemoryInput,
    ChunkText,
    DeleteInput,
    DocumentType,
    GetDocumentInput,
    IngestedContent,
    ListInput,
    RelateInput,
    SearchFilter,
    SearchInput,
)
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.ingestor import Ingestor
from mouseion.ingest.repo import RepoService
from mouseion.services.bulk_ingest import BatchIngestItem, BulkIngestService
from mouseion.services.exporter import Exporter
from mouseion.services.graph import GraphService
from mouseion.services.search import SearchService
from mouseion.services.service import MouseionService
from mouseion.storage.db import SQLiteStore


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 767 + [1.0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.0] * 767 + [1.0] for _ in texts]


class OrthogonalSearchEmbedder(FakeEmbedder):
    async def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 767


@pytest.fixture
async def mouseion_service(tmp_path: Path) -> AsyncIterator[tuple[SQLiteStore, MouseionService]]:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    store = SQLiteStore(settings)
    await store.open()
    embedder = FakeEmbedder()
    graph = GraphService(store, settings)
    service = MouseionService(
        store,
        Ingestor(settings),
        Chunker(settings),
        embedder,  # type: ignore[arg-type]
        SearchService(store, embedder, settings.rrf_k),  # type: ignore[arg-type]
        graph,
        RepoService(settings),
        Exporter(settings, store),
    )
    try:
        yield store, service
    finally:
        await store.close()


async def test_memory_ingest_is_immediately_searchable(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    await service.add_memory(
        AddMemoryInput(
            content="This exact phrase should be found by full text search.",
            tags=["test"],
        )
    )

    result = await service.search(SearchInput(query="exact phrase", top_k=3))

    assert len(result["results"]) == 1


async def test_memory_replacement_preserves_related_edges(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    first = await service.add_memory(
        AddMemoryInput(content="Stable memory one for replacement.", tags=["old"])
    )
    second = await service.add_memory(AddMemoryInput(content="Stable memory two.", tags=["other"]))
    await service.relate(
        RelateInput(
            from_id=UUID(first["memory_id"]),
            to_id=UUID(second["memory_id"]),
            label="related",
        )
    )

    replacement = await service.add_memory(
        AddMemoryInput(content="Stable memory one for replacement.", tags=["new"])
    )
    document = await service.get_document(
        GetDocumentInput(document_id=UUID(replacement["memory_id"]))
    )

    assert replacement["memory_id"] == first["memory_id"]
    assert len(document["related_documents"]) == 1
    assert document["document"]["tags"] == ["new"]


async def test_same_content_in_different_ingest_types_creates_separate_documents(
    tmp_path: Path,
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    content = "Shared content should not collapse across document types."
    memory = await service.add_memory(AddMemoryInput(content=content, tags=["memory"]))
    file_path = tmp_path / "shared.txt"
    file_path.write_text(content, encoding="utf-8")

    file = await service.add_file(AddFileInput(file_path=str(file_path), tags=["file"]))
    documents = await service.list_documents(ListInput(type="all"))

    assert file["document_id"] != memory["memory_id"]
    assert documents["total"] == 2


async def test_search_filter_is_applied_before_candidate_limit(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    for index in range(20):
        await service.add_memory(AddMemoryInput(content=f"needle untagged {index}", tags=[]))
    await service.add_memory(AddMemoryInput(content="needle tagged result", tags=["target"]))

    result = await service.search(
        SearchInput(query="needle", top_k=1, filter=SearchFilter(tags=["target"]))
    )

    assert len(result["results"]) == 1
    assert result["results"][0]["document"]["tags"] == ["target"]


async def test_exact_lexical_match_beats_unhelpful_vector_rank(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    await service.add_memory(AddMemoryInput(content="Irrelevant first memory.", tags=[]))
    await service.add_memory(
        AddMemoryInput(content="Waterfall Transformer exact title and phrase.", tags=[])
    )

    result = await service.search(SearchInput(query="Waterfall Transformer", top_k=2))

    assert result["results"][0]["content"] == "Waterfall Transformer exact title and phrase."
    assert "lexical" in result["results"][0]["match"]


async def test_plain_search_handles_punctuation_and_source_metadata(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    bulk_ingest = BulkIngestService(service.store, service.chunker, service.embedder, service.graph)
    await bulk_ingest.ingest(
        [
            _batch_item(
                "arxiv:2411.18944",
                "Pose estimation content without the identifier.",
                title="Waterfall Transformer for Multi-person Pose Estimation",
            )
        ],
        edge_policy="skip",
    )

    result = await service.search(SearchInput(query="2411.18944", top_k=3))

    assert result["results"][0]["document"]["source"] == "arxiv:2411.18944"


async def test_search_indexes_metadata_and_tags(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    bulk_ingest = BulkIngestService(service.store, service.chunker, service.embedder, service.graph)
    await bulk_ingest.ingest(
        [
            _batch_item(
                "metadata:paper",
                "Body text does not contain the author name.",
                tags=["arxiv:category:cs.ir"],
                metadata={"authors": "Navin Ranjan", "categories": "cs.IR"},
            )
        ],
        edge_policy="skip",
    )

    author_result = await service.search(SearchInput(query="Navin Ranjan", top_k=3))
    category_result = await service.search(SearchInput(query="cs.IR", top_k=3))

    assert author_result["results"][0]["document"]["source"] == "metadata:paper"
    assert category_result["results"][0]["document"]["source"] == "metadata:paper"


async def test_vector_only_search_requires_confident_similarity(tmp_path: Path) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    store = SQLiteStore(settings)
    await store.open()
    embedder = OrthogonalSearchEmbedder()
    graph = GraphService(store, settings)
    service = MouseionService(
        store,
        Ingestor(settings),
        Chunker(settings),
        embedder,  # type: ignore[arg-type]
        SearchService(store, embedder, settings.rrf_k),  # type: ignore[arg-type]
        graph,
        RepoService(settings),
        Exporter(settings, store),
    )
    try:
        await service.add_memory(AddMemoryInput(content="Known stored memory.", tags=[]))
        result = await service.search(SearchInput(query="unrelatedzzzz", top_k=3))
    finally:
        await store.close()

    assert result["results"] == []
    assert result["message"] == "No confident results."


async def test_similarity_edges_are_stored_once_as_canonical_pairs(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    await service.add_memory(AddMemoryInput(content="first similarity document", tags=[]))
    await service.add_memory(AddMemoryInput(content="second similarity document", tags=[]))

    recompute = await service.recompute_edges()
    rows = (await store.execute("SELECT from_chunk, to_chunk FROM similar_to")).rows

    assert recompute["edges_created"] == 1
    assert rows == [{"from_chunk": 1, "to_chunk": 2}]


async def test_service_stats_include_documents_chunks_and_edges(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    first = await service.add_memory(AddMemoryInput(content="first stats document", tags=["stats"]))
    second = await service.add_memory(AddMemoryInput(content="second stats document", tags=[]))
    await service.relate(
        RelateInput(from_id=UUID(first["memory_id"]), to_id=UUID(second["memory_id"]))
    )
    await service.recompute_edges()

    stats = await service.stats()

    assert stats["documents"] == 2
    assert stats["documents_by_type"] == {"memory": 2}
    assert stats["chunks"] == 2
    assert stats["tags"] == 1
    assert stats["edges"] == {"total": 2, "related": 1, "similar": 1}


async def test_batch_ingest_batches_embeddings_and_can_skip_edges(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    bulk_ingest = BulkIngestService(store, service.chunker, service.embedder, service.graph)
    embedder = service.embedder  # type: ignore[assignment]
    embedder.calls = []  # type: ignore[attr-defined]

    output = await bulk_ingest.ingest(
        [
            _batch_item("batch:one", "First batch document exactphrase", tags=["bulk"]),
            _batch_item("batch:two", "Second batch document otherphrase", tags=["bulk"]),
        ],
        edge_policy="skip",
    )
    similar_edges = await store.execute("SELECT count(*) AS total FROM similar_to")

    assert output["inserted"] == 2
    assert output["updated"] == 0
    assert output["edges_created"] == 0
    assert embedder.calls == [  # type: ignore[attr-defined]
        ["First batch document exactphrase", "Second batch document otherphrase"]
    ]
    assert int(similar_edges.first()["total"]) == 0  # type: ignore[index]


async def test_batch_ingest_enforces_source_identity_inside_write_transaction(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    bulk_ingest = BulkIngestService(store, service.chunker, service.embedder, service.graph)

    preview = await bulk_ingest.preview(
        [
            _batch_item("batch:dupe", "First duplicate batch document"),
            _batch_item("batch:dupe", "Second duplicate batch document"),
        ]
    )
    output = await bulk_ingest.ingest(
        [
            _batch_item("batch:dupe", "First duplicate batch document"),
            _batch_item("batch:dupe", "Second duplicate batch document"),
        ],
        edge_policy="skip",
    )
    documents = await store.execute(
        "SELECT count(*) AS total FROM documents WHERE type = ? AND source = ?",
        (str(DocumentType.DOCUMENT), "batch:dupe"),
    )
    chunks = await store.execute(
        """
        SELECT c.content
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.source = ?
        """,
        ("batch:dupe",),
    )

    assert preview["inserted"] == 1
    assert preview["updated"] == 1
    assert output["inserted"] == 1
    assert output["updated"] == 1
    assert int(documents.first()["total"]) == 1  # type: ignore[index]
    assert chunks.rows == [{"content": "Second duplicate batch document"}]


async def test_batch_ingest_recomputes_edges_after_insert(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    bulk_ingest = BulkIngestService(store, service.chunker, service.embedder, service.graph)

    output = await bulk_ingest.ingest(
        [
            _batch_item("batch:one", "First recompute batch document"),
            _batch_item("batch:two", "Second recompute batch document"),
        ],
        edge_policy="recompute-after-insert",
    )
    similar_edges = await store.execute("SELECT count(*) AS total FROM similar_to")

    assert output["inserted"] == 2
    assert output["edges_created"] == 1
    assert int(similar_edges.first()["total"]) == 1  # type: ignore[index]


async def test_batch_ingest_skips_unchanged_documents_without_embedding(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    bulk_ingest = BulkIngestService(store, service.chunker, service.embedder, service.graph)
    item = _batch_item("batch:stable", "Stable unchanged batch document")
    await bulk_ingest.ingest([item], edge_policy="skip")
    embedder = service.embedder  # type: ignore[assignment]
    embedder.calls = []  # type: ignore[attr-defined]

    output = await bulk_ingest.ingest([item], edge_policy="skip", skip_unchanged=True)

    assert output["inserted"] == 0
    assert output["updated"] == 0
    assert output["skipped"] == 1
    assert embedder.calls == []  # type: ignore[attr-defined]


async def test_delete_cascades_chunks_fts_vectors_and_edges(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    added = await service.add_memory(
        AddMemoryInput(content="Delete cascade phrase should disappear.", tags=["delete"])
    )
    before = await store.execute("SELECT count(*) AS total FROM chunk_vectors")
    assert int(before.first()["total"]) == 1  # type: ignore[index]

    deleted = await service.delete(DeleteInput(id=UUID(added["memory_id"])))

    assert deleted["deleted_chunks"] == 1
    counts = {
        "chunks": "SELECT count(*) AS total FROM chunks",
        "fts": "SELECT count(*) AS total FROM chunks_fts",
        "search_fts": "SELECT count(*) AS total FROM search_fts",
        "vectors": "SELECT count(*) AS total FROM chunk_vectors",
        "similar_to": "SELECT count(*) AS total FROM similar_to",
    }
    for query in counts.values():
        result = await store.execute(query)
        assert int(result.first()["total"]) == 0  # type: ignore[index]


def _batch_item(
    source: str,
    content: str,
    tags: list[str] | None = None,
    *,
    title: str | None = None,
    metadata: dict[str, object] | None = None,
) -> BatchIngestItem:
    return BatchIngestItem(
        content=IngestedContent(
            title=title or source,
            source=source,
            type=DocumentType.DOCUMENT,
            content=content,
            metadata=metadata or {"source": source},
        ),
        tags=tags or [],
        chunks=[ChunkText(content=content, token_count=len(content.split()))],
    )
