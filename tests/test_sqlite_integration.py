from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import apsw
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
    SearchFilter,
    SearchInput,
)
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.ingestor import Ingestor
from mouseion.ingest.repo import RepoService
from mouseion.services.bulk_ingest import BatchIngestItem, BulkIngestService
from mouseion.services.exporter import Exporter
from mouseion.services.search import SearchService
from mouseion.services.service import MouseionService
from mouseion.storage.db import SQLiteStore, embedding_blob


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
    service = MouseionService(
        store,
        Ingestor(settings),
        Chunker(settings),
        embedder,  # type: ignore[arg-type]
        SearchService(store, embedder, settings.rrf_k),  # type: ignore[arg-type]
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


async def test_fresh_database_uses_sqlite_vector_blob_table(tmp_path: Path) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    store = SQLiteStore(settings)
    await store.open()
    try:
        schema = await store.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = 'chunk_vectors'
            """
        )
        assert "CREATE VIRTUAL TABLE" not in str(schema.first()["sql"])  # type: ignore[index]

        def insert(conn: apsw.Connection) -> None:
            conn.execute(
                """
                INSERT INTO documents(id, type, title, source, content_hash, created_at, updated_at)
                VALUES('doc', 'memory', 'Doc', 'memory:doc', 'hash', '2026-01-01', '2026-01-01')
                """
            )
            conn.execute(
                """
                INSERT INTO chunks(id, document_id, content, chunk_index, token_count, created_at)
                VALUES(1, 'doc', 'vector test', 0, 2, '2026-01-01')
                """
            )
            conn.execute(
                "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES(?, ?)",
                (1, embedding_blob([0.0] * 767 + [1.0])),
            )

        await store.write(insert)
        hits = await store.execute(
            """
            SELECT rowid, distance
            FROM vector_full_scan('chunk_vectors', 'embedding', ?, 1)
            """,
            (embedding_blob([0.0] * 767 + [1.0]),),
        )

        assert hits.rows == [{"rowid": 1, "distance": 0.0}]
    finally:
        await store.close()


async def test_fresh_database_does_not_create_graph_tables(tmp_path: Path) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    store = SQLiteStore(settings)
    await store.open()
    try:
        tables = await store.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN ('similar_to', 'related_to')
            ORDER BY name
            """
        )
    finally:
        await store.close()

    assert tables.rows == []


async def test_opening_existing_database_does_not_drop_graph_tables(tmp_path: Path) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    settings.ensure_directories()
    conn = apsw.Connection(str(settings.sqlite_path))
    try:
        conn.execute("CREATE TABLE similar_to(from_chunk INTEGER, to_chunk INTEGER)")
        conn.execute("CREATE TABLE related_to(from_doc TEXT, to_doc TEXT)")
        conn.execute("INSERT INTO similar_to(from_chunk, to_chunk) VALUES(1, 2)")
        conn.execute("INSERT INTO related_to(from_doc, to_doc) VALUES('1', '2')")
    finally:
        conn.close()

    store = SQLiteStore(settings)
    await store.open()
    try:
        tables = await store.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN ('similar_to', 'related_to')
            ORDER BY name
            """
        )
    finally:
        await store.close()

    assert tables.rows == [{"name": "related_to"}, {"name": "similar_to"}]


async def test_memory_replacement_preserves_document_identity(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    first = await service.add_memory(
        AddMemoryInput(content="Stable memory one for replacement.", tags=["old"])
    )

    replacement = await service.add_memory(
        AddMemoryInput(content="Stable memory one for replacement.", tags=["new"])
    )
    document = await service.get_document(
        GetDocumentInput(document_id=UUID(replacement["memory_id"]))
    )

    assert replacement["memory_id"] == first["memory_id"]
    assert "related_documents" not in document
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
    bulk_ingest = BulkIngestService(service.store, service.chunker, service.embedder)
    await bulk_ingest.ingest(
        [
            _batch_item(
                "arxiv:2411.18944",
                "Pose estimation content without the identifier.",
                title="Waterfall Transformer for Multi-person Pose Estimation",
            )
        ]
    )

    result = await service.search(SearchInput(query="2411.18944", top_k=3))

    assert result["results"][0]["document"]["source"] == "arxiv:2411.18944"


async def test_search_indexes_metadata_and_tags(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    bulk_ingest = BulkIngestService(service.store, service.chunker, service.embedder)
    await bulk_ingest.ingest(
        [
            _batch_item(
                "metadata:paper",
                "Body text does not contain the author name.",
                tags=["arxiv:category:cs.ir"],
                metadata={"authors": "Navin Ranjan", "categories": "cs.IR"},
            )
        ]
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
    service = MouseionService(
        store,
        Ingestor(settings),
        Chunker(settings),
        embedder,  # type: ignore[arg-type]
        SearchService(store, embedder, settings.rrf_k),  # type: ignore[arg-type]
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


async def test_exact_search_returns_memory_file_and_url_hits(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    for doc_type, source, content in (
        (DocumentType.MEMORY, "memory:exact", "exact memory needle"),
        (DocumentType.FILE, "file:///tmp/exact.txt", "exact file needle"),
        (DocumentType.URL, "https://example.test/exact", "exact url needle"),
    ):
        await store.upsert_document_with_chunks(
            document_id=None,
            doc_type=doc_type,
            title=source,
            source=source,
            content_hash=f"hash:{source}",
            tags=[],
            metadata={},
            chunks=[(content, 3, [0.0] * 767 + [1.0])],
        )

    for doc_type in (DocumentType.MEMORY, DocumentType.FILE, DocumentType.URL):
        result = await service.search(
            SearchInput(
                query="exact needle",
                top_k=1,
                filter=SearchFilter(type=doc_type),
            )
        )

        assert len(result["results"]) == 1
        assert result["results"][0]["document"]["type"] == doc_type


async def test_quantized_vector_modes_support_turbo_qbits(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    await service.add_memory(AddMemoryInput(content="quantized qbits memory", tags=[]))

    for qbits in (2, 3, 4):
        status = await store.set_vector_mode_quantized(qbits)
        result = await service.search(SearchInput(query="quantized", top_k=1))

        assert status["configured_mode"] == "quantized"
        assert status["effective_mode"] == "quantized"
        assert status["configured_qbits"] == qbits
        assert status["dirty"] is False
        assert status["quantized_rows"] == 1
        assert len(result["results"]) == 1


async def test_quantized_exact_switch_preserves_quantized_data(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, _ = mouseion_service
    await store.upsert_document_with_chunks(
        document_id=None,
        doc_type=DocumentType.MEMORY,
        title="Switch",
        source="memory:switch",
        content_hash="switch",
        tags=[],
        metadata={},
        chunks=[("switch vector", 2, [0.0] * 767 + [1.0])],
    )
    await store.set_vector_mode_quantized(4)

    exact = await store.set_vector_mode_exact()
    quantized = await store.set_vector_mode_quantized(4)

    assert exact["effective_mode"] == "exact"
    assert exact["quantized_available"] is True
    assert exact["dirty"] is False
    assert quantized["effective_mode"] == "quantized"
    assert quantized["quantized_available"] is True


async def test_environment_vector_mode_overrides_db_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    store = SQLiteStore(settings)
    await store.open()
    try:
        await store.set_vector_mode_quantized(4)
    finally:
        await store.close()

    monkeypatch.setenv("MOUSEION_VECTOR_SEARCH_MODE", "exact")
    override_settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
    )
    override_store = SQLiteStore(override_settings)
    await override_store.open()
    try:
        status = await override_store.vector_status()
    finally:
        await override_store.close()

    assert status["configured_mode"] == "exact"
    assert status["effective_mode"] == "exact"
    assert status["quantized_available"] is True


async def test_dirty_quantized_mode_falls_back_to_exact_with_warning(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    await service.add_memory(AddMemoryInput(content="clean quantized memory", tags=[]))
    await store.set_vector_mode_quantized(4)

    await service.add_memory(AddMemoryInput(content="dirty quantized memory", tags=[]))
    status = await store.vector_status()
    result = await service.search(SearchInput(query="dirty", top_k=2))

    assert status["configured_mode"] == "quantized"
    assert status["effective_mode"] == "exact"
    assert status["dirty"] is True
    assert "falling back to exact" in str(status["warning"])
    assert result["warnings"] == [status["warning"]]
    assert len(result["results"]) >= 1


async def test_vector_cleanup_marks_quantized_data_unavailable(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    await service.add_memory(AddMemoryInput(content="cleanup quantized memory", tags=[]))
    await store.set_vector_mode_quantized(4)

    status = await store.cleanup_quantized_vectors()

    assert status["configured_mode"] == "quantized"
    assert status["effective_mode"] == "exact"
    assert status["quantized_available"] is False
    assert status["quantized_rows"] == 0
    assert "falling back to exact" in str(status["warning"])


async def test_service_stats_include_documents_chunks_tags_and_types(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    _, service = mouseion_service
    await service.add_memory(AddMemoryInput(content="first stats document", tags=["stats"]))
    await service.add_memory(AddMemoryInput(content="second stats document", tags=[]))

    stats = await service.stats()

    assert stats["documents"] == 2
    assert stats["documents_by_type"] == {"memory": 2}
    assert stats["chunks"] == 2
    assert stats["tags"] == 1
    assert "edges" not in stats


async def test_batch_ingest_batches_embeddings(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    bulk_ingest = BulkIngestService(store, service.chunker, service.embedder)
    embedder = service.embedder  # type: ignore[assignment]
    embedder.calls = []  # type: ignore[attr-defined]

    output = await bulk_ingest.ingest(
        [
            _batch_item("batch:one", "First batch document exactphrase", tags=["bulk"]),
            _batch_item("batch:two", "Second batch document otherphrase", tags=["bulk"]),
        ]
    )

    assert output["inserted"] == 2
    assert output["updated"] == 0
    assert "edges_created" not in output
    assert "edge_policy" not in output
    assert embedder.calls == [  # type: ignore[attr-defined]
        ["First batch document exactphrase", "Second batch document otherphrase"]
    ]


async def test_batch_ingest_enforces_source_identity_inside_write_transaction(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    bulk_ingest = BulkIngestService(store, service.chunker, service.embedder)

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
        ]
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


async def test_batch_ingest_skips_unchanged_documents_without_embedding(
    mouseion_service: tuple[SQLiteStore, MouseionService],
) -> None:
    store, service = mouseion_service
    bulk_ingest = BulkIngestService(store, service.chunker, service.embedder)
    item = _batch_item("batch:stable", "Stable unchanged batch document")
    await bulk_ingest.ingest([item])
    embedder = service.embedder  # type: ignore[assignment]
    embedder.calls = []  # type: ignore[attr-defined]

    output = await bulk_ingest.ingest([item], skip_unchanged=True)

    assert output["inserted"] == 0
    assert output["updated"] == 0
    assert output["skipped"] == 1
    assert embedder.calls == []  # type: ignore[attr-defined]


async def test_delete_cascades_chunks_fts_and_vectors(
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
