from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from mouseion.config import Settings
from mouseion.domain.models import (
    AddFileInput,
    AddMemoryInput,
    DeleteInput,
    GetDocumentInput,
    ListInput,
    RelateInput,
    SearchFilter,
    SearchInput,
)
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.ingestor import Ingestor
from mouseion.ingest.repo import RepoService
from mouseion.services.exporter import Exporter
from mouseion.services.graph import GraphService
from mouseion.services.search import SearchService
from mouseion.services.service import MouseionService
from mouseion.storage.db import SQLiteStore


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0] * 767 + [1.0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 767 + [1.0] for _ in texts]


@pytest.fixture
async def mouseion_service(tmp_path: Path) -> tuple[SQLiteStore, MouseionService]:
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
        "vectors": "SELECT count(*) AS total FROM chunk_vectors",
        "similar_to": "SELECT count(*) AS total FROM similar_to",
    }
    for query in counts.values():
        result = await store.execute(query)
        assert int(result.first()["total"]) == 0  # type: ignore[index]
