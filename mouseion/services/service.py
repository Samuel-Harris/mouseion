from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from mouseion.domain.models import (
    AddFileInput,
    AddMemoryInput,
    AddRepoInput,
    AddUrlInput,
    DeleteInput,
    DocumentType,
    GetDocumentInput,
    IngestedContent,
    ListInput,
    SearchInput,
)
from mouseion.errors import DocumentNotFoundError
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.embedder import Embedder
from mouseion.ingest.ingestor import Ingestor
from mouseion.ingest.repo import RepoService
from mouseion.services.exporter import Exporter
from mouseion.services.search import SearchService
from mouseion.storage.db import SQLiteStore
from mouseion.support.utils import canonical_text_hash

JsonDict = dict[str, Any]


@dataclass(slots=True)
class MouseionService:
    store: SQLiteStore
    ingestor: Ingestor
    chunker: Chunker
    embedder: Embedder
    searcher: SearchService
    repos: RepoService
    exporter: Exporter

    async def add_url(self, input: AddUrlInput) -> JsonDict:
        content = await self.ingestor.fetch_url(str(input.url))
        return await self._ingest(content, input.tags)

    async def add_file(self, input: AddFileInput) -> JsonDict:
        content = await self.ingestor.read_file(input.file_path)
        return await self._ingest(content, input.tags)

    async def add_memory(self, input: AddMemoryInput) -> JsonDict:
        content = self.ingestor.memory(input.content)
        output = await self._ingest(content, input.tags)
        return {
            "memory_id": output["document_id"],
            "chunks_created": output["chunks_created"],
            "status": output["status"],
            "action": output["action"],
        }

    async def add_repo(self, input: AddRepoInput) -> JsonDict:
        return await self.repos.add_repo(input.repo_url, input.name)

    async def search(self, input: SearchInput) -> JsonDict:
        return await self.searcher.search(
            input.query,
            top_k=input.top_k,
            filter=input.filter,
            search_syntax=input.search_syntax,
        )

    async def get_document(self, input: GetDocumentInput) -> JsonDict:
        document = await self.store.get_document(input.document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {input.document_id}")
        chunks = await self.store.get_chunks_for_document(input.document_id)
        return {
            "document": document.model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }

    async def list_documents(self, input: ListInput) -> JsonDict:
        items, total = await self.store.list_documents(input.type, input.limit, input.offset)
        return {"items": [item.model_dump(mode="json") for item in items], "total": total}

    async def delete(self, input: DeleteInput) -> JsonDict:
        deleted_chunks = await self.store.delete_document(input.id)
        return {
            "deleted_chunks": deleted_chunks,
            "status": "deleted",
        }

    async def vector_status(self) -> JsonDict:
        return await self.store.vector_status()

    async def set_vector_mode(self, mode: str, qbits: int | None = None) -> JsonDict:
        if mode == "exact":
            return await self.store.set_vector_mode_exact()
        if mode == "quantized":
            return await self.store.set_vector_mode_quantized(
                qbits or self.store.settings.vector_quantization_qbits
            )
        raise ValueError(f"Unsupported vector search mode: {mode}")

    async def quantize_vectors(self, qbits: int, *, preload: bool = False) -> JsonDict:
        return await self.store.quantize_vectors(qbits=qbits, preload=preload)

    async def cleanup_quantized_vectors(self) -> JsonDict:
        return await self.store.cleanup_quantized_vectors()

    async def export(self) -> JsonDict:
        return await self.exporter.export()

    async def stats(self) -> JsonDict:
        return await self.store.stats()

    async def _ingest(self, content: IngestedContent, tags: list[str]) -> JsonDict:
        content_hash = canonical_text_hash(content.content)
        source = content.source
        existing = await self.store.find_document_for_ingest(
            content.type, source=source, content_hash=content_hash
        )
        if content.type == DocumentType.MEMORY and existing is None:
            source = f"memory:{uuid4()}"
        chunks = self.chunker.chunk(content.content)
        embeddings = await self.embedder.embed_many([chunk.content for chunk in chunks])
        chunk_rows = [
            (chunk.content, chunk.token_count, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        document_id, action, chunks_created = await self.store.upsert_document_with_chunks(
            document_id=existing.id if existing else None,
            doc_type=content.type,
            title=content.title,
            source=source,
            content_hash=content_hash,
            tags=tags,
            metadata=content.metadata,
            chunks=chunk_rows,
        )
        return {
            "document_id": str(document_id),
            "title": content.title,
            "chunks_created": chunks_created,
            "status": "ok",
            "action": action,
        }
