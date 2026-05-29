from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from mouseion.domain.models import (
    ChunkText,
    Document,
    DocumentType,
    IngestedContent,
    normalize_tags,
)
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.embedder import Embedder
from mouseion.services.graph import GraphService
from mouseion.storage.db import DocumentChunkUpsert, SQLiteStore
from mouseion.support.utils import canonical_text_hash

EdgePolicy = Literal["skip", "incremental", "recompute-after-insert"]
_BatchAction = Literal["created", "replaced"]
JsonDict = dict[str, Any]


class BatchIngestItem(BaseModel):
    content: IngestedContent
    tags: list[str] = Field(default_factory=list)
    chunks: list[ChunkText] | None = None

    @field_validator("tags")
    @classmethod
    def normalize_batch_tags(cls, value: list[str]) -> list[str]:
        return normalize_tags(value)


@dataclass(slots=True)
class BulkIngestService:
    store: SQLiteStore
    chunker: Chunker
    embedder: Embedder
    graph: GraphService

    async def preview(
        self,
        items: list[BatchIngestItem],
        *,
        skip_unchanged: bool = True,
        metadata_compare_exclude: set[str] | None = None,
    ) -> JsonDict:
        plan = await self._plan(
            items,
            skip_unchanged=skip_unchanged,
            metadata_compare_exclude=metadata_compare_exclude or set(),
        )
        return _batch_ingest_output(
            edge_policy="skip",
            documents=[_planned_document_output(item, 0) for item in plan.writable]
            + plan.skipped_documents,
            inserted=sum(1 for item in plan.writable if item.action == "created"),
            updated=sum(1 for item in plan.writable if item.action == "replaced"),
            skipped=len(plan.skipped_documents),
            chunks_created=0,
            edges_created=0,
        )

    async def ingest(
        self,
        items: list[BatchIngestItem],
        *,
        edge_policy: EdgePolicy = "incremental",
        skip_unchanged: bool = False,
        metadata_compare_exclude: set[str] | None = None,
    ) -> JsonDict:
        if edge_policy not in {"skip", "incremental", "recompute-after-insert"}:
            raise ValueError(f"Unsupported edge policy: {edge_policy}")

        plan = await self._plan(
            items,
            skip_unchanged=skip_unchanged,
            metadata_compare_exclude=metadata_compare_exclude or set(),
        )
        if not plan.writable:
            return _batch_ingest_output(
                edge_policy=edge_policy,
                documents=plan.skipped_documents,
                inserted=0,
                updated=0,
                skipped=len(plan.skipped_documents),
                chunks_created=0,
                edges_created=0,
            )

        upserts = await self._prepare_upserts(plan.writable)
        upsert_results = await self.store.upsert_documents_with_chunks(upserts)
        document_outputs = [
            {
                "document_id": str(result.document_id),
                "source": result.source,
                "title": item.content.title,
                "action": result.action,
                "chunks_created": result.chunks_created,
            }
            for item, result in zip(plan.writable, upsert_results, strict=True)
        ]
        document_outputs.extend(plan.skipped_documents)

        edges_created = 0
        if edge_policy == "incremental":
            for result in upsert_results:
                edges_created += await self.graph.create_incremental_edges(result.document_id)
        elif edge_policy == "recompute-after-insert":
            recompute = await self.graph.recompute_all()
            edges_created = int(recompute["edges_created"])

        return _batch_ingest_output(
            edge_policy=edge_policy,
            documents=document_outputs,
            inserted=sum(1 for result in upsert_results if result.action == "created"),
            updated=sum(1 for result in upsert_results if result.action == "replaced"),
            skipped=len(plan.skipped_documents),
            chunks_created=sum(result.chunks_created for result in upsert_results),
            edges_created=edges_created,
        )

    async def _plan(
        self,
        items: list[BatchIngestItem],
        *,
        skip_unchanged: bool,
        metadata_compare_exclude: set[str],
    ) -> _BatchPlan:
        if not items:
            return _BatchPlan(writable=[], skipped_documents=[])

        prepared = [_prepare_batch_item(item) for item in items]
        existing = await self.store.find_documents_for_ingest(
            [(item.content.type, item.source, item.content_hash) for item in prepared]
        )

        writable: list[_PreparedBatchItem] = []
        skipped_documents: list[JsonDict] = []
        planned_identities: set[tuple[DocumentType, str]] = set()
        for item in prepared:
            identity = _ingest_identity(item.content.type, item.source, item.content_hash)
            existing_document = existing.get((item.content.type, identity))
            if (
                skip_unchanged
                and existing_document is not None
                and _matches_existing(item, existing_document, metadata_compare_exclude)
            ):
                skipped_documents.append(_skipped_document_output(item, existing_document))
                continue

            if item.content.type == DocumentType.MEMORY and existing_document is None:
                item.source = f"memory:{uuid4()}"
            item.document_id = existing_document.id if existing_document else None
            identity_key = (item.content.type, identity)
            item.action = (
                "replaced"
                if existing_document is not None or identity_key in planned_identities
                else "created"
            )
            planned_identities.add(identity_key)
            writable.append(item)
        return _BatchPlan(writable=writable, skipped_documents=skipped_documents)

    async def _prepare_upserts(self, items: list[_PreparedBatchItem]) -> list[DocumentChunkUpsert]:
        flat_chunk_texts: list[str] = []
        for item in items:
            item.chunks = item.explicit_chunks or self.chunker.chunk(item.content.content)
            flat_chunk_texts.extend(chunk.content for chunk in item.chunks)

        embeddings = await self.embedder.embed_many(flat_chunk_texts)
        offset = 0
        upserts: list[DocumentChunkUpsert] = []
        for item in items:
            if item.chunks is None:
                raise RuntimeError("batch ingest item was not chunked")
            count = len(item.chunks)
            chunk_embeddings = embeddings[offset : offset + count]
            offset += count
            upserts.append(
                DocumentChunkUpsert(
                    document_id=item.document_id,
                    doc_type=item.content.type,
                    title=item.content.title,
                    source=item.source,
                    content_hash=item.content_hash,
                    tags=item.tags,
                    metadata=item.content.metadata,
                    chunks=[
                        (chunk.content, chunk.token_count, embedding)
                        for chunk, embedding in zip(item.chunks, chunk_embeddings, strict=True)
                    ],
                )
            )
        return upserts


@dataclass(slots=True)
class _PreparedBatchItem:
    content: IngestedContent
    source: str
    content_hash: str
    tags: list[str]
    explicit_chunks: list[ChunkText] | None
    document_id: UUID | None = None
    action: _BatchAction = "created"
    chunks: list[ChunkText] | None = None


@dataclass(slots=True)
class _BatchPlan:
    writable: list[_PreparedBatchItem]
    skipped_documents: list[JsonDict]


def _prepare_batch_item(item: BatchIngestItem) -> _PreparedBatchItem:
    return _PreparedBatchItem(
        content=item.content,
        source=item.content.source,
        content_hash=canonical_text_hash(item.content.content),
        tags=item.tags,
        explicit_chunks=item.chunks,
    )


def _ingest_identity(doc_type: DocumentType, source: str, content_hash: str) -> str:
    return content_hash if doc_type == DocumentType.MEMORY else source


def _matches_existing(
    item: _PreparedBatchItem, document: Document, metadata_compare_exclude: set[str]
) -> bool:
    return (
        item.content_hash == document.content_hash
        and sorted(item.tags) == sorted(document.tags)
        and _comparable_metadata(item.content.metadata, metadata_compare_exclude)
        == _comparable_metadata(document.metadata, metadata_compare_exclude)
    )


def _comparable_metadata(metadata: dict[str, object], excluded_keys: set[str]) -> dict[str, object]:
    return {key: value for key, value in metadata.items() if key not in excluded_keys}


def _skipped_document_output(item: _PreparedBatchItem, document: Document) -> JsonDict:
    return {
        "document_id": str(document.id),
        "source": document.source,
        "title": item.content.title,
        "action": "skipped",
        "chunks_created": 0,
    }


def _planned_document_output(item: _PreparedBatchItem, chunks_created: int) -> JsonDict:
    return {
        "document_id": str(item.document_id) if item.document_id else "",
        "source": item.source,
        "title": item.content.title,
        "action": item.action,
        "chunks_created": chunks_created,
    }


def _batch_ingest_output(
    *,
    edge_policy: EdgePolicy,
    documents: list[JsonDict],
    inserted: int,
    updated: int,
    skipped: int,
    chunks_created: int,
    edges_created: int,
) -> JsonDict:
    return {
        "status": "ok",
        "edge_policy": edge_policy,
        "documents": documents,
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "chunks_created": chunks_created,
        "edges_created": edges_created,
    }
