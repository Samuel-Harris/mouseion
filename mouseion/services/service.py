from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

import apsw

from mouseion.domain.models import (
    AddFileInput,
    AddMemoryInput,
    AddRepoInput,
    AddUrlInput,
    BatchIngestItem,
    ChunkText,
    DeleteInput,
    Document,
    DocumentType,
    GetDocumentInput,
    IngestedContent,
    ListInput,
    RelateInput,
    SearchInput,
    utc_now,
)
from mouseion.errors import DocumentNotFoundError
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.embedder import Embedder
from mouseion.ingest.ingestor import Ingestor
from mouseion.ingest.repo import RepoService
from mouseion.services.exporter import Exporter
from mouseion.services.graph import GraphService
from mouseion.services.search import SearchService
from mouseion.storage.db import DocumentChunkUpsert, SQLiteStore, document_from_row
from mouseion.support.utils import canonical_text_hash, deterministic_edge_id

EdgePolicy = Literal["skip", "incremental", "recompute-after-insert"]


@dataclass(slots=True)
class _PreparedBatchItem:
    content: IngestedContent
    source: str
    content_hash: str
    tags: list[str]
    explicit_chunks: list[ChunkText] | None
    document_id: UUID | None = None
    chunks: list[ChunkText] | None = None


@dataclass(slots=True)
class MouseionService:
    store: SQLiteStore
    ingestor: Ingestor
    chunker: Chunker
    embedder: Embedder
    searcher: SearchService
    graph: GraphService
    repos: RepoService
    exporter: Exporter

    async def add_url(self, input: AddUrlInput) -> dict:
        content = await self.ingestor.fetch_url(str(input.url))
        return await self._ingest(content, input.tags)

    async def add_file(self, input: AddFileInput) -> dict:
        content = await self.ingestor.read_file(input.file_path)
        return await self._ingest(content, input.tags)

    async def add_memory(self, input: AddMemoryInput) -> dict:
        content = self.ingestor.memory(input.content)
        output = await self._ingest(content, input.tags)
        return {
            "memory_id": output["document_id"],
            "chunks_created": output["chunks_created"],
            "edges_created": output["edges_created"],
            "status": output["status"],
            "action": output["action"],
        }

    async def add_repo(self, input: AddRepoInput) -> dict:
        return await self.repos.add_repo(input.repo_url, input.name)

    async def search(self, input: SearchInput) -> dict:
        return await self.searcher.search(
            input.query,
            top_k=input.top_k,
            include_graph_neighbours=input.include_graph_neighbours,
            filter=input.filter,
        )

    async def get_document(self, input: GetDocumentInput) -> dict:
        document = await self.store.get_document(input.document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {input.document_id}")
        chunks = await self.store.get_chunks_for_document(input.document_id)
        related = await self._related_documents(input.document_id)
        return {
            "document": document.model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            "related_documents": related,
        }

    async def list_documents(self, input: ListInput) -> dict:
        items, total = await self.store.list_documents(input.type, input.limit, input.offset)
        return {"items": [item.model_dump(mode="json") for item in items], "total": total}

    async def relate(self, input: RelateInput) -> dict:
        edge_id = deterministic_edge_id(input.from_id, input.to_id, input.label)

        def run(conn: apsw.Connection) -> None:
            conn.execute(
                """
                INSERT OR IGNORE INTO related_to(from_doc, to_doc, label, note, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    str(input.from_id),
                    str(input.to_id),
                    input.label or "",
                    input.note or "",
                    utc_now().isoformat(),
                ),
            )

        await self.store.write(run)
        return {"edge_id": str(edge_id), "status": "ok"}

    async def delete(self, input: DeleteInput) -> dict:
        deleted_chunks, deleted_edges = await self.store.delete_document(input.id)
        return {
            "deleted_chunks": deleted_chunks,
            "deleted_edges": deleted_edges,
            "status": "deleted",
        }

    async def recompute_edges(self) -> dict:
        return await self.graph.recompute_all()

    async def export(self) -> dict:
        return await self.exporter.export()

    async def stats(self) -> dict:
        counts = await self._table_counts(
            {
                "documents": "documents",
                "chunks": "chunks",
                "tags": "tags",
                "related_edges": "related_to",
                "similar_edges": "similar_to",
            }
        )
        document_types = await self.store.execute(
            """
            SELECT type, count(*) AS total
            FROM documents
            GROUP BY type
            ORDER BY type
            """
        )
        related_edges = counts["related_edges"]
        similar_edges = counts["similar_edges"]
        return {
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "tags": counts["tags"],
            "edges": {
                "total": related_edges + similar_edges,
                "related": related_edges,
                "similar": similar_edges,
            },
            "documents_by_type": {
                str(row["type"]): int(row["total"]) for row in document_types.rows
            },
        }

    async def _table_counts(self, tables: dict[str, str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key, table in tables.items():
            result = await self.store.execute(f"SELECT count(*) AS total FROM {table}")
            row = result.first()
            counts[key] = int(row["total"] if row else 0)
        return counts

    async def _ingest(self, content: IngestedContent, tags: list[str]) -> dict:
        output = await self.batch_ingest(
            [BatchIngestItem(content=content, tags=tags)],
            edge_policy="incremental",
        )
        document = output["documents"][0]
        return {
            "document_id": document["document_id"],
            "title": content.title,
            "chunks_created": document["chunks_created"],
            "edges_created": output["edges_created"],
            "status": "ok",
            "action": document["action"],
        }

    async def batch_ingest(
        self,
        items: list[BatchIngestItem],
        *,
        edge_policy: EdgePolicy = "incremental",
        skip_unchanged: bool = False,
        metadata_compare_exclude: set[str] | None = None,
    ) -> dict:
        if not items:
            return _empty_batch_ingest_output(edge_policy)
        if edge_policy not in {"skip", "incremental", "recompute-after-insert"}:
            raise ValueError(f"Unsupported edge policy: {edge_policy}")

        prepared = [_prepare_batch_item(item) for item in items]
        existing = await self.store.find_documents_for_ingest(
            [(item.content.type, item.source, item.content_hash) for item in prepared]
        )

        writable = []
        skipped_documents = []
        for item in prepared:
            identity = _ingest_identity(item.content.type, item.source, item.content_hash)
            existing_document = existing.get((item.content.type, identity))
            if (
                skip_unchanged
                and existing_document is not None
                and _matches_existing(item, existing_document, metadata_compare_exclude or set())
            ):
                skipped_documents.append(
                    {
                        "document_id": str(existing_document.id),
                        "source": existing_document.source,
                        "title": item.content.title,
                        "action": "skipped",
                        "chunks_created": 0,
                    }
                )
                continue
            if item.content.type == DocumentType.MEMORY and existing_document is None:
                item.source = f"memory:{uuid4()}"
            item.document_id = existing_document.id if existing_document else None
            writable.append(item)

        flat_chunk_texts: list[str] = []
        for item in writable:
            item.chunks = item.explicit_chunks or self.chunker.chunk(item.content.content)
            flat_chunk_texts.extend(chunk.content for chunk in item.chunks)
        if not writable:
            return _batch_ingest_output(
                edge_policy=edge_policy,
                documents=skipped_documents,
                inserted=0,
                updated=0,
                skipped=len(skipped_documents),
                chunks_created=0,
                edges_created=0,
            )

        embeddings = await self.embedder.embed_many(flat_chunk_texts)
        offset = 0
        upserts: list[DocumentChunkUpsert] = []
        for item in writable:
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

        upsert_results = await self.store.upsert_documents_with_chunks(upserts)
        document_outputs = [
            {
                "document_id": str(result.document_id),
                "source": result.source,
                "title": item.content.title,
                "action": result.action,
                "chunks_created": result.chunks_created,
            }
            for item, result in zip(writable, upsert_results, strict=True)
        ]
        document_outputs.extend(skipped_documents)

        edges_created = 0
        if edge_policy == "incremental":
            for result in upsert_results:
                edges_created += await self.graph.create_incremental_edges(result.document_id)
        elif edge_policy == "recompute-after-insert" and upsert_results:
            recompute = await self.graph.recompute_all()
            edges_created = int(recompute["edges_created"])

        inserted = sum(1 for result in upsert_results if result.action == "created")
        updated = sum(1 for result in upsert_results if result.action == "replaced")
        skipped = len(skipped_documents)
        return _batch_ingest_output(
            edge_policy=edge_policy,
            documents=document_outputs,
            inserted=inserted,
            updated=updated,
            skipped=skipped,
            chunks_created=sum(result.chunks_created for result in upsert_results),
            edges_created=edges_created,
        )

    async def _related_documents(self, document_id: UUID) -> list[dict]:
        result = await self.store.execute(
            """
            SELECT d.id,
                   d.type,
                   d.title,
                   d.source,
                   d.content_hash,
                   d.created_at,
                   d.updated_at,
                   d.metadata,
                   COALESCE(
                     (
                       SELECT json_group_array(tag)
                       FROM (SELECT tag FROM tags WHERE document_id = d.id ORDER BY tag)
                     ),
                     '[]'
                   ) AS tags,
                   r.label,
                   r.note,
                   r.created_at AS related_created_at
            FROM related_to r
            JOIN documents d
              ON d.id = CASE WHEN r.from_doc = ? THEN r.to_doc ELSE r.from_doc END
            WHERE r.from_doc = ? OR r.to_doc = ?
            ORDER BY d.updated_at DESC
            """,
            (str(document_id), str(document_id), str(document_id)),
        )
        related = []
        for row in result.rows:
            document = document_from_row(row)
            related.append(
                {
                    "document": document.model_dump(mode="json"),
                    "label": row.get("label") or None,
                    "note": row.get("note") or None,
                    "created_at": str(row.get("related_created_at") or ""),
                }
            )
        return related


def _empty_batch_ingest_output(edge_policy: EdgePolicy) -> dict:
    return _batch_ingest_output(
        edge_policy=edge_policy,
        documents=[],
        inserted=0,
        updated=0,
        skipped=0,
        chunks_created=0,
        edges_created=0,
    )


def _batch_ingest_output(
    *,
    edge_policy: EdgePolicy,
    documents: list[dict],
    inserted: int,
    updated: int,
    skipped: int,
    chunks_created: int,
    edges_created: int,
) -> dict:
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
