from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import apsw

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
from mouseion.storage.db import SQLiteStore, document_from_row
from mouseion.support.utils import canonical_text_hash, deterministic_edge_id

JsonDict = dict[str, Any]


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
            "edges_created": output["edges_created"],
            "status": output["status"],
            "action": output["action"],
        }

    async def add_repo(self, input: AddRepoInput) -> JsonDict:
        return await self.repos.add_repo(input.repo_url, input.name)

    async def search(self, input: SearchInput) -> JsonDict:
        return await self.searcher.search(
            input.query,
            top_k=input.top_k,
            include_graph_neighbours=input.include_graph_neighbours,
            filter=input.filter,
        )

    async def get_document(self, input: GetDocumentInput) -> JsonDict:
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

    async def list_documents(self, input: ListInput) -> JsonDict:
        items, total = await self.store.list_documents(input.type, input.limit, input.offset)
        return {"items": [item.model_dump(mode="json") for item in items], "total": total}

    async def relate(self, input: RelateInput) -> JsonDict:
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

    async def delete(self, input: DeleteInput) -> JsonDict:
        deleted_chunks, deleted_edges = await self.store.delete_document(input.id)
        return {
            "deleted_chunks": deleted_chunks,
            "deleted_edges": deleted_edges,
            "status": "deleted",
        }

    async def recompute_edges(self) -> JsonDict:
        return await self.graph.recompute_all()

    async def export(self) -> JsonDict:
        return await self.exporter.export()

    async def stats(self) -> JsonDict:
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
        edges_created = await self.graph.create_incremental_edges(document_id)
        return {
            "document_id": str(document_id),
            "title": content.title,
            "chunks_created": chunks_created,
            "edges_created": edges_created,
            "status": "ok",
            "action": action,
        }

    async def _related_documents(self, document_id: UUID) -> list[JsonDict]:
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
        related: list[JsonDict] = []
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
