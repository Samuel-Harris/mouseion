from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mouseion.domain.models import SearchFilter
from mouseion.ingest.embedder import Embedder
from mouseion.services.search_fusion import (
    FusedHit,
    HybridFusionPolicy,
    RankedHit,
    rrf_fuse,
)
from mouseion.services.search_retrievers import (
    ExactLexicalRetriever,
    FullTextRetriever,
    VectorRetriever,
)
from mouseion.storage.db import SQLiteStore, chunk_from_row, document_from_row

SEARCH_SNIPPET_CHARS = 700

__all__ = ["RankedHit", "SearchService", "rrf_fuse"]


@dataclass(slots=True)
class SearchService:
    store: SQLiteStore
    embedder: Embedder
    rrf_k: int
    exact_retriever: ExactLexicalRetriever = field(init=False)
    full_text_retriever: FullTextRetriever = field(init=False)
    vector_retriever: VectorRetriever = field(init=False)

    def __post_init__(self) -> None:
        self.exact_retriever = ExactLexicalRetriever(self.store)
        self.full_text_retriever = FullTextRetriever(self.store)
        self.vector_retriever = VectorRetriever(self.store)

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        filter: SearchFilter | None = None,
        search_syntax: str = "plain",
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"results": [], "message": "Enter a search query."}
        fusion_k = min(max(top_k * 8, 40), 200)
        fusion = HybridFusionPolicy(top_k=top_k)

        exact_hits = await self.exact_retriever.retrieve(query, top_k, filter)
        exact_result = fusion.exact_lexical(exact_hits)
        if not exact_result.requires_vector:
            return await self._search_output(exact_result.hits)

        fts_hits = await self.full_text_retriever.retrieve(
            query,
            fusion_k,
            filter,
            search_syntax=search_syntax,
        )
        fast_path_result = fusion.lexical_fast_path(fts_hits)
        if not fast_path_result.requires_vector:
            return await self._search_output(fast_path_result.hits)

        query_vector = await self.embedder.embed(query)
        vec_hits = await self.vector_retriever.retrieve(query_vector, fusion_k, filter)
        fused = fusion.hybrid(fts_hits, vec_hits)
        output = await self._search_output(fused.hits)
        vector_config = await self.store.vector_runtime_config()
        warnings: list[str] = []
        if vector_config.warning is not None:
            warnings.append(vector_config.warning)
        if warnings:
            output["warnings"] = warnings
        return output

    async def _search_output(self, fused: list[FusedHit]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        hydrated_by_id = await self._hydrate_chunks([hit.chunk_id for hit in fused])
        for hit in fused:
            hydrated = hydrated_by_id.get(hit.chunk_id)
            if hydrated is None:
                continue
            hydrated["score"] = hit.score
            hydrated["match"] = hit.match
            results.append(hydrated)
        output: dict[str, Any] = {"results": results}
        if not results:
            output["message"] = "No confident results."
        return output

    async def _hydrate_chunks(self, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        result = await self.store.execute(
            f"""
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
                   c.id AS "c.id",
                   c.document_id AS "c.document_id",
                   c.content AS "c.content",
                   c.chunk_index AS "c.chunk_index",
                   c.token_count AS "c.token_count",
                   c.created_at AS "c.created_at"
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            tuple(chunk_ids),
        )
        hydrated: dict[int, dict[str, Any]] = {}
        for row in result.rows:
            document = document_from_row(row)
            chunk = chunk_from_row(row)
            metadata = _compact_metadata(document.metadata)
            document_payload = {
                "id": str(document.id),
                "type": str(document.type),
                "title": document.title,
                "source": document.source,
                "tags": document.tags,
                "metadata": metadata,
            }
            snippet = _snippet(chunk.content)
            hydrated[chunk.id] = {
                "chunk_id": str(chunk.id),
                "document_id": str(document.id),
                "title": document.title,
                "source": document.source,
                "authors": metadata.get("authors", ""),
                "tags": document.tags,
                "snippet": snippet,
                "content": snippet,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "document": document_payload,
            }
        return hydrated


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "authors",
        "update_date",
        "published",
        "html_url",
        "pdf_url",
        "categories",
        "unresolved_categories",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _snippet(content: str) -> str:
    compact = " ".join(content.split())
    if len(compact) <= SEARCH_SNIPPET_CHARS:
        return compact
    return compact[: SEARCH_SNIPPET_CHARS - 3].rstrip() + "..."
