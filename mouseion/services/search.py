from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mouseion.domain.models import SearchFilter
from mouseion.ingest.embedder import Embedder
from mouseion.storage.db import SQLiteStore, chunk_from_row, document_from_row, embedding_blob
from mouseion.support.logging_config import get_logger


@dataclass(slots=True)
class RankedHit:
    chunk_id: int
    score: float
    rank: int


@dataclass(slots=True)
class SqlFilter:
    clause: str
    params: tuple[Any, ...]

    @property
    def active(self) -> bool:
        return bool(self.clause)


@dataclass(slots=True)
class SearchService:
    store: SQLiteStore
    embedder: Embedder
    rrf_k: int

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        filter: SearchFilter | None = None,
    ) -> dict[str, Any]:
        query_vector = await self.embedder.embed(query)
        fusion_k = top_k * 3
        vec_hits = await self._vector_hits(query_vector, fusion_k, filter)
        fts_hits = await self._fts_hits(query, fusion_k, filter)
        fused = rrf_fuse([vec_hits, fts_hits], self.rrf_k)[:top_k]
        vector_config = await self.store.vector_runtime_config()
        warnings: list[str] = []
        if vector_config.warning is not None:
            warnings.append(vector_config.warning)
        results: list[dict[str, Any]] = []
        for chunk_id, score in fused:
            hydrated = await self._hydrate_chunk(chunk_id)
            if hydrated is None:
                continue
            hydrated["score"] = score
            results.append(hydrated)
        output: dict[str, Any] = {"results": results}
        if warnings:
            output["warnings"] = warnings
        return output

    async def _vector_hits(
        self, query_vector: list[float], limit: int, filter: SearchFilter | None
    ) -> list[RankedHit]:
        sql_filter = _sql_filter(filter)
        vector_config = await self.store.vector_runtime_config()
        if vector_config.warning is not None:
            get_logger(__name__).warning("vector_quantization_stale", message=vector_config.warning)
        scan_function = (
            "vector_quantize_scan"
            if vector_config.active_mode == "quantized"
            else "vector_full_scan"
        )
        query_blob = embedding_blob(query_vector)
        if sql_filter.active:
            result = await self.store.execute(
                f"""
                SELECT v.rowid AS chunk_id,
                       v.distance
                FROM {scan_function}('chunk_vectors', 'embedding', ?) AS v
                JOIN chunks c ON c.id = v.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE 1 = 1
                {sql_filter.clause}
                ORDER BY v.distance
                LIMIT ?
                """,
                (query_blob, *sql_filter.params, limit),
            )
        else:
            result = await self.store.execute(
                f"""
                SELECT v.rowid AS chunk_id,
                       v.distance
                FROM {scan_function}('chunk_vectors', 'embedding', ?, ?) AS v
                ORDER BY v.distance
                """,
                (query_blob, limit),
            )
        return [
            RankedHit(
                chunk_id=int(row["chunk_id"]),
                score=1.0 - float(row["distance"]),
                rank=index + 1,
            )
            for index, row in enumerate(result.rows)
        ]

    async def _fts_hits(
        self, query: str, limit: int, filter: SearchFilter | None
    ) -> list[RankedHit]:
        sql_filter = _sql_filter(filter)
        result = await self.store.execute(
            f"""
            SELECT f.rowid AS chunk_id,
                   bm25(chunks_fts) AS score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
            {sql_filter.clause}
            ORDER BY score
            LIMIT ?
            """,
            (query, *sql_filter.params, limit),
        )
        return [
            RankedHit(chunk_id=int(row["chunk_id"]), score=float(row["score"]), rank=index + 1)
            for index, row in enumerate(result.rows)
        ]

    async def _hydrate_chunk(self, chunk_id: int) -> dict[str, Any] | None:
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
                   c.id AS "c.id",
                   c.document_id AS "c.document_id",
                   c.content AS "c.content",
                   c.chunk_index AS "c.chunk_index",
                   c.token_count AS "c.token_count",
                   c.created_at AS "c.created_at",
                   v.embedding AS "c.embedding"
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
            WHERE c.id = ?
            """,
            (chunk_id,),
        )
        row = result.first()
        if row is None:
            return None
        document = document_from_row(row)
        chunk = chunk_from_row(row)
        return {
            "chunk_id": str(chunk.id),
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "document": document.model_dump(mode="json"),
        }

def rrf_fuse(hit_lists: list[list[RankedHit]], k: int) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for hits in hit_lists:
        for hit in hits:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _sql_filter(filter: SearchFilter | None) -> SqlFilter:
    if filter is None:
        return SqlFilter("", ())
    clauses: list[str] = []
    params: list[Any] = []
    if filter.type is not None:
        clauses.append("d.type = ?")
        params.append(str(filter.type))
    for index, tag in enumerate(filter.tags or []):
        alias = f"filter_tag_{index}"
        clauses.append(
            f"EXISTS (SELECT 1 FROM tags {alias} WHERE {alias}.document_id = d.id "
            f"AND {alias}.tag = ?)"
        )
        params.append(tag)
    if not clauses:
        return SqlFilter("", ())
    return SqlFilter("AND " + " AND ".join(clauses), tuple(params))
