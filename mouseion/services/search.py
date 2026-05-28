from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mouseion.domain.models import SearchFilter
from mouseion.ingest.embedder import Embedder
from mouseion.storage.db import SQLiteStore, _embedding_blob, chunk_from_row, document_from_row


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
        include_graph_neighbours: bool = False,
        filter: SearchFilter | None = None,
    ) -> dict[str, Any]:
        query_vector = await self.embedder.embed(query)
        fusion_k = top_k * 3
        vec_hits = await self._vector_hits(query_vector, fusion_k, filter)
        fts_hits = await self._fts_hits(query, fusion_k, filter)
        fused = rrf_fuse([vec_hits, fts_hits], self.rrf_k)[:top_k]
        results = []
        for chunk_id, score in fused:
            hydrated = await self._hydrate_chunk(chunk_id)
            if hydrated is None:
                continue
            hydrated["score"] = score
            if include_graph_neighbours:
                hydrated["graph_neighbours"] = await self.expand_similar_to(chunk_id, cap=3)
            results.append(hydrated)
        return {"results": results}

    async def _vector_hits(
        self, query_vector: list[float], limit: int, filter: SearchFilter | None
    ) -> list[RankedHit]:
        sql_filter = _sql_filter(filter)
        search_window = await self._vector_search_window(limit, sql_filter)
        if search_window == 0:
            return []
        result = await self.store.execute(
            f"""
            SELECT v.chunk_id,
                   v.distance
            FROM chunk_vectors v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE embedding MATCH ? AND k = ?
            {sql_filter.clause}
            ORDER BY distance
            LIMIT ?
            """,
            (_embedding_blob(query_vector), search_window, *sql_filter.params, limit),
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

    async def _vector_search_window(self, limit: int, sql_filter: SqlFilter) -> int:
        if not sql_filter.active:
            return limit * 4
        result = await self.store.execute("SELECT count(*) AS total FROM chunk_vectors")
        row = result.first()
        return int(row["total"] if row else 0)

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

    async def expand_similar_to(self, chunk_id: int, cap: int) -> list[dict[str, Any]]:
        result = await self.store.execute(
            """
            SELECT n.id AS "c.id",
                   n.document_id AS "c.document_id",
                   n.content AS "c.content",
                   n.chunk_index AS "c.chunk_index",
                   n.token_count AS "c.token_count",
                   n.created_at AS "c.created_at",
                   v.embedding AS "c.embedding",
                   d.id,
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
                   links.score
            FROM (
              SELECT CASE WHEN from_chunk = ? THEN to_chunk ELSE from_chunk END AS other_id,
                     score
              FROM similar_to
              WHERE from_chunk = ? OR to_chunk = ?
            ) links
            JOIN chunks n ON n.id = links.other_id
            JOIN documents d ON d.id = n.document_id
            LEFT JOIN chunk_vectors v ON v.chunk_id = n.id
            ORDER BY links.score DESC
            LIMIT ?
            """,
            (chunk_id, chunk_id, chunk_id, cap),
        )
        neighbours = []
        for row in result.rows:
            document = document_from_row(row)
            chunk = chunk_from_row(row)
            neighbours.append(
                {
                    "chunk_id": str(chunk.id),
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "score": float(row.get("score", 0.0)),
                    "document": document.model_dump(mode="json"),
                }
            )
        return neighbours


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
