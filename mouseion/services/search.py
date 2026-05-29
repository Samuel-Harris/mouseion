from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mouseion.domain.models import SearchFilter
from mouseion.ingest.embedder import Embedder
from mouseion.storage.db import SQLiteStore, chunk_from_row, document_from_row, embedding_blob


@dataclass(slots=True)
class RankedHit:
    chunk_id: int
    score: float
    rank: int
    source: str = ""


@dataclass(slots=True)
class FusedHit:
    chunk_id: int
    score: float
    match: str


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
        search_syntax: str = "plain",
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"results": [], "message": "Enter a search query."}
        fusion_k = min(max(top_k * 8, 40), 200)
        fts_hits = await self._fts_hits(query, fusion_k, filter, search_syntax=search_syntax)
        query_vector = await self.embedder.embed(query)
        vec_hits = await self._vector_hits(query_vector, fusion_k, filter)
        fused = hybrid_fuse(fts_hits, vec_hits, top_k=top_k)
        results: list[dict[str, Any]] = []
        for hit in fused:
            hydrated = await self._hydrate_chunk(hit.chunk_id)
            if hydrated is None:
                continue
            hydrated["score"] = hit.score
            hydrated["match"] = hit.match
            if include_graph_neighbours:
                hydrated["graph_neighbours"] = await self.expand_similar_to(hit.chunk_id, cap=3)
            results.append(hydrated)
        output: dict[str, Any] = {"results": results}
        if not results:
            output["message"] = "No confident results."
        return output

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
            (embedding_blob(query_vector), search_window, *sql_filter.params, limit),
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
        self,
        query: str,
        limit: int,
        filter: SearchFilter | None,
        *,
        search_syntax: str = "plain",
    ) -> list[RankedHit]:
        sql_filter = _sql_filter(filter)
        fts_queries = _fts_queries(query, search_syntax)
        if not fts_queries:
            return []

        for fts_query in fts_queries:
            result = await self.store.execute(
                f"""
                SELECT f.rowid AS chunk_id,
                       bm25(search_fts, 5.0, 4.0, 3.0, 2.0, 1.0) AS bm25_score,
                       (
                         CASE WHEN lower(d.title) = ? THEN 20.0 ELSE 0.0 END
                         + CASE WHEN lower(d.source) = ? THEN 20.0 ELSE 0.0 END
                         + CASE WHEN lower(d.title) LIKE ? ESCAPE '\\' THEN 8.0 ELSE 0.0 END
                         + CASE WHEN lower(d.source) LIKE ? ESCAPE '\\' THEN 10.0 ELSE 0.0 END
                         + CASE WHEN lower(c.content) LIKE ? ESCAPE '\\' THEN 3.0 ELSE 0.0 END
                         + CASE
                             WHEN EXISTS (
                               SELECT 1
                               FROM tags exact_tag
                               WHERE exact_tag.document_id = d.id
                                 AND exact_tag.tag = ?
                             )
                             THEN 8.0
                             ELSE 0.0
                           END
                         + CASE
                             WHEN EXISTS (
                               SELECT 1
                               FROM tags partial_tag
                               WHERE partial_tag.document_id = d.id
                                 AND partial_tag.tag LIKE ? ESCAPE '\\'
                             )
                             THEN 4.0
                             ELSE 0.0
                           END
                       ) AS field_boost
                FROM search_fts f
                JOIN chunks c ON c.id = f.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE search_fts MATCH ?
                {sql_filter.clause}
                ORDER BY bm25_score - field_boost
                LIMIT ?
                """,
                (
                    query.lower(),
                    query.lower(),
                    _like_contains_pattern(query),
                    _like_contains_pattern(query),
                    _like_contains_pattern(query),
                    query.lower(),
                    _like_contains_pattern(query),
                    fts_query,
                    *sql_filter.params,
                    limit,
                ),
            )
            hits = [
                RankedHit(
                    chunk_id=int(row["chunk_id"]),
                    score=max(0.0, -float(row["bm25_score"])) + float(row["field_boost"]),
                    rank=index + 1,
                    source="lexical",
                )
                for index, row in enumerate(result.rows)
            ]
            if hits or search_syntax == "advanced":
                return hits
        return []

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
        neighbours: list[dict[str, Any]] = []
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


def hybrid_fuse(
    lexical_hits: list[RankedHit],
    vector_hits: list[RankedHit],
    *,
    top_k: int,
) -> list[FusedHit]:
    scores: dict[int, float] = {}
    sources: dict[int, set[str]] = {}

    for hit in lexical_hits:
        lexical_score = 2.0 + min(hit.score, 30.0) / 8.0 + 1.0 / hit.rank
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + lexical_score
        sources.setdefault(hit.chunk_id, set()).add("lexical")

    for hit in vector_hits:
        if not lexical_hits and hit.score < 0.78:
            continue
        semantic_score = max(0.0, hit.score - 0.70) * 2.0 + 0.25 / hit.rank
        if semantic_score <= 0:
            continue
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + semantic_score
        sources.setdefault(hit.chunk_id, set()).add("semantic")

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            item[1],
            "lexical" in sources.get(item[0], set()),
            item[0] * -1,
        ),
        reverse=True,
    )
    return [
        FusedHit(chunk_id=chunk_id, score=score, match="+".join(sorted(sources[chunk_id])))
        for chunk_id, score in ranked[:top_k]
    ]


def _fts_queries(query: str, search_syntax: str) -> list[str]:
    if search_syntax == "advanced":
        return [query]

    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        return []
    quoted_terms = [f'"{term}"' for term in terms]
    and_query = " ".join(quoted_terms)
    if len(quoted_terms) == 1:
        return [and_query]
    return [and_query, " OR ".join(quoted_terms)]


def _like_contains_pattern(query: str) -> str:
    escaped = query.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


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
