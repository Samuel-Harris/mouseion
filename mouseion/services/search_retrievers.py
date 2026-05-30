from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mouseion.domain.models import SearchFilter
from mouseion.services.search_fusion import RankedHit
from mouseion.storage.db import SQLiteStore, embedding_blob
from mouseion.storage.vector import VECTOR_COLUMN, VECTOR_TABLE, vector_scan_function
from mouseion.support.logging_config import get_logger


@dataclass(slots=True)
class SqlFilter:
    clause: str
    params: tuple[Any, ...]

    @property
    def active(self) -> bool:
        return bool(self.clause)


@dataclass(slots=True)
class ExactLexicalRetriever:
    store: SQLiteStore

    async def retrieve(
        self, query: str, limit: int, filter: SearchFilter | None
    ) -> list[RankedHit]:
        sql_filter = _sql_filter(filter)
        normalized = query.lower()
        source_candidates = _source_candidates(normalized)
        placeholders = ",".join("?" for _ in source_candidates)
        result = await self.store.execute(
            f"""
            SELECT c.id AS chunk_id,
                   (
                     CASE WHEN d.title = ? COLLATE NOCASE THEN 30.0 ELSE 0.0 END
                     + CASE WHEN d.source IN ({placeholders}) THEN 35.0 ELSE 0.0 END
                   ) AS lexical_score
            FROM documents d
            JOIN chunks c ON c.document_id = d.id
            WHERE c.chunk_index = 0
              AND (
                d.title = ? COLLATE NOCASE
                OR d.source IN ({placeholders})
              )
              {sql_filter.clause}
            ORDER BY lexical_score DESC, d.updated_at DESC
            LIMIT ?
            """,
            (
                normalized,
                *source_candidates,
                normalized,
                *source_candidates,
                *sql_filter.params,
                limit,
            ),
        )
        return [
            RankedHit(
                chunk_id=int(row["chunk_id"]),
                score=float(row["lexical_score"]),
                rank=index + 1,
                source="lexical",
            )
            for index, row in enumerate(result.rows)
        ]


@dataclass(slots=True)
class FullTextRetriever:
    store: SQLiteStore

    async def retrieve(
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
        background = await self.store.background_status()
        if background["search_fts"]["status"] != "complete":
            return await self._content_only_hits(
                query,
                limit,
                sql_filter,
                search_syntax=search_syntax,
                fts_queries=fts_queries,
            )
        return await self._search_index_hits(
            query,
            limit,
            sql_filter,
            search_syntax=search_syntax,
            fts_queries=fts_queries,
        )

    async def _search_index_hits(
        self,
        query: str,
        limit: int,
        sql_filter: SqlFilter,
        *,
        search_syntax: str,
        fts_queries: list[str],
    ) -> list[RankedHit]:
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
            hits = _ranked_lexical_hits(result.rows)
            if hits or search_syntax == "advanced":
                return hits
        return []

    async def _content_only_hits(
        self,
        query: str,
        limit: int,
        sql_filter: SqlFilter,
        *,
        search_syntax: str,
        fts_queries: list[str],
    ) -> list[RankedHit]:
        for fts_query in fts_queries:
            result = await self.store.execute(
                f"""
                SELECT f.rowid AS chunk_id,
                       bm25(chunks_fts) AS bm25_score,
                       CASE WHEN lower(c.content) LIKE ? ESCAPE '\\' THEN 3.0 ELSE 0.0 END
                         AS field_boost
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ?
                {sql_filter.clause}
                ORDER BY bm25_score - field_boost
                LIMIT ?
                """,
                (
                    _like_contains_pattern(query),
                    fts_query,
                    *sql_filter.params,
                    limit,
                ),
            )
            hits = _ranked_lexical_hits(result.rows)
            if hits or search_syntax == "advanced":
                return hits
        return []


@dataclass(slots=True)
class VectorRetriever:
    store: SQLiteStore

    async def retrieve(
        self, query_vector: list[float], limit: int, filter: SearchFilter | None
    ) -> list[RankedHit]:
        sql_filter = _sql_filter(filter)
        vector_config = await self.store.vector_runtime_config()
        if vector_config.warning is not None:
            get_logger(__name__).warning("vector_quantization_stale", message=vector_config.warning)
        scan_function = vector_scan_function(vector_config.active_mode)
        query_blob = embedding_blob(query_vector)
        if sql_filter.active:
            result = await self.store.execute(
                f"""
                SELECT v.rowid AS chunk_id,
                       v.distance
                FROM {scan_function}('{VECTOR_TABLE}', '{VECTOR_COLUMN}', ?) AS v
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
                FROM {scan_function}('{VECTOR_TABLE}', '{VECTOR_COLUMN}', ?, ?) AS v
                ORDER BY v.distance
                """,
                (query_blob, limit),
            )
        return [
            RankedHit(
                chunk_id=int(row["chunk_id"]),
                score=1.0 - float(row["distance"]),
                rank=index + 1,
                source="semantic",
            )
            for index, row in enumerate(result.rows)
        ]


def _ranked_lexical_hits(rows: list[dict[str, Any]]) -> list[RankedHit]:
    return [
        RankedHit(
            chunk_id=int(row["chunk_id"]),
            score=max(0.0, -float(row["bm25_score"])) + float(row["field_boost"]),
            rank=index + 1,
            source="lexical",
        )
        for index, row in enumerate(rows)
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


def _source_candidates(normalized_query: str) -> list[str]:
    candidates = [normalized_query]
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", normalized_query):
        candidates.append(f"arxiv:{normalized_query}")
    return sorted(set(candidates))


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
