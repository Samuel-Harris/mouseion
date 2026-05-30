from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mouseion.domain.models import SearchFilter
from mouseion.ingest.embedder import Embedder
from mouseion.storage.db import SQLiteStore, chunk_from_row, document_from_row, embedding_blob
from mouseion.storage.vector import VECTOR_COLUMN, VECTOR_TABLE, vector_scan_function
from mouseion.support.logging_config import get_logger

LEXICAL_FAST_PATH_SCORE = 18.0
SEARCH_SNIPPET_CHARS = 700


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
        filter: SearchFilter | None = None,
        search_syntax: str = "plain",
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"results": [], "message": "Enter a search query."}
        fusion_k = min(max(top_k * 8, 40), 200)
        exact_hits = await self._exact_lexical_hits(query, top_k, filter)
        if exact_hits:
            return await self._search_output(
                [FusedHit(hit.chunk_id, hit.score, "lexical") for hit in exact_hits]
            )
        fts_hits = await self._fts_hits(query, fusion_k, filter, search_syntax=search_syntax)
        if fts_hits and fts_hits[0].score >= LEXICAL_FAST_PATH_SCORE:
            return await self._search_output(
                [
                    FusedHit(hit.chunk_id, hit.score, "lexical")
                    for hit in fts_hits[:top_k]
                ]
            )
        query_vector = await self.embedder.embed(query)
        vec_hits = await self._vector_hits(query_vector, fusion_k, filter)
        fused = hybrid_fuse(fts_hits, vec_hits, top_k=top_k)
        output = await self._search_output(fused)
        vector_config = await self.store.vector_runtime_config()
        warnings: list[str] = []
        if vector_config.warning is not None:
            warnings.append(vector_config.warning)
        if warnings:
            output["warnings"] = warnings
        return output

    async def _search_output(self, fused: list[FusedHit]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for hit in fused:
            hydrated = await self._hydrate_chunk(hit.chunk_id)
            if hydrated is None:
                continue
            hydrated["score"] = hit.score
            hydrated["match"] = hit.match
            results.append(hydrated)
        output: dict[str, Any] = {"results": results}
        if not results:
            output["message"] = "No confident results."
        return output

    async def _exact_lexical_hits(
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

    async def _vector_hits(
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
        background = await self.store.background_status()
        if background["search_fts"]["status"] != "complete":
            return await self._content_fts_hits(
                query,
                limit,
                filter,
                search_syntax=search_syntax,
                fts_queries=fts_queries,
            )

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

    async def _content_fts_hits(
        self,
        query: str,
        limit: int,
        filter: SearchFilter | None,
        *,
        search_syntax: str,
        fts_queries: list[str],
    ) -> list[RankedHit]:
        sql_filter = _sql_filter(filter)
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
                   c.created_at AS "c.created_at"
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (chunk_id,),
        )
        row = result.first()
        if row is None:
            return None
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
        return {
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


def _source_candidates(normalized_query: str) -> list[str]:
    candidates = [normalized_query]
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", normalized_query):
        candidates.append(f"arxiv:{normalized_query}")
    return sorted(set(candidates))


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
