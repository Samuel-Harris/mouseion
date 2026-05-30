from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

import apsw

from mouseion.config import Settings
from mouseion.domain.models import utc_now
from mouseion.storage.db import SQLiteStore, VectorRuntimeConfig, embedding_blob
from mouseion.support.logging_config import get_logger


@dataclass(slots=True)
class GraphService:
    store: SQLiteStore
    settings: Settings

    async def create_incremental_edges(self, document_id: UUID) -> int:
        vector_config = await self.store.vector_runtime_config()
        _log_vector_warning(vector_config)
        scan_function = _scan_function(vector_config)

        def write(conn: apsw.Connection) -> int:
            chunks = conn.cursor().execute(
                """
                SELECT c.id, v.embedding
                FROM chunks c
                JOIN chunk_vectors v ON v.chunk_id = c.id
                WHERE c.document_id = ?
                ORDER BY c.chunk_index
                """,
                (str(document_id),),
            )
            edges = 0
            for chunk_id, embedding in chunks:
                edges += self._create_edges_for_chunk_sync(
                    conn, int(chunk_id), embedding, scan_function
                )
            return edges

        return int(await self.store.write(write))

    async def recompute_all(self) -> dict[str, float | int]:
        logger = get_logger(__name__)
        started = perf_counter()
        vector_config = await self.store.vector_runtime_config()
        _log_vector_warning(vector_config)
        scan_function = _scan_function(vector_config)

        def recompute(conn: apsw.Connection) -> tuple[int, int]:
            conn.execute("DELETE FROM similar_to")
            chunks = list(
                conn.cursor().execute(
                    """
                    SELECT c.id, v.embedding
                    FROM chunks c
                    JOIN chunk_vectors v ON v.chunk_id = c.id
                    ORDER BY c.id
                    """
                )
            )
            edge_scores: dict[tuple[int, int], float] = {}
            for chunk_id, embedding in chunks:
                for from_chunk, to_chunk, score in self._candidate_edges_for_chunk_sync(
                    conn, int(chunk_id), embedding, scan_function
                ):
                    # Preserve the previous chunk-id iteration semantics when both directions match.
                    edge_scores.setdefault((from_chunk, to_chunk), score)
            if edge_scores:
                conn.executemany(
                    """
                    INSERT INTO similar_to(from_chunk, to_chunk, score)
                    VALUES(?, ?, ?)
                    """,
                    (
                        (from_chunk, to_chunk, score)
                        for (from_chunk, to_chunk), score in edge_scores.items()
                    ),
                )
            conn.execute(
                """
                INSERT INTO meta(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("last_recompute_at", utc_now().isoformat()),
            )
            return len(chunks), len(edge_scores)

        chunks_processed, edges = await self.store.write(recompute)
        duration = perf_counter() - started
        logger.info(
            "similarity_recompute_complete",
            chunks_processed=chunks_processed,
            edges_created=edges,
            duration_seconds=duration,
        )
        return {
            "chunks_processed": chunks_processed,
            "edges_created": edges,
            "duration_seconds": duration,
        }

    def _create_edges_for_chunk_sync(
        self, conn: apsw.Connection, chunk_id: int, embedding: Any, scan_function: str
    ) -> int:
        created = 0
        for from_chunk, to_chunk, score in self._candidate_edges_for_chunk_sync(
            conn, chunk_id, embedding, scan_function
        ):
            conn.execute(
                """
                INSERT OR IGNORE INTO similar_to(from_chunk, to_chunk, score)
                VALUES(?, ?, ?)
                """,
                (from_chunk, to_chunk, score),
            )
            created += int(conn.changes())
        return created

    def _candidate_edges_for_chunk_sync(
        self, conn: apsw.Connection, chunk_id: int, embedding: Any, scan_function: str
    ) -> list[tuple[int, int, float]]:
        limit = self.settings.similarity_top_k + 1
        rows = conn.cursor().execute(
            f"""
            SELECT rowid, distance
            FROM {scan_function}('chunk_vectors', 'embedding', ?, ?)
            """,
            (embedding_blob(embedding), limit),
        )
        edges: list[tuple[int, int, float]] = []
        candidates = 0
        for row in rows:
            other_id = int(row[0])
            if other_id == chunk_id:
                continue
            score = 1.0 - float(row[1])
            if score >= self.settings.similarity_threshold:
                from_chunk, to_chunk = sorted((chunk_id, other_id))
                edges.append((from_chunk, to_chunk, score))
                candidates += 1
            if candidates >= self.settings.similarity_top_k:
                break
        return edges


def _scan_function(config: VectorRuntimeConfig) -> str:
    return "vector_quantize_scan" if config.active_mode == "quantized" else "vector_full_scan"


def _log_vector_warning(config: VectorRuntimeConfig) -> None:
    if config.warning is not None:
        get_logger(__name__).warning("vector_quantization_stale", message=config.warning)
