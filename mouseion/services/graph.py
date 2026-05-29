from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

import apsw

from mouseion.config import Settings
from mouseion.domain.models import utc_now
from mouseion.storage.db import SQLiteStore, embedding_blob
from mouseion.support.logging_config import get_logger


@dataclass(slots=True)
class GraphService:
    store: SQLiteStore
    settings: Settings

    async def create_incremental_edges(self, document_id: UUID) -> int:
        chunks = await self.store.get_chunk_embeddings_for_document(document_id)
        edges = 0
        for row in chunks:
            edges += await self._create_edges_for_chunk(int(row["id"]), row["embedding"])
        return edges

    async def recompute_all(self) -> dict[str, float | int]:
        logger = get_logger(__name__)
        started = perf_counter()

        def clear(conn: apsw.Connection) -> None:
            conn.execute("DELETE FROM similar_to")

        await self.store.write(clear)
        chunks = await self.store.get_all_chunk_embeddings()
        edges = 0
        for row in chunks:
            edges += await self._create_edges_for_chunk(int(row["id"]), row["embedding"])
        await self.store.set_meta("last_recompute_at", utc_now().isoformat())
        duration = perf_counter() - started
        logger.info(
            "similarity_recompute_complete",
            chunks_processed=len(chunks),
            edges_created=edges,
            duration_seconds=duration,
        )
        return {
            "chunks_processed": len(chunks),
            "edges_created": edges,
            "duration_seconds": duration,
        }

    async def _create_edges_for_chunk(self, chunk_id: int, embedding: Any) -> int:
        limit = self.settings.similarity_top_k + 1
        result = await self.store.execute(
            """
            SELECT chunk_id, distance
            FROM chunk_vectors
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (embedding_blob(embedding), limit),
        )
        candidates: list[tuple[int, float]] = []
        for row in result.rows:
            other_id = int(row["chunk_id"])
            if other_id == chunk_id:
                continue
            score = 1.0 - float(row["distance"])
            if score >= self.settings.similarity_threshold:
                candidates.append((other_id, score))
            if len(candidates) >= self.settings.similarity_top_k:
                break

        def write(conn: apsw.Connection) -> int:
            created = 0
            for other_id, score in candidates:
                from_chunk, to_chunk = sorted((chunk_id, other_id))
                exists = _edge_exists(conn, from_chunk, to_chunk)
                if exists:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO similar_to(from_chunk, to_chunk, score)
                    VALUES(?, ?, ?)
                    """,
                    (from_chunk, to_chunk, score),
                )
                created += 1
            return created

        return int(await self.store.write(write))


def _edge_exists(conn: apsw.Connection, from_chunk: int, to_chunk: int) -> bool:
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT 1 FROM similar_to WHERE from_chunk = ? AND to_chunk = ? LIMIT 1",
        (from_chunk, to_chunk),
    )
    return next(iter(rows), None) is not None
