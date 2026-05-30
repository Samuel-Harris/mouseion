from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

import apsw

from mouseion.storage.search_index import backfill_search_fts_batch
from mouseion.storage.stats import STATS_COUNTERS_INITIALISED, refresh_stats_counters_sync

T = TypeVar("T")


class BackgroundStore(Protocol):
    async def get_meta(self, key: str) -> str | None: ...

    async def set_meta(self, key: str, value: str) -> None: ...

    def write(self, fn: Callable[[apsw.Connection], T]) -> Awaitable[T]: ...


async def backfill_stats_counters(store: BackgroundStore) -> None:
    status = await store.get_meta("stats_counters_status")
    if status not in {"pending", "running", "error"}:
        return

    def run(conn: apsw.Connection) -> None:
        _set_meta_sync(conn, "stats_counters_status", "running")
        refresh_stats_counters_sync(conn)
        _set_meta_sync(conn, STATS_COUNTERS_INITIALISED, "true")
        _set_meta_sync(conn, "stats_counters_status", "complete")
        _set_meta_sync(conn, "stats_counters_error", "")

    try:
        await store.write(run)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await store.set_meta("stats_counters_status", "error")
        await store.set_meta("stats_counters_error", str(exc))


async def backfill_search_fts(store: BackgroundStore) -> None:
    status = await store.get_meta("search_fts_backfill_status")
    if status not in {"pending", "running", "error"}:
        return

    try:
        await store.set_meta("search_fts_backfill_status", "running")
        while True:
            inserted = await store.write(backfill_search_fts_batch)
            if inserted == 0:
                await store.set_meta("search_fts_backfill_status", "complete")
                await store.set_meta("search_fts_backfill_error", "")
                return
            current = await store.get_meta("search_fts_backfilled_rows")
            await store.set_meta(
                "search_fts_backfilled_rows",
                str(int(current or "0") + inserted),
            )
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await store.set_meta("search_fts_backfill_status", "error")
        await store.set_meta("search_fts_backfill_error", str(exc))


def _set_meta_sync(conn: apsw.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
