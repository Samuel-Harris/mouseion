from __future__ import annotations

from typing import Any

import apsw

STATS_COUNTERS_INITIALISED = "stats_counters_initialised"
LEGACY_STATS_COUNTERS_INITIALIZED = "stats_counters_initialized"


def ensure_stats_counters_sync(conn: apsw.Connection) -> None:
    for name in ("documents", "chunks", "tags"):
        conn.execute(
            "INSERT OR IGNORE INTO stats_counters(name, value) VALUES(?, 0)",
            (name,),
        )
    if stats_counters_initialised_sync(conn):
        _set_meta_sync(conn, STATS_COUNTERS_INITIALISED, "true")
        return
    has_rows = any(
        _fetch_one(conn, f"SELECT 1 FROM {table} LIMIT 1") is not None
        for table in ("documents", "chunks", "tags")
    )
    if has_rows:
        _set_meta_sync(conn, STATS_COUNTERS_INITIALISED, "false")
        _set_meta_sync(conn, "stats_counters_status", "pending")
    else:
        _set_meta_sync(conn, STATS_COUNTERS_INITIALISED, "true")
        _set_meta_sync(conn, "stats_counters_status", "complete")
        _set_meta_sync(conn, "stats_counters_error", "")


def refresh_stats_counters_sync(conn: apsw.Connection) -> None:
    conn.execute("DELETE FROM stats_counters")
    conn.executemany(
        "INSERT INTO stats_counters(name, value) VALUES(?, ?)",
        [
            ("documents", _count_table_sync(conn, "documents")),
            ("chunks", _count_table_sync(conn, "chunks")),
            ("tags", _count_table_sync(conn, "tags")),
        ],
    )
    conn.execute("DELETE FROM stats_document_types")
    conn.execute(
        """
        INSERT INTO stats_document_types(type, value)
        SELECT type, count(*)
        FROM documents
        GROUP BY type
        """
    )


def stats_snapshot_sync(conn: apsw.Connection) -> dict[str, Any]:
    use_counters = stats_counters_initialised_sync(conn)
    values = _counter_values_sync(conn) if use_counters else _exact_counter_values_sync(conn)
    background = background_status_sync(conn)
    return {
        "documents": values.get("documents", 0),
        "chunks": values.get("chunks", 0),
        "tags": values.get("tags", 0),
        "documents_by_type": _document_type_values_sync(conn, use_counters=use_counters),
        "stats_ready": use_counters,
        "background_tasks": background,
    }


def background_status_sync(conn: apsw.Connection) -> dict[str, Any]:
    meta = {
        str(row["key"]): str(row["value"])
        for row in _rows_from_cursor(
            conn.cursor(),
            """
            SELECT key, value
            FROM meta
            WHERE key IN (
              'stats_counters_status',
              ?,
              ?,
              'stats_counters_error',
              'search_fts_backfill_status',
              'search_fts_backfilled_rows',
              'search_fts_backfill_error'
            )
            """,
            (STATS_COUNTERS_INITIALISED, LEGACY_STATS_COUNTERS_INITIALIZED),
        )
    }
    return {
        "stats_counters": {
            "status": meta.get("stats_counters_status", "unknown"),
            "initialised": _bool_meta(
                meta.get(STATS_COUNTERS_INITIALISED)
                or meta.get(LEGACY_STATS_COUNTERS_INITIALIZED)
            ),
            "error": meta.get("stats_counters_error"),
        },
        "search_fts": {
            "status": meta.get("search_fts_backfill_status", "unknown"),
            "backfilled_rows": int(meta.get("search_fts_backfilled_rows", "0")),
            "error": meta.get("search_fts_backfill_error"),
        },
    }


def stats_counters_initialised_sync(conn: apsw.Connection) -> bool:
    row = _fetch_one(
        conn,
        "SELECT value FROM meta WHERE key IN (?, ?) ORDER BY key LIMIT 1",
        (STATS_COUNTERS_INITIALISED, LEGACY_STATS_COUNTERS_INITIALIZED),
    )
    return row is not None and _bool_meta(str(row["value"]))


def _counter_values_sync(conn: apsw.Connection) -> dict[str, int]:
    return {
        str(row["name"]): int(row["value"])
        for row in _rows_from_cursor(
            conn.cursor(),
            """
            SELECT name, value
            FROM stats_counters
            WHERE name IN ('documents', 'chunks', 'tags')
            """,
        )
    }


def _exact_counter_values_sync(conn: apsw.Connection) -> dict[str, int]:
    return {
        "documents": _count_table_sync(conn, "documents"),
        "chunks": _count_table_sync(conn, "chunks"),
        "tags": _count_table_sync(conn, "tags"),
    }


def _document_type_values_sync(conn: apsw.Connection, *, use_counters: bool) -> dict[str, int]:
    if use_counters:
        return {
            str(row["type"]): int(row["value"])
            for row in _rows_from_cursor(
                conn.cursor(),
                """
                SELECT type, value
                FROM stats_document_types
                WHERE value > 0
                ORDER BY type
                """,
            )
        }
    return {
        str(row["type"]): int(row["total"])
        for row in _rows_from_cursor(
            conn.cursor(),
            """
            SELECT type, count(*) AS total
            FROM documents
            GROUP BY type
            ORDER BY type
            """,
        )
    }


def _count_table_sync(conn: apsw.Connection, table: str) -> int:
    row = _fetch_one(conn, f"SELECT count(*) AS total FROM {table}")
    return int(row["total"] if row else 0)


def _set_meta_sync(conn: apsw.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _fetch_one(
    conn: apsw.Connection, query: str, parameters: tuple[Any, ...] = ()
) -> dict[str, Any] | None:
    rows = _rows_from_cursor(conn.cursor(), query, parameters)
    return rows[0] if rows else None


def _rows_from_cursor(
    cursor: apsw.Cursor, query: str, parameters: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    results = iter(cursor.execute(query, parameters))
    try:
        first = next(results)
    except StopIteration:
        return []
    description = cursor.getdescription() or []
    names = [str(column[0]) for column in description]
    rows = [first, *results]
    return [dict(zip(names, row, strict=False)) for row in rows]


def _bool_meta(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}
