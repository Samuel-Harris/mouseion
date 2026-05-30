from __future__ import annotations

from typing import Any, cast

import apsw

from mouseion.support.utils import json_loads


def ensure_search_fts_backfill_status_sync(conn: apsw.Connection) -> None:
    row = _fetch_one(
        conn,
        "SELECT value FROM meta WHERE key = ?",
        ("search_fts_backfill_status",),
    )
    if row is not None and str(row["value"]) == "running":
        _set_meta_sync(conn, "search_fts_backfill_status", "pending")
        return
    if row is not None:
        return
    if _fetch_one(conn, "SELECT 1 FROM chunks LIMIT 1") is None:
        _set_meta_sync(conn, "search_fts_backfill_status", "complete")
    else:
        _set_meta_sync(conn, "search_fts_backfill_status", "pending")
    _set_meta_sync(conn, "search_fts_backfilled_rows", "0")
    _set_meta_sync(conn, "search_fts_backfill_error", "")


def backfill_search_fts_batch(conn: apsw.Connection, batch_size: int = 1000) -> int:
    rows = _rows_from_cursor(
        conn.cursor(),
        """
        SELECT c.id,
               d.title,
               d.source,
               COALESCE(
                 (
                   SELECT group_concat(tag, ' ')
                   FROM tags
                   WHERE document_id = d.id
                 ),
                 ''
               ) AS tags,
               COALESCE(d.metadata, '') AS metadata,
               c.content
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        LEFT JOIN search_fts f ON f.rowid = c.id
        WHERE f.rowid IS NULL
        ORDER BY c.id
        LIMIT ?
        """,
        (batch_size,),
    )
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO search_fts(rowid, title, source, tags, metadata, content)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        [
            (
                int(row["id"]),
                str(row["title"]),
                str(row["source"]),
                str(row["tags"]),
                metadata_search_text(json_loads(row["metadata"])),
                str(row["content"]),
            )
            for row in rows
        ],
    )
    return len(rows)


def insert_search_index_row_sync(
    conn: apsw.Connection,
    *,
    chunk_id: int,
    title: str,
    source: str,
    tags: list[str],
    metadata: dict[str, Any],
    content: str,
) -> None:
    conn.execute(
        """
        INSERT INTO search_fts(rowid, title, source, tags, metadata, content)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, title, source, " ".join(tags), metadata_search_text(metadata), content),
    )


def metadata_search_text(value: Any) -> str:
    parts: list[str] = []

    def collect(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str | int | float | bool):
            parts.append(str(item))
            return
        if isinstance(item, dict):
            for key, nested in cast(dict[Any, Any], item).items():
                parts.append(str(key))
                collect(nested)
            return
        if isinstance(item, list | tuple):
            for nested in cast(list[Any] | tuple[Any, ...], item):
                collect(nested)
            return
        parts.append(str(item))

    collect(value)
    return " ".join(parts)


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
