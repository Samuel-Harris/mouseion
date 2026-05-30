from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from struct import pack, unpack
from typing import Any, TypeVar, cast
from uuid import UUID, uuid4

import apsw
from anyio.to_thread import run_sync

from mouseion.config import Settings
from mouseion.domain.models import Chunk, Document, DocumentType, utc_now
from mouseion.storage.vector import SQLiteVectorBackend, VectorRuntimeConfig
from mouseion.support.utils import json_dumps, json_loads

SCHEMA_VERSION = "3"
T = TypeVar("T")


def _to_db_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _from_db_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


@dataclass(slots=True)
class QueryResult:
    rows: list[dict[str, Any]]

    def first(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


@dataclass(slots=True)
class DocumentChunkUpsert:
    document_id: UUID | None
    doc_type: DocumentType
    title: str
    source: str
    content_hash: str
    tags: list[str]
    metadata: dict[str, Any]
    chunks: list[tuple[str, int, list[float]]]


@dataclass(frozen=True, slots=True)
class DocumentChunkUpsertResult:
    document_id: UUID
    source: str
    action: str
    chunks_created: int


class SQLiteStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_backend = SQLiteVectorBackend(settings)
        self.database: apsw.Connection | None = None
        self._readonly = False
        self._write_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def open(self) -> None:
        self.settings.ensure_directories()
        await run_sync(self._open_sync)
        await self.bootstrap()
        self.start_background_tasks()

    async def open_readonly(self) -> None:
        await run_sync(self._open_sync, True)

    def _open_sync(self, readonly: bool = False) -> None:
        self._readonly = readonly
        self.database = self._connect_sync(readonly=readonly)

    def _connect_sync(self, *, readonly: bool) -> apsw.Connection:
        flags = (
            apsw.SQLITE_OPEN_READONLY
            if readonly
            else apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE
        )
        conn = apsw.Connection(str(self.settings.sqlite_path), flags=flags)
        self.vector_backend.load_extension(conn)
        if not readonly:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA recursive_triggers=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        if readonly:
            self.vector_backend.initialize_if_present_sync(conn)
        return conn

    async def close(self) -> None:
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*self._background_tasks)
        self._background_tasks.clear()
        if self.database is not None:
            self.database.close()
        self.database = None

    def connection(self) -> apsw.Connection:
        if self.database is None:
            raise RuntimeError("SQLite database is not open")
        return self.database

    async def execute(
        self, query: str, parameters: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None
    ) -> QueryResult:
        return await run_sync(self._execute_sync, query, parameters or ())

    def _execute_sync(
        self, query: str, parameters: dict[str, Any] | tuple[Any, ...] | list[Any]
    ) -> QueryResult:
        conn = self._connect_sync(readonly=True)
        try:
            return QueryResult(_rows_from_cursor(conn.cursor(), query, parameters))
        finally:
            conn.close()

    async def write(self, fn: Callable[[apsw.Connection], T]) -> T:
        if self._readonly:
            raise RuntimeError("SQLite database was opened read-only")
        async with self._write_lock:
            return await run_sync(self._write_sync, fn)

    def _write_sync(self, fn: Callable[[apsw.Connection], T]) -> T:
        conn = self.connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = fn(conn)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return result

    async def bootstrap(self) -> None:
        def run(conn: apsw.Connection) -> None:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            conn.execute("DELETE FROM meta WHERE key = ?", ("last_recompute_at",))
            self.vector_backend.ensure_bootstrap_meta_sync(conn)
            _ensure_stats_counters_sync(conn)
            _ensure_search_fts_backfill_status_sync(conn)
            conn.execute(
                """
                INSERT INTO meta(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("schema_version", SCHEMA_VERSION),
            )
            self.vector_backend.initialize_sync(conn)

        await self.write(run)

    def start_background_tasks(self) -> None:
        if self._readonly:
            return
        for runner in (self._backfill_stats_counters, self._backfill_search_fts):
            task = asyncio.create_task(runner())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def background_status(self) -> dict[str, Any]:
        result = await self.execute(
            """
            SELECT key, value
            FROM meta
            WHERE key IN (
              'stats_counters_status',
              'stats_counters_initialized',
              'stats_counters_error',
              'search_fts_backfill_status',
              'search_fts_backfilled_rows',
              'search_fts_backfill_error'
            )
            """
        )
        meta = {str(row["key"]): str(row["value"]) for row in result.rows}
        return {
            "stats_counters": {
                "status": meta.get("stats_counters_status", "unknown"),
                "initialized": meta.get("stats_counters_initialized") == "true",
                "error": meta.get("stats_counters_error"),
            },
            "search_fts": {
                "status": meta.get("search_fts_backfill_status", "unknown"),
                "backfilled_rows": int(meta.get("search_fts_backfilled_rows", "0")),
                "error": meta.get("search_fts_backfill_error"),
            },
        }

    async def _backfill_stats_counters(self) -> None:
        status = await self.get_meta("stats_counters_status")
        if status not in {"pending", "running", "error"}:
            return

        def run(conn: apsw.Connection) -> None:
            _set_meta_sync(conn, "stats_counters_status", "running")
            _refresh_stats_counters_sync(conn)
            _set_meta_sync(conn, "stats_counters_initialized", "true")
            _set_meta_sync(conn, "stats_counters_status", "complete")
            _set_meta_sync(conn, "stats_counters_error", "")

        try:
            await self.write(run)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.set_meta("stats_counters_status", "error")
            await self.set_meta("stats_counters_error", str(exc))

    async def _backfill_search_fts(self) -> None:
        status = await self.get_meta("search_fts_backfill_status")
        if status not in {"pending", "running", "error"}:
            return

        try:
            await self.set_meta("search_fts_backfill_status", "running")
            while True:
                inserted = await self.write(_backfill_search_fts_batch)
                if inserted == 0:
                    await self.set_meta("search_fts_backfill_status", "complete")
                    await self.set_meta("search_fts_backfill_error", "")
                    return
                current = await self.get_meta("search_fts_backfilled_rows")
                await self.set_meta(
                    "search_fts_backfilled_rows",
                    str(int(current or "0") + inserted),
                )
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.set_meta("search_fts_backfill_status", "error")
            await self.set_meta("search_fts_backfill_error", str(exc))

    def _ensure_meta_sync(self, conn: apsw.Connection, key: str, value: str) -> None:
        row = _fetch_one(conn, "SELECT value FROM meta WHERE key = ?", (key,))
        if row is None:
            conn.execute("INSERT INTO meta(key, value) VALUES(?, ?)", (key, value))

    async def get_meta(self, key: str) -> str | None:
        result = await self.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = result.first()
        return None if row is None else str(row["value"])

    async def set_meta(self, key: str, value: str) -> None:
        def run(conn: apsw.Connection) -> None:
            conn.execute(
                """
                INSERT INTO meta(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

        await self.write(run)

    async def vector_runtime_config(self) -> VectorRuntimeConfig:
        return await self.vector_backend.runtime_config(self.execute)

    async def vector_status(self) -> dict[str, Any]:
        return await self.vector_backend.status(self.execute)

    async def stats(self) -> dict[str, Any]:
        counters = await self.execute(
            """
            SELECT name, value
            FROM stats_counters
            WHERE name IN ('documents', 'chunks', 'tags')
            """
        )
        documents_by_type = await self.execute(
            """
            SELECT type, value
            FROM stats_document_types
            WHERE value > 0
            ORDER BY type
            """
        )
        values = {str(row["name"]): int(row["value"]) for row in counters.rows}
        background = await self.background_status()
        return {
            "documents": values.get("documents", 0),
            "chunks": values.get("chunks", 0),
            "tags": values.get("tags", 0),
            "documents_by_type": {
                str(row["type"]): int(row["value"]) for row in documents_by_type.rows
            },
            "stats_ready": background["stats_counters"]["initialized"],
            "background_tasks": background,
        }

    async def vector_quantize_memory(self, qbits: int) -> int:
        return await self.vector_backend.quantize_memory(self.execute, qbits)

    async def set_vector_mode_exact(self) -> dict[str, Any]:
        return await self.vector_backend.set_mode_exact(self.write, self.execute)

    async def set_vector_mode_quantized(self, qbits: int) -> dict[str, Any]:
        return await self.vector_backend.set_mode_quantized(self.write, self.execute, qbits)

    async def quantize_vectors(self, *, qbits: int, preload: bool = False) -> dict[str, Any]:
        return await self.vector_backend.quantize(
            self.write,
            self.execute,
            qbits=qbits,
            preload=preload,
        )

    async def cleanup_quantized_vectors(self) -> dict[str, Any]:
        return await self.vector_backend.cleanup(self.write, self.execute)

    async def find_document_for_ingest(
        self, doc_type: DocumentType, source: str, content_hash: str
    ) -> Document | None:
        identity_column = "content_hash" if doc_type == DocumentType.MEMORY else "source"
        identity_value = content_hash if doc_type == DocumentType.MEMORY else source
        result = await self.execute(
            f"""
            {_document_select_sql()}
            WHERE d.type = ? AND d.{identity_column} = ?
            LIMIT 1
            """,
            (str(doc_type), identity_value),
        )
        row = result.first()
        return None if row is None else document_from_row(row)

    async def find_documents_for_ingest(
        self, identities: list[tuple[DocumentType, str, str]]
    ) -> dict[tuple[DocumentType, str], Document]:
        found: dict[tuple[DocumentType, str], Document] = {}
        values_by_type: dict[DocumentType, set[str]] = {}
        for doc_type, source, content_hash in identities:
            identity_value = content_hash if doc_type == DocumentType.MEMORY else source
            values_by_type.setdefault(doc_type, set()).add(identity_value)

        for doc_type, identity_values in values_by_type.items():
            identity_column = "content_hash" if doc_type == DocumentType.MEMORY else "source"
            for batch in _batched(sorted(identity_values), 900):
                placeholders = ",".join("?" for _ in batch)
                result = await self.execute(
                    f"""
                    {_document_select_sql()}
                    WHERE d.type = ? AND d.{identity_column} IN ({placeholders})
                    """,
                    (str(doc_type), *batch),
                )
                for row in result.rows:
                    document = document_from_row(row)
                    key_value = (
                        document.content_hash
                        if document.type == DocumentType.MEMORY
                        else document.source
                    )
                    found[(document.type, key_value)] = document
        return found

    async def get_document(self, document_id: UUID) -> Document | None:
        result = await self.execute(
            f"{_document_select_sql()} WHERE d.id = ?",
            (str(document_id),),
        )
        row = result.first()
        return None if row is None else document_from_row(row)

    async def list_documents(
        self, type_filter: str = "all", limit: int = 50, offset: int = 0
    ) -> tuple[list[Document], int]:
        params: list[Any] = []
        where = ""
        if type_filter != "all":
            where = "WHERE d.type = ?"
            params.append(type_filter)
        items = await self.execute(
            f"""
            {_document_select_sql()}
            {where}
            ORDER BY d.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        )
        total = await self.execute(
            f"SELECT count(*) AS total FROM documents d {where}",
            tuple(params),
        )
        total_row = total.first()
        return [document_from_row(row) for row in items.rows], int(
            total_row["total"] if total_row else 0
        )

    async def get_chunks_for_document(self, document_id: UUID) -> list[Chunk]:
        result = await self.execute(
            """
            SELECT c.id,
                   c.document_id,
                   c.content,
                   c.chunk_index,
                   c.token_count,
                   c.created_at,
                   v.embedding AS embedding
            FROM chunks c
            LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
            WHERE c.document_id = ?
            ORDER BY c.chunk_index
            """,
            (str(document_id),),
        )
        return [chunk_from_row(row) for row in result.rows]

    async def get_chunk_embeddings_for_document(self, document_id: UUID) -> list[dict[str, Any]]:
        return (
            await self.execute(
                """
                SELECT c.id, v.embedding
                FROM chunks c
                JOIN chunk_vectors v ON v.chunk_id = c.id
                WHERE c.document_id = ?
                ORDER BY c.chunk_index
                """,
                (str(document_id),),
            )
        ).rows

    async def get_all_chunk_embeddings(self) -> list[dict[str, Any]]:
        return (
            await self.execute(
                """
                SELECT c.id, v.embedding
                FROM chunks c
                JOIN chunk_vectors v ON v.chunk_id = c.id
                ORDER BY c.id
                """
            )
        ).rows

    async def upsert_document_with_chunks(
        self,
        *,
        document_id: UUID | None,
        doc_type: DocumentType,
        title: str,
        source: str,
        content_hash: str,
        tags: list[str],
        metadata: dict[str, Any],
        chunks: list[tuple[str, int, list[float]]],
    ) -> tuple[UUID, str, int]:
        results = await self.upsert_documents_with_chunks(
            [
                DocumentChunkUpsert(
                    document_id=document_id,
                    doc_type=doc_type,
                    title=title,
                    source=source,
                    content_hash=content_hash,
                    tags=tags,
                    metadata=metadata,
                    chunks=chunks,
                )
            ]
        )
        result = results[0]
        return result.document_id, result.action, result.chunks_created

    async def upsert_documents_with_chunks(
        self, documents: list[DocumentChunkUpsert]
    ) -> list[DocumentChunkUpsertResult]:
        if not documents:
            return []

        now = utc_now()

        def run(conn: apsw.Connection) -> list[DocumentChunkUpsertResult]:
            now_timestamp = _to_db_timestamp(now)
            results: list[DocumentChunkUpsertResult] = []

            for document in documents:
                document_id, exists = _resolve_document_upsert_identity(conn, document)
                document_id_text = str(document_id)
                if exists:
                    _delete_document_children(conn, [document_id_text])

                _upsert_document_row(conn, document_id_text, document, now_timestamp)
                _insert_document_tags(conn, document_id_text, document.tags)
                _insert_document_chunks(conn, document_id_text, document, now_timestamp)
                if exists or document.chunks:
                    self.vector_backend.mark_dirty_sync(conn)
                results.append(
                    DocumentChunkUpsertResult(
                        document_id=document_id,
                        source=document.source,
                        action="replaced" if exists else "created",
                        chunks_created=len(document.chunks),
                    )
                )

            return results

        return await self.write(run)

    async def delete_document(self, document_id: UUID) -> int:
        def run(conn: apsw.Connection) -> int:
            chunk_ids = [
                int(row["id"])
                for row in _rows_from_cursor(
                    conn.cursor(),
                    "SELECT id FROM chunks WHERE document_id = ?",
                    (str(document_id),),
                )
            ]
            conn.execute("DELETE FROM documents WHERE id = ?", (str(document_id),))
            if chunk_ids:
                self.vector_backend.mark_dirty_sync(conn)
            return len(chunk_ids)

        deleted = await self.write(run)
        return int(deleted)


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS documents(
      id TEXT PRIMARY KEY,
      type TEXT CHECK(type IN('document','memory','url','file')),
      title TEXT,
      source TEXT,
      content_hash TEXT,
      created_at TEXT,
      updated_at TEXT,
      metadata TEXT DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_doc_source ON documents(source)",
    "CREATE INDEX IF NOT EXISTS idx_doc_title_nocase ON documents(title COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_doc_type_source ON documents(type, source)",
    "CREATE INDEX IF NOT EXISTS idx_doc_type_content_hash ON documents(type, content_hash)",
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_unique_type_source
    ON documents(type, source)
    WHERE type != 'memory'
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_unique_memory_content_hash
    ON documents(type, content_hash)
    WHERE type = 'memory'
    """,
    """
    CREATE TABLE IF NOT EXISTS tags(
      document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
      tag TEXT,
      PRIMARY KEY(document_id, tag)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks(
      id INTEGER PRIMARY KEY,
      document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
      content TEXT,
      chunk_index INTEGER,
      token_count INTEGER,
      created_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id)",
    """
    CREATE TABLE IF NOT EXISTS chunk_vectors(
      chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
      embedding BLOB NOT NULL
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
      content,
      content='chunks',
      content_rowid='id',
      tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, content) VALUES(new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, content)
      VALUES('delete', old.id, old.content);
      DELETE FROM chunk_vectors WHERE chunk_id = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, content)
      VALUES('delete', old.id, old.content);
      INSERT INTO chunks_fts(rowid, content) VALUES(new.id, new.content);
    END
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
      title,
      source,
      tags,
      metadata,
      content,
      tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_search_ad AFTER DELETE ON chunks BEGIN
      DELETE FROM search_fts WHERE rowid = old.id;
    END
    """,
    """
    CREATE TABLE IF NOT EXISTS stats_counters(
      name TEXT PRIMARY KEY,
      value INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stats_document_types(
      type TEXT PRIMARY KEY,
      value INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_stats_ai AFTER INSERT ON documents BEGIN
      INSERT INTO stats_counters(name, value) VALUES('documents', 1)
      ON CONFLICT(name) DO UPDATE SET value = value + 1;
      INSERT INTO stats_document_types(type, value) VALUES(new.type, 1)
      ON CONFLICT(type) DO UPDATE SET value = value + 1;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_stats_ad AFTER DELETE ON documents BEGIN
      UPDATE stats_counters SET value = max(value - 1, 0) WHERE name = 'documents';
      UPDATE stats_document_types SET value = max(value - 1, 0) WHERE type = old.type;
      DELETE FROM stats_document_types WHERE value = 0;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_stats_au AFTER UPDATE OF type ON documents
    WHEN old.type != new.type
    BEGIN
      UPDATE stats_document_types SET value = max(value - 1, 0) WHERE type = old.type;
      DELETE FROM stats_document_types WHERE value = 0;
      INSERT INTO stats_document_types(type, value) VALUES(new.type, 1)
      ON CONFLICT(type) DO UPDATE SET value = value + 1;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_stats_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO stats_counters(name, value) VALUES('chunks', 1)
      ON CONFLICT(name) DO UPDATE SET value = value + 1;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_stats_ad AFTER DELETE ON chunks BEGIN
      UPDATE stats_counters SET value = max(value - 1, 0) WHERE name = 'chunks';
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tags_stats_ai AFTER INSERT ON tags BEGIN
      INSERT INTO stats_counters(name, value) VALUES('tags', 1)
      ON CONFLICT(name) DO UPDATE SET value = value + 1;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tags_stats_ad AFTER DELETE ON tags BEGIN
      UPDATE stats_counters SET value = max(value - 1, 0) WHERE name = 'tags';
    END
    """,
    "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)",
]


def _rows_from_cursor(
    cursor: apsw.Cursor, query: str, parameters: dict[str, Any] | tuple[Any, ...] | list[Any] = ()
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


def _fetch_one(
    conn: apsw.Connection, query: str, parameters: tuple[Any, ...] = ()
) -> dict[str, Any] | None:
    rows = _rows_from_cursor(conn.cursor(), query, parameters)
    return rows[0] if rows else None


def _set_meta_sync(conn: apsw.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _batched[T](items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _resolve_document_upsert_identity(
    conn: apsw.Connection, document: DocumentChunkUpsert
) -> tuple[UUID, bool]:
    natural_row = _fetch_one(
        conn,
        f"""
        SELECT id
        FROM documents
        WHERE type = ? AND {_ingest_identity_column(document.doc_type)} = ?
        LIMIT 1
        """,
        (str(document.doc_type), _ingest_identity_value(document)),
    )
    if natural_row is not None:
        return _uuid(natural_row["id"]), True

    if document.document_id is None:
        return uuid4(), False

    id_row = _fetch_one(
        conn,
        "SELECT id FROM documents WHERE id = ? LIMIT 1",
        (str(document.document_id),),
    )
    return document.document_id, id_row is not None


def _ingest_identity_column(doc_type: DocumentType) -> str:
    return "content_hash" if doc_type == DocumentType.MEMORY else "source"


def _ingest_identity_value(document: DocumentChunkUpsert) -> str:
    return document.content_hash if document.doc_type == DocumentType.MEMORY else document.source


def _delete_document_children(conn: apsw.Connection, document_ids: list[str]) -> None:
    if not document_ids:
        return
    for document_id_batch in _batched(document_ids, 900):
        placeholders = ",".join("?" for _ in document_id_batch)
        conn.execute(f"DELETE FROM chunks WHERE document_id IN ({placeholders})", document_id_batch)
        conn.execute(f"DELETE FROM tags WHERE document_id IN ({placeholders})", document_id_batch)


def _ensure_stats_counters_sync(conn: apsw.Connection) -> None:
    for name in ("documents", "chunks", "tags"):
        conn.execute(
            "INSERT OR IGNORE INTO stats_counters(name, value) VALUES(?, 0)",
            (name,),
        )
    initialized = _fetch_one(
        conn,
        "SELECT value FROM meta WHERE key = ?",
        ("stats_counters_initialized",),
    )
    if initialized is not None and str(initialized["value"]) == "true":
        return
    has_rows = any(
        _fetch_one(conn, f"SELECT 1 FROM {table} LIMIT 1") is not None
        for table in ("documents", "chunks", "tags")
    )
    if has_rows:
        _set_meta_sync(conn, "stats_counters_initialized", "false")
        _set_meta_sync(conn, "stats_counters_status", "pending")
    else:
        _set_meta_sync(conn, "stats_counters_initialized", "true")
        _set_meta_sync(conn, "stats_counters_status", "complete")
        _set_meta_sync(conn, "stats_counters_error", "")


def _refresh_stats_counters_sync(conn: apsw.Connection) -> None:
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


def _count_table_sync(conn: apsw.Connection, table: str) -> int:
    row = _fetch_one(conn, f"SELECT count(*) AS total FROM {table}")
    return int(row["total"] if row else 0)


def _ensure_search_fts_backfill_status_sync(conn: apsw.Connection) -> None:
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


def _backfill_search_fts_batch(conn: apsw.Connection, batch_size: int = 1000) -> int:
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
                _metadata_search_text(json_loads(row["metadata"])),
                str(row["content"]),
            )
            for row in rows
        ],
    )
    return len(rows)


def _upsert_document_row(
    conn: apsw.Connection,
    document_id: str,
    document: DocumentChunkUpsert,
    now_timestamp: str,
) -> None:
    conn.execute(
        """
        INSERT INTO documents(
            id, type, title, source, content_hash, created_at, updated_at, metadata
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            type = excluded.type,
            title = excluded.title,
            source = excluded.source,
            content_hash = excluded.content_hash,
            updated_at = excluded.updated_at,
            metadata = excluded.metadata
        """,
        (
            document_id,
            str(document.doc_type),
            document.title,
            document.source,
            document.content_hash,
            now_timestamp,
            now_timestamp,
            json_dumps(document.metadata),
        ),
    )


def _insert_document_tags(conn: apsw.Connection, document_id: str, tags: list[str]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO tags(document_id, tag) VALUES(?, ?)",
        [(document_id, tag) for tag in tags],
    )


def _metadata_search_text(value: Any) -> str:
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


def _insert_document_chunks(
    conn: apsw.Connection,
    document_id: str,
    document: DocumentChunkUpsert,
    now_timestamp: str,
) -> None:
    tags_text = " ".join(document.tags)
    metadata_text = _metadata_search_text(document.metadata)
    for chunk_index, (content, token_count, embedding) in enumerate(document.chunks):
        conn.execute(
            """
            INSERT INTO chunks(document_id, content, chunk_index, token_count, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (document_id, content, chunk_index, token_count, now_timestamp),
        )
        chunk_id = conn.last_insert_rowid()
        conn.execute(
            "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES(?, ?)",
            (chunk_id, embedding_blob(embedding)),
        )
        conn.execute(
            """
            INSERT INTO search_fts(rowid, title, source, tags, metadata, content)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, document.title, document.source, tags_text, metadata_text, content),
        )


def _document_select_sql() -> str:
    return """
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
           ) AS tags
    FROM documents d
    """


def embedding_blob(embedding: list[float] | bytes | memoryview) -> bytes:
    if isinstance(embedding, bytes):
        return embedding
    if isinstance(embedding, memoryview):
        return embedding.tobytes()
    return pack(f"<{len(embedding)}f", *embedding)


def _embedding_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return [float(item) for item in json.loads(value)]
    data = bytes(value)
    if not data:
        return []
    return [float(item) for item in unpack(f"<{len(data) // 4}f", data)]


def document_from_row(row: dict[str, Any]) -> Document:
    data = _extract_prefixed(row, "d")
    raw_tags: object = data.get("tags")
    if isinstance(raw_tags, str):
        loaded_tags: object = json.loads(raw_tags)
        tags = (
            [str(tag) for tag in cast(list[object], loaded_tags)]
            if isinstance(loaded_tags, list)
            else []
        )
    elif isinstance(raw_tags, list):
        tags = [str(tag) for tag in cast(list[object], raw_tags)]
    else:
        tags = []
    return Document(
        id=_uuid(data["id"]),
        type=DocumentType(str(data["type"])),
        title=str(data["title"]),
        source=str(data["source"]),
        content_hash=str(data["content_hash"]),
        created_at=_from_db_timestamp(data["created_at"]),
        updated_at=_from_db_timestamp(data["updated_at"]),
        tags=tags,
        metadata=json_loads(data.get("metadata")),
    )


def chunk_from_row(row: dict[str, Any]) -> Chunk:
    data = _extract_prefixed(row, "c")
    return Chunk(
        id=int(data["id"]),
        document_id=_uuid(data["document_id"]),
        content=str(data["content"]),
        chunk_index=int(data["chunk_index"]),
        embedding=_embedding_list(data.get("embedding")),
        token_count=int(data["token_count"]),
        created_at=_from_db_timestamp(data["created_at"]),
    )


def _extract_prefixed(row: dict[str, Any], alias: str) -> dict[str, Any]:
    prefix = f"{alias}."
    extracted = {
        key.removeprefix(prefix): value for key, value in row.items() if key.startswith(prefix)
    }
    if extracted:
        return extracted
    return row
