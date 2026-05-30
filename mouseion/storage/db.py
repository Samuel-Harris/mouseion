from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from struct import pack, unpack
from typing import Any, TypeVar, cast
from uuid import UUID, uuid4

import apsw
from anyio.to_thread import run_sync

from mouseion.config import Settings
from mouseion.domain.models import Chunk, Document, DocumentType, utc_now
from mouseion.storage import background
from mouseion.storage.schema import SCHEMA_STATEMENTS, SCHEMA_VERSION
from mouseion.storage.search_index import (
    ensure_search_fts_backfill_status_sync,
    insert_search_index_row_sync,
)
from mouseion.storage.stats import (
    background_status_sync,
    ensure_stats_counters_sync,
    stats_snapshot_sync,
)
from mouseion.storage.vector import SQLiteVectorBackend, VectorRuntimeConfig
from mouseion.support.utils import json_dumps, json_loads

READ_POOL_SIZE = 4
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


@dataclass(frozen=True, slots=True)
class DocumentChunkStats:
    total_chunks: int
    total_chars: int
    total_tokens: int


class SQLiteStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_backend = SQLiteVectorBackend(settings)
        self.database: apsw.Connection | None = None
        self._readonly = False
        self._write_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._read_pool: list[apsw.Connection] = []
        self._read_pool_lock = threading.Lock()

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
        self._close_read_pool_sync()
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

    async def read(self, fn: Callable[[apsw.Connection], T]) -> T:
        return await run_sync(self._read_sync, fn)

    def _read_sync(self, fn: Callable[[apsw.Connection], T]) -> T:
        with self._pooled_read_connection_sync() as conn:
            return fn(conn)

    def _execute_sync(
        self, query: str, parameters: dict[str, Any] | tuple[Any, ...] | list[Any]
    ) -> QueryResult:
        with self._pooled_read_connection_sync() as conn:
            return QueryResult(_rows_from_cursor(conn.cursor(), query, parameters))

    @contextlib.contextmanager
    def _pooled_read_connection_sync(self) -> Generator[apsw.Connection]:
        conn: apsw.Connection | None = None
        with self._read_pool_lock:
            if self._read_pool:
                conn = self._read_pool.pop()
        if conn is None:
            conn = self._connect_sync(readonly=True)
        try:
            yield conn
        finally:
            should_close = False
            with self._read_pool_lock:
                if len(self._read_pool) < READ_POOL_SIZE:
                    self._read_pool.append(conn)
                else:
                    should_close = True
            if should_close:
                conn.close()

    def _close_read_pool_sync(self) -> None:
        with self._read_pool_lock:
            connections = self._read_pool
            self._read_pool = []
        for conn in connections:
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
            ensure_stats_counters_sync(conn)
            ensure_search_fts_backfill_status_sync(conn)
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
        for runner in (background.backfill_stats_counters, background.backfill_search_fts):
            task = asyncio.create_task(runner(self))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def background_status(self) -> dict[str, Any]:
        return await self.read(background_status_sync)

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
        return await self.read(stats_snapshot_sync)

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

    async def get_text_chunks_for_document(
        self,
        document_id: UUID,
        *,
        start_chunk_index: int = 0,
        limit: int | None = None,
    ) -> list[Chunk]:
        limit_sql = "" if limit is None else "LIMIT ?"
        params: tuple[Any, ...] = (
            (str(document_id), start_chunk_index)
            if limit is None
            else (str(document_id), start_chunk_index, limit)
        )
        result = await self.execute(
            f"""
            SELECT c.id,
                   c.document_id,
                   c.content,
                   c.chunk_index,
                   c.token_count,
                   c.created_at
            FROM chunks c
            WHERE c.document_id = ?
              AND c.chunk_index >= ?
            ORDER BY c.chunk_index
            {limit_sql}
            """,
            params,
        )
        return [chunk_from_row(row) for row in result.rows]

    async def document_chunk_stats(self, document_id: UUID) -> DocumentChunkStats:
        result = await self.execute(
            """
            SELECT count(*) AS total_chunks,
                   COALESCE(sum(length(content)), 0) AS total_chars,
                   COALESCE(sum(token_count), 0) AS total_tokens
            FROM chunks
            WHERE document_id = ?
            """,
            (str(document_id),),
        )
        row = result.first() or {}
        return DocumentChunkStats(
            total_chunks=int(row.get("total_chunks", 0)),
            total_chars=int(row.get("total_chars", 0)),
            total_tokens=int(row.get("total_tokens", 0)),
        )

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


def _insert_document_chunks(
    conn: apsw.Connection,
    document_id: str,
    document: DocumentChunkUpsert,
    now_timestamp: str,
) -> None:
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
        insert_search_index_row_sync(
            conn,
            chunk_id=chunk_id,
            title=document.title,
            source=document.source,
            tags=document.tags,
            metadata=document.metadata,
            content=content,
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
