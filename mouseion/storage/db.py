from __future__ import annotations

import asyncio
import importlib.resources
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from struct import pack, unpack
from typing import Any, Literal, TypeVar, cast
from uuid import UUID, uuid4

import apsw
from anyio.to_thread import run_sync

from mouseion.config import Settings
from mouseion.domain.models import Chunk, Document, DocumentType, utc_now
from mouseion.support.utils import json_dumps, json_loads

SCHEMA_VERSION = "3"
EMBEDDING_DIMS = 768
VECTOR_TABLE = "chunk_vectors"
VECTOR_COLUMN = "embedding"
VECTOR_INIT_OPTIONS = f"type=FLOAT32,dimension={EMBEDDING_DIMS},distance=COSINE"
VECTOR_META_MODE = "vector_search_mode"
VECTOR_META_QBITS = "vector_quantization_qbits"
VECTOR_META_DIRTY = "vector_quantization_dirty"
VECTOR_META_AVAILABLE = "vector_quantization_available"
VECTOR_META_ROWS = "vector_quantized_rows"
VectorSearchMode = Literal["exact", "quantized"]
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
class VectorRuntimeConfig:
    requested_mode: VectorSearchMode
    active_mode: VectorSearchMode
    qbits: Literal[2, 3, 4]
    dirty: bool
    quantized_available: bool
    warning: str | None = None


class SQLiteStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database: apsw.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def open(self) -> None:
        self.settings.ensure_directories()
        await run_sync(self._open_sync)
        await self.bootstrap()

    async def open_readonly(self) -> None:
        await run_sync(self._open_sync, True)

    def _open_sync(self, readonly: bool = False) -> None:
        flags = (
            apsw.SQLITE_OPEN_READONLY
            if readonly
            else apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE
        )
        conn = apsw.Connection(str(self.settings.sqlite_path), flags=flags)
        _load_sqlite_vector(conn)
        if not readonly:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA recursive_triggers=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        if readonly and _table_exists_sync(conn, VECTOR_TABLE):
            _initialize_vector_sync(conn)
        self.database = conn

    async def close(self) -> None:
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
        return QueryResult(_rows_from_cursor(self.connection().cursor(), query, parameters))

    async def write(self, fn: Callable[[apsw.Connection], T]) -> T:
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
            self._ensure_meta_sync(conn, "schema_version", SCHEMA_VERSION)
            self._ensure_meta_sync(conn, VECTOR_META_MODE, "exact")
            self._ensure_meta_sync(
                conn, VECTOR_META_QBITS, str(self.settings.vector_quantization_qbits)
            )
            self._ensure_meta_sync(conn, VECTOR_META_DIRTY, "true")
            self._ensure_meta_sync(conn, VECTOR_META_AVAILABLE, "false")
            self._ensure_meta_sync(conn, VECTOR_META_ROWS, "0")
            conn.execute(
                """
                INSERT INTO meta(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("schema_version", SCHEMA_VERSION),
            )
            _initialize_vector_sync(conn)

        await self.write(run)

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
        result = await self.execute(
            """
            SELECT key, value
            FROM meta
            WHERE key IN (?, ?, ?, ?)
            """,
            (VECTOR_META_MODE, VECTOR_META_QBITS, VECTOR_META_DIRTY, VECTOR_META_AVAILABLE),
        )
        meta = {str(row["key"]): str(row["value"]) for row in result.rows}
        requested_mode = _effective_vector_mode(self.settings, meta)
        qbits = _effective_vector_qbits(self.settings, meta)
        dirty = _bool_meta(meta.get(VECTOR_META_DIRTY), default=True)
        available = _bool_meta(meta.get(VECTOR_META_AVAILABLE), default=False)
        warning: str | None = None
        active_mode: VectorSearchMode = requested_mode
        if requested_mode == "quantized" and (dirty or not available):
            reason = "dirty" if dirty else "unavailable"
            warning = f"Quantized vectors are {reason}; falling back to exact vector scan."
            active_mode = "exact"
        return VectorRuntimeConfig(
            requested_mode=requested_mode,
            active_mode=active_mode,
            qbits=qbits,
            dirty=dirty,
            quantized_available=available,
            warning=warning,
        )

    async def vector_status(self) -> dict[str, Any]:
        config = await self.vector_runtime_config()
        result = await self.execute(
            """
            SELECT key, value
            FROM meta
            WHERE key IN (?, ?)
            """,
            (VECTOR_META_ROWS, VECTOR_META_QBITS),
        )
        meta = {str(row["key"]): str(row["value"]) for row in result.rows}
        quantized_rows = _int_meta(meta.get(VECTOR_META_ROWS), default=0)
        return {
            "effective_mode": config.active_mode,
            "configured_mode": config.requested_mode,
            "configured_qbits": config.qbits,
            "dirty": config.dirty,
            "quantized_available": config.quantized_available,
            "quantized_rows": quantized_rows,
            "estimated_preload_memory": await self.vector_quantize_memory(config.qbits),
            "warning": config.warning,
        }

    async def vector_quantize_memory(self, qbits: int) -> int:
        rows = await self.execute(f"SELECT count(*) AS total FROM {VECTOR_TABLE}")
        row = rows.first()
        row_count = int(row["total"] if row else 0)
        try:
            result = await self.execute(
                f"SELECT vector_quantize_memory('{VECTOR_TABLE}', '{VECTOR_COLUMN}') AS bytes"
            )
            row = result.first()
            if row is not None:
                estimated = int(row["bytes"])
                if estimated > 0 or row_count == 0:
                    return estimated
        except apsw.Error:
            pass
        return int(row_count * ((EMBEDDING_DIMS * qbits / 8) + 8))

    async def set_vector_mode_exact(self) -> dict[str, Any]:
        def run(conn: apsw.Connection) -> None:
            _set_meta_sync(conn, VECTOR_META_MODE, "exact")

        await self.write(run)
        return await self.vector_status()

    async def set_vector_mode_quantized(self, qbits: int) -> dict[str, Any]:
        await self.quantize_vectors(qbits=qbits, preload=self.settings.vector_quantize_preload)

        def run(conn: apsw.Connection) -> None:
            _set_meta_sync(conn, VECTOR_META_MODE, "quantized")

        await self.write(run)
        return await self.vector_status()

    async def quantize_vectors(self, *, qbits: int, preload: bool = False) -> dict[str, Any]:
        validated_qbits = _validate_qbits(qbits)

        def run(conn: apsw.Connection) -> int:
            _initialize_vector_sync(conn)
            options = (
                f"qtype=TURBO,qbits={validated_qbits},"
                f"max_memory={self.settings.vector_quantize_max_memory}"
            )
            row = _fetch_one(
                conn,
                f"""
                SELECT vector_quantize('{VECTOR_TABLE}', '{VECTOR_COLUMN}', ?) AS total
                """,
                (options,),
            )
            if preload:
                conn.execute(f"SELECT vector_quantize_preload('{VECTOR_TABLE}', '{VECTOR_COLUMN}')")
            total = int(row["total"] if row else 0)
            _set_meta_sync(conn, VECTOR_META_QBITS, str(validated_qbits))
            _set_meta_sync(conn, VECTOR_META_DIRTY, "false")
            _set_meta_sync(conn, VECTOR_META_AVAILABLE, "true")
            _set_meta_sync(conn, VECTOR_META_ROWS, str(total))
            return total

        rows = await self.write(run)
        status = await self.vector_status()
        status["quantized_rows"] = rows
        return status

    async def cleanup_quantized_vectors(self) -> dict[str, Any]:
        def run(conn: apsw.Connection) -> None:
            _initialize_vector_sync(conn)
            conn.execute(f"SELECT vector_quantize_cleanup('{VECTOR_TABLE}', '{VECTOR_COLUMN}')")
            _set_meta_sync(conn, VECTOR_META_AVAILABLE, "false")
            _set_meta_sync(conn, VECTOR_META_DIRTY, "true")
            _set_meta_sync(conn, VECTOR_META_ROWS, "0")

        await self.write(run)
        return await self.vector_status()

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
                _insert_document_chunks(conn, document_id_text, document.chunks, now_timestamp)
                if exists or document.chunks:
                    _mark_quantization_dirty_sync(conn)
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
                _mark_quantization_dirty_sync(conn)
            return len(chunk_ids)

        deleted = await self.write(run)
        return int(deleted)


SCHEMA_STATEMENTS = [
    "DROP TABLE IF EXISTS similar_to",
    "DROP TABLE IF EXISTS related_to",
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
    "DROP INDEX IF EXISTS idx_sim_from",
    "DROP INDEX IF EXISTS idx_sim_to",
    "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)",
]


def _load_sqlite_vector(conn: apsw.Connection) -> None:
    ext_path = importlib.resources.files("sqlite_vector.binaries") / "vector"
    conn.enableloadextension(True)
    try:
        conn.load_extension(str(ext_path))
    finally:
        conn.enableloadextension(False)


def _initialize_vector_sync(conn: apsw.Connection) -> None:
    conn.execute(
        "SELECT vector_init(?, ?, ?)",
        (VECTOR_TABLE, VECTOR_COLUMN, VECTOR_INIT_OPTIONS),
    )


def _table_exists_sync(conn: apsw.Connection, table: str) -> bool:
    row = _fetch_one(
        conn,
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    )
    return row is not None


def _set_meta_sync(conn: apsw.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _mark_quantization_dirty_sync(conn: apsw.Connection) -> None:
    _set_meta_sync(conn, VECTOR_META_DIRTY, "true")


def _effective_vector_mode(settings: Settings, meta: dict[str, str]) -> VectorSearchMode:
    if "MOUSEION_VECTOR_SEARCH_MODE" in os.environ:
        return settings.vector_search_mode
    value = meta.get(VECTOR_META_MODE, settings.vector_search_mode)
    return "quantized" if value == "quantized" else "exact"


def _effective_vector_qbits(settings: Settings, meta: dict[str, str]) -> Literal[2, 3, 4]:
    if "MOUSEION_VECTOR_QUANTIZATION_QBITS" in os.environ:
        return settings.vector_quantization_qbits
    return _validate_qbits(
        _int_meta(meta.get(VECTOR_META_QBITS), settings.vector_quantization_qbits)
    )


def _validate_qbits(value: int) -> Literal[2, 3, 4]:
    if value not in {2, 3, 4}:
        raise ValueError("Vector quantization qbits must be one of 2, 3, or 4")
    return cast(Literal[2, 3, 4], value)


def _bool_meta(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int_meta(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


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
    chunks: list[tuple[str, int, list[float]]],
    now_timestamp: str,
) -> None:
    for chunk_index, (content, token_count, embedding) in enumerate(chunks):
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
