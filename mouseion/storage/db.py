from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from struct import pack, unpack
from typing import Any, TypeVar, cast
from uuid import UUID, uuid4

import apsw
import sqlite_vec
from anyio.to_thread import run_sync

from mouseion.config import Settings
from mouseion.domain.models import Chunk, Document, DocumentType, utc_now
from mouseion.support.utils import json_dumps, json_loads

SCHEMA_VERSION = "1"
EMBEDDING_DIMS = 768
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
        conn.enableloadextension(True)
        sqlite_vec.load(conn)
        conn.enableloadextension(False)
        if not readonly:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA recursive_triggers=ON")
        conn.execute("PRAGMA busy_timeout=5000")
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
            self._ensure_meta_sync(conn, "schema_version", SCHEMA_VERSION)
            self._ensure_meta_sync(conn, "last_recompute_at", "")

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

    async def delete_document(self, document_id: UUID) -> tuple[int, int]:
        def run(conn: apsw.Connection) -> tuple[int, int]:
            chunk_ids = [
                int(row["id"])
                for row in _rows_from_cursor(
                    conn.cursor(),
                    "SELECT id FROM chunks WHERE document_id = ?",
                    (str(document_id),),
                )
            ]
            related_edges = _count(
                conn,
                "SELECT count(*) FROM related_to WHERE from_doc = ? OR to_doc = ?",
                (str(document_id), str(document_id)),
            )
            similar_edges = 0
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                similar_edges = _count(
                    conn,
                    f"""
                    SELECT count(*)
                    FROM similar_to
                    WHERE from_chunk IN ({placeholders}) OR to_chunk IN ({placeholders})
                    """,
                    (*chunk_ids, *chunk_ids),
                )
            conn.execute("DELETE FROM documents WHERE id = ?", (str(document_id),))
            return len(chunk_ids), len(chunk_ids) + related_edges + similar_edges

        deleted = await self.write(run)
        return (int(deleted[0]), int(deleted[1]))


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
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
      chunk_id INTEGER PRIMARY KEY,
      embedding FLOAT[{EMBEDDING_DIMS}] distance_metric=cosine
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
    CREATE TABLE IF NOT EXISTS similar_to(
      from_chunk INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
      to_chunk INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
      score REAL,
      CHECK(from_chunk < to_chunk),
      PRIMARY KEY(from_chunk, to_chunk)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sim_from ON similar_to(from_chunk)",
    "CREATE INDEX IF NOT EXISTS idx_sim_to ON similar_to(to_chunk)",
    """
    CREATE TABLE IF NOT EXISTS related_to(
      from_doc TEXT REFERENCES documents(id) ON DELETE CASCADE,
      to_doc TEXT REFERENCES documents(id) ON DELETE CASCADE,
      label TEXT DEFAULT '',
      note TEXT,
      created_at TEXT,
      PRIMARY KEY(from_doc, to_doc, label)
    )
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


def _count(conn: apsw.Connection, query: str, parameters: tuple[Any, ...] = ()) -> int:
    row = _fetch_one(conn, query, parameters)
    if not row:
        return 0
    return int(next(iter(row.values())))


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
