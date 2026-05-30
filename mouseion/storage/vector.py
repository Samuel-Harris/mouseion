from __future__ import annotations

import importlib.resources
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, cast

import apsw

from mouseion.config import Settings

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


class QueryResponse(Protocol):
    rows: list[dict[str, Any]]

    def first(self) -> dict[str, Any] | None: ...


class QueryExecutor(Protocol):
    def __call__(
        self,
        query: str,
        parameters: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    ) -> Awaitable[QueryResponse]: ...


class WriteExecutor(Protocol):
    def __call__(self, fn: Callable[[apsw.Connection], T]) -> Awaitable[T]: ...


@dataclass(frozen=True, slots=True)
class VectorRuntimeConfig:
    requested_mode: VectorSearchMode
    active_mode: VectorSearchMode
    qbits: Literal[2, 3, 4]
    dirty: bool
    quantized_available: bool
    warning: str | None = None


@dataclass(slots=True)
class SQLiteVectorBackend:
    settings: Settings

    def load_extension(self, conn: apsw.Connection) -> None:
        ext_path = importlib.resources.files("sqlite_vector.binaries") / "vector"
        conn.enableloadextension(True)
        try:
            conn.load_extension(str(ext_path))
        finally:
            conn.enableloadextension(False)

    def initialize_if_present_sync(self, conn: apsw.Connection) -> None:
        if _table_exists_sync(conn, VECTOR_TABLE):
            self.initialize_sync(conn)

    def initialize_sync(self, conn: apsw.Connection) -> None:
        conn.execute(
            "SELECT vector_init(?, ?, ?)",
            (VECTOR_TABLE, VECTOR_COLUMN, VECTOR_INIT_OPTIONS),
        )

    def ensure_bootstrap_meta_sync(self, conn: apsw.Connection) -> None:
        _ensure_meta_sync(conn, VECTOR_META_MODE, "exact")
        _ensure_meta_sync(conn, VECTOR_META_QBITS, str(self.settings.vector_quantization_qbits))
        _ensure_meta_sync(conn, VECTOR_META_DIRTY, "true")
        _ensure_meta_sync(conn, VECTOR_META_AVAILABLE, "false")
        _ensure_meta_sync(conn, VECTOR_META_ROWS, "0")

    def mark_dirty_sync(self, conn: apsw.Connection) -> None:
        _set_meta_sync(conn, VECTOR_META_DIRTY, "true")

    async def runtime_config(self, execute: QueryExecutor) -> VectorRuntimeConfig:
        result = await execute(
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

    async def status(self, execute: QueryExecutor) -> dict[str, Any]:
        config = await self.runtime_config(execute)
        result = await execute(
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
            "estimated_preload_memory": await self.quantize_memory(execute, config.qbits),
            "warning": config.warning,
        }

    async def quantize_memory(self, execute: QueryExecutor, qbits: int) -> int:
        rows = await execute(f"SELECT count(*) AS total FROM {VECTOR_TABLE}")
        row = rows.first()
        row_count = int(row["total"] if row else 0)
        try:
            result = await execute(
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

    async def set_mode_exact(
        self, write: WriteExecutor, execute: QueryExecutor
    ) -> dict[str, Any]:
        def run(conn: apsw.Connection) -> None:
            _set_meta_sync(conn, VECTOR_META_MODE, "exact")

        await write(run)
        return await self.status(execute)

    async def set_mode_quantized(
        self, write: WriteExecutor, execute: QueryExecutor, qbits: int
    ) -> dict[str, Any]:
        await self.quantize(
            write,
            execute,
            qbits=qbits,
            preload=self.settings.vector_quantize_preload,
        )

        def run(conn: apsw.Connection) -> None:
            _set_meta_sync(conn, VECTOR_META_MODE, "quantized")

        await write(run)
        return await self.status(execute)

    async def quantize(
        self,
        write: WriteExecutor,
        execute: QueryExecutor,
        *,
        qbits: int,
        preload: bool = False,
    ) -> dict[str, Any]:
        validated_qbits = _validate_qbits(qbits)

        def run(conn: apsw.Connection) -> int:
            self.initialize_sync(conn)
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

        rows = await write(run)
        status = await self.status(execute)
        status["quantized_rows"] = rows
        return status

    async def cleanup(self, write: WriteExecutor, execute: QueryExecutor) -> dict[str, Any]:
        def run(conn: apsw.Connection) -> None:
            self.initialize_sync(conn)
            conn.execute(f"SELECT vector_quantize_cleanup('{VECTOR_TABLE}', '{VECTOR_COLUMN}')")
            _set_meta_sync(conn, VECTOR_META_AVAILABLE, "false")
            _set_meta_sync(conn, VECTOR_META_DIRTY, "true")
            _set_meta_sync(conn, VECTOR_META_ROWS, "0")

        await write(run)
        return await self.status(execute)


def vector_scan_function(mode: VectorSearchMode) -> str:
    return "vector_quantize_scan" if mode == "quantized" else "vector_full_scan"


def _ensure_meta_sync(conn: apsw.Connection, key: str, value: str) -> None:
    row = _fetch_one(conn, "SELECT value FROM meta WHERE key = ?", (key,))
    if row is None:
        conn.execute("INSERT INTO meta(key, value) VALUES(?, ?)", (key, value))


def _set_meta_sync(conn: apsw.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


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


def _table_exists_sync(conn: apsw.Connection, table: str) -> bool:
    row = _fetch_one(
        conn,
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    )
    return row is not None


def _fetch_one(
    conn: apsw.Connection, query: str, parameters: tuple[Any, ...] = ()
) -> dict[str, Any] | None:
    cursor = conn.cursor()
    results = iter(cursor.execute(query, parameters))
    try:
        first = next(results)
    except StopIteration:
        return None
    description = cursor.getdescription() or []
    names = [str(column[0]) for column in description]
    return dict(zip(names, first, strict=False))
