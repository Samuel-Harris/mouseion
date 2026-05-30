from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import apsw
import uvicorn

from mouseion.config import Settings
from mouseion.storage.db import SQLiteStore
from mouseion.support.logging_config import configure_logging, get_logger

OLLAMA_START_TIMEOUT_SECONDS = 20.0
OLLAMA_POLL_SECONDS = 0.25
NUKE_DB_CONFIRMATION = "nuke mouseion db"
STATUS_TIMEOUT_SECONDS = 2.0
JsonDict = dict[str, Any]
ObjectMapping = Mapping[str, object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mouseion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="Start the mouseion daemon")
    serve.add_argument(
        "--no-ollama",
        action="store_false",
        dest="manage_ollama",
        help="Do not start Ollama or pull the embedding model before serving",
    )
    subparsers.add_parser("status", help="Show daemon status and database summary stats")
    vector = subparsers.add_parser("vector", help="Manage SQLite vector search mode")
    vector_subparsers = vector.add_subparsers(dest="vector_command", required=True)
    vector_subparsers.add_parser("status", help="Show vector backend status")
    vector_mode = vector_subparsers.add_parser("mode", help="Switch vector search mode")
    vector_mode_subparsers = vector_mode.add_subparsers(dest="mode", required=True)
    vector_mode_subparsers.add_parser("exact", help="Use exact full-scan vector search")
    quantized_mode = vector_mode_subparsers.add_parser(
        "quantized", help="Quantize vectors and use TurboQuant search"
    )
    quantized_mode.add_argument("--qbits", type=int, choices=[2, 3, 4], default=None)
    vector_quantize = vector_subparsers.add_parser(
        "quantize", help="Rebuild TurboQuant data without changing search mode"
    )
    vector_quantize.add_argument("--qbits", type=int, choices=[2, 3, 4], required=True)
    vector_quantize.add_argument("--preload", action="store_true", help="Preload quantized data")
    vector_subparsers.add_parser("cleanup", help="Remove TurboQuant data")
    nuke_db = subparsers.add_parser("nuke-db", help="Delete the mouseion SQLite database")
    nuke_db.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Delete database files without prompting for confirmation",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    if args.command == "serve":
        with _managed_ollama(settings, enabled=args.manage_ollama):
            settings.warn_if_non_loopback(logger)
            uvicorn.run(
                "mouseion.api.server:create_app",
                factory=True,
                host=settings.mouseion_host,
                port=settings.mouseion_port,
                log_config=None,
            )
    elif args.command == "status":
        _print_status(_collect_status(settings))
    elif args.command == "vector":
        _print_vector_result(asyncio.run(_vector_command(settings, args)))
    elif args.command == "nuke-db":
        _nuke_db(settings, assume_yes=args.yes)


async def _vector_command(settings: Settings, args: argparse.Namespace) -> JsonDict:
    command = str(args.vector_command)
    if command == "status":
        return await _vector_daemon_or_local(settings, "api/vector/status")
    if command == "mode":
        mode = str(args.mode)
        payload: JsonDict = {"mode": mode}
        if mode == "quantized" and args.qbits is not None:
            payload["qbits"] = int(args.qbits)
        return await _vector_daemon_or_local(
            settings, "api/vector/mode", method="POST", payload=payload
        )
    if command == "quantize":
        return await _vector_daemon_or_local(
            settings,
            "api/vector/quantize",
            method="POST",
            payload={"qbits": int(args.qbits), "preload": bool(args.preload)},
        )
    if command == "cleanup":
        return await _vector_daemon_or_local(settings, "api/vector/cleanup", method="POST")
    raise ValueError(f"Unsupported vector command: {command}")


async def _vector_daemon_or_local(
    settings: Settings,
    path: str,
    *,
    method: str = "GET",
    payload: JsonDict | None = None,
) -> JsonDict:
    daemon_url = _mouseion_base_url(settings)
    try:
        result = _mouseion_json(
            daemon_url,
            path,
            payload=payload,
            method=method,
            timeout=STATUS_TIMEOUT_SECONDS,
        )
        return {"source": "daemon", **result}
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        store = SQLiteStore(settings)
        await store.open()
        try:
            if path == "api/vector/status":
                result = await store.vector_status()
            elif path == "api/vector/mode":
                if payload is None:
                    raise ValueError("Vector mode command requires a payload")
                mode = str(payload.get("mode", ""))
                if mode == "exact":
                    result = await store.set_vector_mode_exact()
                elif mode == "quantized":
                    qbits_value = payload.get("qbits")
                    qbits = (
                        int(qbits_value)
                        if qbits_value is not None
                        else settings.vector_quantization_qbits
                    )
                    result = await store.set_vector_mode_quantized(qbits)
                else:
                    raise ValueError(f"Unsupported vector mode: {mode}")
            elif path == "api/vector/quantize":
                if payload is None:
                    raise ValueError("Vector quantize command requires a payload")
                result = await store.quantize_vectors(
                    qbits=int(payload["qbits"]),
                    preload=bool(payload.get("preload", False)),
                )
            elif path == "api/vector/cleanup":
                result = await store.cleanup_quantized_vectors()
            else:
                raise ValueError(f"Unsupported vector API path: {path}")
        finally:
            await store.close()
        return {"source": "local database", **result}


def _print_vector_result(result: JsonDict) -> None:
    print("Mouseion vector")
    print(f"Source: {result['source']}")
    print(f"Configured mode: {result.get('configured_mode', 'exact')}")
    print(f"Effective mode: {result.get('effective_mode', 'exact')}")
    print(f"Configured qbits: {int(result.get('configured_qbits', 4))}")
    print(f"Dirty: {'yes' if result.get('dirty', True) else 'no'}")
    print(f"Quantized available: {'yes' if result.get('quantized_available', False) else 'no'}")
    print(f"Quantized rows: {int(result.get('quantized_rows', 0))}")
    print(f"Estimated preload memory bytes: {int(result.get('estimated_preload_memory', 0))}")
    warning = result.get("warning")
    if warning:
        print(f"Warning: {warning}")


def _collect_status(settings: Settings) -> JsonDict:
    daemon_url = _mouseion_base_url(settings)
    try:
        stats = _mouseion_json(daemon_url, "api/stats", timeout=STATUS_TIMEOUT_SECONDS)
        return {
            "running": True,
            "daemon_url": daemon_url,
            "database": str(settings.sqlite_path),
            "stats_source": "daemon",
            "stats": _normalize_stats(stats),
        }
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        local = _local_database_stats(settings)
        return {
            "running": False,
            "daemon_url": daemon_url,
            "database": str(settings.sqlite_path),
            "stats_source": local["source"],
            "stats": local["stats"],
        }


def _print_status(status: JsonDict) -> None:
    stats = cast(ObjectMapping, status["stats"])
    print("Mouseion status")
    print(f"Running: {'yes' if status['running'] else 'no'}")
    print(f"Daemon: {status['daemon_url']}")
    print(f"Database: {status['database']}")
    print(f"Stats source: {status['stats_source']}")
    print(f"Documents: {stats['documents']}")
    documents_by_type = cast(ObjectMapping, stats["documents_by_type"])
    for doc_type, total in documents_by_type.items():
        print(f"  {doc_type}: {total}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Tags: {stats['tags']}")
    background_tasks = stats.get("background_tasks")
    if isinstance(background_tasks, Mapping):
        print("Background tasks:")
        for name, details in cast(ObjectMapping, background_tasks).items():
            if isinstance(details, Mapping):
                status_value = cast(ObjectMapping, details).get("status", "unknown")
                print(f"  {name}: {status_value}")


def _mouseion_base_url(settings: Settings) -> str:
    host = settings.mouseion_host
    if "://" not in host:
        host = f"http://{host}"
    return f"{host.rstrip('/')}:{settings.mouseion_port}"


def _mouseion_json(
    base_url: str,
    path: str,
    *,
    timeout: float = 10,
    method: str = "GET",
    payload: JsonDict | None = None,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        urljoin(base_url.rstrip("/") + "/", path),
        data=data,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))


def _local_database_stats(settings: Settings) -> JsonDict:
    if not settings.sqlite_path.exists():
        return {"source": "local database (not found)", "stats": _empty_stats()}
    try:
        conn = apsw.Connection(str(settings.sqlite_path), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            return {"source": "local database", "stats": _stats_from_connection(conn)}
        finally:
            conn.close()
    except apsw.Error:
        return {"source": "local database (unreadable)", "stats": _empty_stats()}


def _stats_from_connection(conn: apsw.Connection) -> JsonDict:
    document_types: dict[str, int] = {}
    if _table_exists(conn, "stats_document_types"):
        for doc_type, total in conn.execute(
            "SELECT type, value FROM stats_document_types WHERE value > 0 ORDER BY type"
        ):
            document_types[str(doc_type)] = int(total)
    elif _table_exists(conn, "documents"):
        for doc_type, total in conn.execute(
            "SELECT type, count(*) FROM documents GROUP BY type ORDER BY type"
        ):
            document_types[str(doc_type)] = int(total)

    counters = _stats_counters(conn)
    if counters is not None:
        return {
            "documents": counters.get("documents", 0),
            "chunks": counters.get("chunks", 0),
            "tags": counters.get("tags", 0),
            "documents_by_type": document_types,
        }

    return {
        "documents": _count_table(conn, "documents"),
        "chunks": _count_table(conn, "chunks"),
        "tags": _count_table(conn, "tags"),
        "documents_by_type": document_types,
    }


def _stats_counters(conn: apsw.Connection) -> dict[str, int] | None:
    if not _table_exists(conn, "stats_counters"):
        return None
    return {
        str(name): int(value)
        for name, value in conn.execute("SELECT name, value FROM stats_counters")
    }


def _count_table(conn: apsw.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    row = next(conn.execute(f"SELECT count(*) FROM {table}"))
    return int(row[0])


def _table_exists(conn: apsw.Connection, table: str) -> bool:
    row = next(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ),
        None,
    )
    return row is not None


def _empty_stats() -> JsonDict:
    return {
        "documents": 0,
        "chunks": 0,
        "tags": 0,
        "documents_by_type": {},
        "background_tasks": {},
    }


def _normalize_stats(stats: ObjectMapping) -> JsonDict:
    normalized = _empty_stats()
    normalized["documents"] = _int_value(stats.get("documents"), 0)
    normalized["chunks"] = _int_value(stats.get("chunks"), 0)
    normalized["tags"] = _int_value(stats.get("tags"), 0)

    documents_by_type = stats.get("documents_by_type")
    if isinstance(documents_by_type, Mapping):
        document_type_values = cast(ObjectMapping, documents_by_type)
        normalized["documents_by_type"] = {
            str(doc_type): _int_value(total, 0) for doc_type, total in document_type_values.items()
        }
    background_tasks = stats.get("background_tasks")
    if isinstance(background_tasks, Mapping):
        normalized["background_tasks"] = dict(cast(ObjectMapping, background_tasks))
    return normalized


def _int_value(value: object, default: int) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    return default


def _nuke_db(settings: Settings, *, assume_yes: bool = False) -> list[Path]:
    if not assume_yes and not _confirm_nuke_db(settings):
        print("Aborted.")
        return []

    removed: list[Path] = []
    for path in _database_paths(settings):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(path)

    if removed:
        print("Deleted database files:")
        for path in removed:
            print(f"- {path}")
    else:
        print(f"No database files found for {settings.sqlite_path}")
    return removed


def _confirm_nuke_db(settings: Settings) -> bool:
    print("WARNING: this permanently deletes Mouseion's SQLite database.")
    print(f"Database: {settings.sqlite_path}")
    print("Stop the Mouseion daemon before continuing.")
    answer = input(f"Type {NUKE_DB_CONFIRMATION!r} to continue: ")
    return answer == NUKE_DB_CONFIRMATION


def _database_paths(settings: Settings) -> list[Path]:
    database = settings.sqlite_path
    return [
        database,
        database.with_name(f"{database.name}-wal"),
        database.with_name(f"{database.name}-shm"),
    ]


@contextmanager
def _managed_ollama(settings: Settings, *, enabled: bool) -> Generator[None]:
    if not enabled:
        yield
        return

    process: subprocess.Popen[bytes] | None = None
    if not _ollama_reachable(settings.ollama_host):
        process = subprocess.Popen(["ollama", "serve"])
    try:
        if process is not None:
            _wait_for_ollama(settings.ollama_host, process)
        _ensure_ollama_model(settings.ollama_host, settings.embedding_model)
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _wait_for_ollama(host: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + OLLAMA_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("`ollama serve` exited before the API became reachable")
        if _ollama_reachable(host):
            return
        time.sleep(OLLAMA_POLL_SECONDS)
    raise RuntimeError(f"Ollama did not become reachable at {host} within 20 seconds")


def _ollama_reachable(host: str) -> bool:
    try:
        with urlopen(urljoin(host.rstrip("/") + "/", "api/version"), timeout=1):
            return True
    except (OSError, URLError):
        return False


def _ensure_ollama_model(host: str, model: str) -> None:
    if _ollama_model_available(host, model):
        return
    print(f"Pulling Ollama model {model!r}...")
    _ollama_json(host, "api/pull", {"name": model, "stream": False}, timeout=3600)


def _ollama_model_available(host: str, model: str) -> bool:
    payload = _ollama_json(host, "api/tags")
    models = payload.get("models", [])
    if not isinstance(models, list):
        return False
    for item in cast(list[object], models):
        if not isinstance(item, Mapping):
            continue
        model_info = cast(ObjectMapping, item)
        name = model_info.get("name") or model_info.get("model")
        if isinstance(name, str) and _model_names_match(name, model):
            return True
    return False


def _model_names_match(available: str, requested: str) -> bool:
    if available == requested:
        return True
    if ":" in requested:
        return False
    return available == f"{requested}:latest" or available.split(":", maxsplit=1)[0] == requested


def _ollama_json(
    host: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 10
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(urljoin(host.rstrip("/") + "/", path), data=data, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))


if __name__ == "__main__":
    main()
