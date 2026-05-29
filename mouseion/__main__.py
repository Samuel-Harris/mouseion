from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import apsw
import uvicorn

from mouseion.config import Settings
from mouseion.support.logging_config import configure_logging, get_logger

OLLAMA_START_TIMEOUT_SECONDS = 20.0
OLLAMA_POLL_SECONDS = 0.25
NUKE_DB_CONFIRMATION = "nuke mouseion db"
STATUS_TIMEOUT_SECONDS = 2.0


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
    elif args.command == "nuke-db":
        _nuke_db(settings, assume_yes=args.yes)


def _collect_status(settings: Settings) -> dict[str, Any]:
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


def _print_status(status: dict[str, Any]) -> None:
    stats = status["stats"]
    edges = stats["edges"]
    print("Mouseion status")
    print(f"Running: {'yes' if status['running'] else 'no'}")
    print(f"Daemon: {status['daemon_url']}")
    print(f"Database: {status['database']}")
    print(f"Stats source: {status['stats_source']}")
    print(f"Documents: {stats['documents']}")
    for doc_type, total in stats["documents_by_type"].items():
        print(f"  {doc_type}: {total}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Edges: {edges['total']}")
    print(f"  related: {edges['related']}")
    print(f"  similar: {edges['similar']}")
    print(f"Tags: {stats['tags']}")


def _mouseion_base_url(settings: Settings) -> str:
    host = settings.mouseion_host
    if "://" not in host:
        host = f"http://{host}"
    return f"{host.rstrip('/')}:{settings.mouseion_port}"


def _mouseion_json(base_url: str, path: str, *, timeout: float = 10) -> dict[str, Any]:
    request = Request(urljoin(base_url.rstrip("/") + "/", path))
    with urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))


def _local_database_stats(settings: Settings) -> dict[str, Any]:
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


def _stats_from_connection(conn: apsw.Connection) -> dict[str, Any]:
    document_types: dict[str, int] = {}
    if _table_exists(conn, "documents"):
        for doc_type, total in conn.execute(
            "SELECT type, count(*) FROM documents GROUP BY type ORDER BY type"
        ):
            document_types[str(doc_type)] = int(total)

    related_edges = _count_table(conn, "related_to")
    similar_edges = _count_table(conn, "similar_to")
    return {
        "documents": _count_table(conn, "documents"),
        "chunks": _count_table(conn, "chunks"),
        "tags": _count_table(conn, "tags"),
        "edges": {
            "total": related_edges + similar_edges,
            "related": related_edges,
            "similar": similar_edges,
        },
        "documents_by_type": document_types,
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


def _empty_stats() -> dict[str, Any]:
    return {
        "documents": 0,
        "chunks": 0,
        "tags": 0,
        "edges": {"total": 0, "related": 0, "similar": 0},
        "documents_by_type": {},
    }


def _normalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    normalized = _empty_stats()
    normalized["documents"] = int(stats.get("documents", 0))
    normalized["chunks"] = int(stats.get("chunks", 0))
    normalized["tags"] = int(stats.get("tags", 0))

    edges = stats.get("edges")
    if isinstance(edges, dict):
        related = int(edges.get("related", 0))
        similar = int(edges.get("similar", 0))
        total = int(edges.get("total", related + similar))
        normalized["edges"] = {"total": total, "related": related, "similar": similar}

    documents_by_type = stats.get("documents_by_type")
    if isinstance(documents_by_type, dict):
        normalized["documents_by_type"] = {
            str(doc_type): int(total) for doc_type, total in documents_by_type.items()
        }
    return normalized


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
def _managed_ollama(settings: Settings, *, enabled: bool) -> Iterator[None]:
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


def _wait_for_ollama(host: str, process: subprocess.Popen) -> None:
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
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
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
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(urljoin(host.rstrip("/") + "/", path), data=data, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))


if __name__ == "__main__":
    main()
