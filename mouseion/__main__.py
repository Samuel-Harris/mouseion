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

import uvicorn

from mouseion.config import Settings
from mouseion.support.logging_config import configure_logging, get_logger

OLLAMA_START_TIMEOUT_SECONDS = 20.0
OLLAMA_POLL_SECONDS = 0.25
NUKE_DB_CONFIRMATION = "nuke mouseion db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="museion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="Start the mouseion daemon")
    serve.add_argument(
        "--no-ollama",
        action="store_false",
        dest="manage_ollama",
        help="Do not start Ollama or pull the embedding model before serving",
    )
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
    elif args.command == "nuke-db":
        _nuke_db(settings, assume_yes=args.yes)


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
