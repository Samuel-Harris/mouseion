from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.error import URLError

import apsw
import pytest

from mouseion.__main__ import (
    NUKE_DB_CONFIRMATION,
    _collect_status,
    _ensure_ollama_model,
    _model_names_match,
    _nuke_db,
    _recompute_edges,
    build_parser,
)
from mouseion.config import Settings


def test_serve_parser_manages_ollama_by_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve"])

    assert parser.prog == "mouseion"
    assert args.command == "serve"
    assert args.manage_ollama is True


def test_serve_parser_can_disable_ollama_management() -> None:
    args = build_parser().parse_args(["serve", "--no-ollama"])

    assert args.manage_ollama is False


def test_nuke_db_parser_requires_confirmation_by_default() -> None:
    args = build_parser().parse_args(["nuke-db"])

    assert args.command == "nuke-db"
    assert args.yes is False


def test_nuke_db_parser_can_skip_confirmation() -> None:
    args = build_parser().parse_args(["nuke-db", "--yes"])

    assert args.yes is True


def test_status_parser() -> None:
    args = build_parser().parse_args(["status"])

    assert args.command == "status"


def test_recompute_edges_parser() -> None:
    args = build_parser().parse_args(["recompute-edges"])

    assert args.command == "recompute-edges"


def test_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_module_help_does_not_shadow_stdlib_logging() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mouseion", "-h"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: mouseion" in result.stdout


def test_model_names_match_latest_default() -> None:
    assert _model_names_match("nomic-embed-text:latest", "nomic-embed-text")
    assert _model_names_match("nomic-embed-text", "nomic-embed-text")
    assert not _model_names_match("other-model:latest", "nomic-embed-text")


def test_model_names_with_explicit_tags_require_exact_match() -> None:
    assert _model_names_match("nomic-embed-text:v1", "nomic-embed-text:v1")
    assert not _model_names_match("nomic-embed-text:latest", "nomic-embed-text:v1")


def test_ensure_ollama_model_pulls_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_ollama_json(
        host: str, path: str, payload: dict[str, object] | None = None, *, timeout: float = 10
    ) -> dict[str, object]:
        calls.append((host, path, payload))
        if path == "api/tags":
            return {"models": []}
        return {"status": "success"}

    monkeypatch.setattr("mouseion.__main__._ollama_json", fake_ollama_json)

    _ensure_ollama_model("http://127.0.0.1:11434", "nomic-embed-text")

    assert calls == [
        ("http://127.0.0.1:11434", "api/tags", None),
        (
            "http://127.0.0.1:11434",
            "api/pull",
            {"name": "nomic-embed-text", "stream": False},
        ),
    ]


def test_status_uses_daemon_stats_when_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")

    def fake_mouseion_json(base_url: str, path: str, *, timeout: float = 10) -> dict[str, object]:
        assert base_url == "http://127.0.0.1:7778"
        assert path == "api/stats"
        return {
            "documents": 3,
            "chunks": 12,
            "tags": 4,
            "edges": {"total": 5, "related": 2, "similar": 3},
            "documents_by_type": {"file": 1, "memory": 2},
        }

    monkeypatch.setattr("mouseion.__main__._mouseion_json", fake_mouseion_json)

    status = _collect_status(settings)

    assert status["running"] is True
    assert status["stats_source"] == "daemon"
    assert status["stats"]["documents"] == 3
    assert status["stats"]["edges"] == {"total": 5, "related": 2, "similar": 3}


def test_status_falls_back_to_local_database_when_daemon_is_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    settings.ensure_directories()
    _create_status_test_database(settings.sqlite_path)

    def fake_mouseion_json(base_url: str, path: str, *, timeout: float = 10) -> dict[str, object]:
        raise URLError("daemon offline")

    monkeypatch.setattr("mouseion.__main__._mouseion_json", fake_mouseion_json)

    status = _collect_status(settings)

    assert status["running"] is False
    assert status["stats_source"] == "local database"
    assert status["stats"]["documents"] == 2
    assert status["stats"]["documents_by_type"] == {"memory": 1, "url": 1}
    assert status["stats"]["chunks"] == 3
    assert status["stats"]["tags"] == 2
    assert status["stats"]["edges"] == {"total": 3, "related": 1, "similar": 2}


def test_status_handles_missing_local_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")

    def fake_mouseion_json(base_url: str, path: str, *, timeout: float = 10) -> dict[str, object]:
        raise URLError("daemon offline")

    monkeypatch.setattr("mouseion.__main__._mouseion_json", fake_mouseion_json)

    status = _collect_status(settings)

    assert status["running"] is False
    assert status["stats_source"] == "local database (not found)"
    assert status["stats"]["documents"] == 0
    assert status["stats"]["edges"] == {"total": 0, "related": 0, "similar": 0}


async def test_recompute_edges_uses_daemon_when_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")

    def fake_mouseion_json(
        base_url: str, path: str, *, timeout: float = 10, method: str = "GET"
    ) -> dict[str, object]:
        assert base_url == "http://127.0.0.1:7778"
        assert path == "api/recompute_edges"
        assert method == "POST"
        return {"chunks_processed": 2, "edges_created": 1, "duration_seconds": 0.5}

    monkeypatch.setattr("mouseion.__main__._mouseion_json", fake_mouseion_json)

    result = await _recompute_edges(settings)

    assert result == {
        "source": "daemon",
        "chunks_processed": 2,
        "edges_created": 1,
        "duration_seconds": 0.5,
    }


async def test_recompute_edges_handles_missing_local_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")

    def fake_mouseion_json(
        base_url: str, path: str, *, timeout: float = 10, method: str = "GET"
    ) -> dict[str, object]:
        raise URLError("daemon offline")

    monkeypatch.setattr("mouseion.__main__._mouseion_json", fake_mouseion_json)

    result = await _recompute_edges(settings)

    assert result == {
        "source": "local database (not found)",
        "chunks_processed": 0,
        "edges_created": 0,
        "duration_seconds": 0.0,
    }


def test_nuke_db_aborts_without_exact_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    settings.ensure_directories()
    settings.sqlite_path.write_text("db")
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    removed = _nuke_db(settings)

    assert removed == []
    assert settings.sqlite_path.exists()


def test_nuke_db_deletes_sqlite_files_only(tmp_path: Path) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    settings.ensure_directories()
    keep = settings.files_dir / "keep.txt"
    keep.write_text("keep")
    database_files = [
        settings.sqlite_path,
        settings.sqlite_path.with_name("mouseion.db-wal"),
        settings.sqlite_path.with_name("mouseion.db-shm"),
    ]
    for path in database_files:
        path.write_text("db")

    removed = _nuke_db(settings, assume_yes=True)

    assert removed == database_files
    assert not any(path.exists() for path in database_files)
    assert keep.exists()


def test_nuke_db_deletes_after_exact_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    settings.ensure_directories()
    settings.sqlite_path.write_text("db")
    monkeypatch.setattr("builtins.input", lambda _: NUKE_DB_CONFIRMATION)

    removed = _nuke_db(settings)

    assert removed == [settings.sqlite_path]
    assert not settings.sqlite_path.exists()


def _create_status_test_database(path: Path) -> None:
    conn = apsw.Connection(str(path))
    try:
        conn.execute("CREATE TABLE documents(id TEXT PRIMARY KEY, type TEXT)")
        conn.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, document_id TEXT)")
        conn.execute("CREATE TABLE tags(document_id TEXT, tag TEXT)")
        conn.execute("CREATE TABLE related_to(from_doc TEXT, to_doc TEXT, label TEXT)")
        conn.execute("CREATE TABLE similar_to(from_chunk INTEGER, to_chunk INTEGER)")
        conn.execute("INSERT INTO documents(id, type) VALUES('1', 'memory'), ('2', 'url')")
        conn.execute("INSERT INTO chunks(id, document_id) VALUES(1, '1'), (2, '1'), (3, '2')")
        conn.execute("INSERT INTO tags(document_id, tag) VALUES('1', 'a'), ('2', 'b')")
        conn.execute("INSERT INTO related_to(from_doc, to_doc, label) VALUES('1', '2', '')")
        conn.execute("INSERT INTO similar_to(from_chunk, to_chunk) VALUES(1, 2), (2, 3)")
    finally:
        conn.close()
