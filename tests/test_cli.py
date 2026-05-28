from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mouseion.__main__ import (
    NUKE_DB_CONFIRMATION,
    _ensure_ollama_model,
    _model_names_match,
    _nuke_db,
    build_parser,
)
from mouseion.config import Settings


def test_serve_parser_manages_ollama_by_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve"])

    assert parser.prog == "museion"
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
    assert "usage: museion" in result.stdout


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
