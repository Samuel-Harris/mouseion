from __future__ import annotations

from pathlib import Path

import pytest

from mouseion.config import Settings


def test_settings_defaults(tmp_path: Path) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    assert settings.mouseion_host == "127.0.0.1"
    assert settings.sqlite_path == tmp_path / "data" / "mouseion.db"
    assert settings.is_loopback_host()


def test_settings_rejects_bad_chunk_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Settings(
            MOUSEION_DATA_DIR=tmp_path / "data",
            MOUSEION_REPOS_DIR=tmp_path / "repos",
            CHUNK_TARGET_TOKENS=2048,
            CHUNK_MAX_TOKENS=1024,
        )


def test_non_loopback_host_detection(tmp_path: Path) -> None:
    settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        MOUSEION_HOST="0.0.0.0",
    )
    assert not settings.is_loopback_host()
