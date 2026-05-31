from __future__ import annotations

from pathlib import Path

import pytest

from mouseion.config import Settings


def test_settings_defaults(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
    )
    assert settings.mouseion_host == "127.0.0.1"
    assert settings.mouseion_mcp_description is None
    assert settings.sqlite_path == tmp_path / "data" / "mouseion.db"
    assert settings.vector_search_mode == "exact"
    assert settings.vector_quantization_qbits == 4
    assert settings.vector_quantize_preload is False
    assert settings.vector_quantize_max_memory == "30MB"
    assert settings.is_loopback_host()


def test_settings_normalizes_mcp_description(tmp_path: Path) -> None:
    settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        MOUSEION_MCP_DESCRIPTION="  Local research corpus  ",
    )
    empty_settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        MOUSEION_MCP_DESCRIPTION=" ",
    )

    assert settings.mouseion_mcp_description == "Local research corpus"
    assert empty_settings.mouseion_mcp_description is None


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


def test_vector_settings_validate_mode_and_qbits(tmp_path: Path) -> None:
    settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        MOUSEION_VECTOR_SEARCH_MODE="QUANTIZED",
        MOUSEION_VECTOR_QUANTIZATION_QBITS=2,
        MOUSEION_VECTOR_QUANTIZE_PRELOAD=True,
        MOUSEION_VECTOR_QUANTIZE_MAX_MEMORY=" 50mb ",
    )

    assert settings.vector_search_mode == "quantized"
    assert settings.vector_quantization_qbits == 2
    assert settings.vector_quantize_preload is True
    assert settings.vector_quantize_max_memory == "50MB"


def test_vector_settings_reject_invalid_qbits(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Settings(
            MOUSEION_DATA_DIR=tmp_path / "data",
            MOUSEION_REPOS_DIR=tmp_path / "repos",
            MOUSEION_VECTOR_QUANTIZATION_QBITS=8,
        )


def test_vector_settings_coerce_qbits_from_env_string(tmp_path: Path) -> None:
    settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        MOUSEION_VECTOR_QUANTIZATION_QBITS="4",
    )

    assert settings.vector_quantization_qbits == 4
