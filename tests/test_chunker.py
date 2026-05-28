from __future__ import annotations

from pathlib import Path

from mouseion.config import Settings
from mouseion.ingest.chunker import Chunker


def test_chunker_respects_hard_cap(tmp_path: Path) -> None:
    settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        CHUNK_TARGET_TOKENS=40,
        CHUNK_MAX_TOKENS=80,
        CHUNK_MIN_TOKENS=10,
    )
    chunker = Chunker(settings)
    text = "# Title\n\n" + "Sentence with several tokens. " * 120
    chunks = chunker.chunk(text)
    assert chunks
    assert all(chunk.token_count <= settings.chunk_max_tokens for chunk in chunks)
