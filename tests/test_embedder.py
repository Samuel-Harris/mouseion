from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mouseion.config import Settings
from mouseion.errors import EmbeddingError
from mouseion.ingest import embedder as embedder_module
from mouseion.ingest.embedder import EMBEDDING_DIMENSIONS, Embedder


def test_embed_many_uses_ollama_batch_endpoint(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def embed(self, *, model: str, input: list[str]) -> dict[str, list[list[float]]]:
            calls.append({"host": self.host, "model": model, "input": input})
            return {"embeddings": [[1.0] * EMBEDDING_DIMENSIONS for _ in input]}

    monkeypatch.setattr(embedder_module.ollama, "Client", FakeClient)
    settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        OLLAMA_HOST="http://example.test",
        EMBEDDING_MODEL="batch-model",
    )

    vectors = Embedder(settings)._embed_many_sync(["first", "second"])

    assert calls == [
        {
            "host": "http://example.test",
            "model": "batch-model",
            "input": ["first", "second"],
        }
    ]
    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIMENSIONS


def test_ensure_ready_checks_ollama_model(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[dict[str, str]] = []

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def show(self, model: str) -> dict[str, str]:
            calls.append({"host": self.host, "model": model})
            return {"model": model}

    monkeypatch.setattr(embedder_module.ollama, "Client", FakeClient)
    settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        OLLAMA_HOST="http://example.test",
        EMBEDDING_MODEL="ready-model",
    )

    Embedder(settings)._ensure_ready_sync()

    assert calls == [{"host": "http://example.test", "model": "ready-model"}]


def test_ensure_ready_wraps_ollama_failures(tmp_path: Path, monkeypatch: Any) -> None:
    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def show(self, model: str) -> None:
            raise RuntimeError("connection refused")

    monkeypatch.setattr(embedder_module.ollama, "Client", FakeClient)
    settings = Settings(
        MOUSEION_DATA_DIR=tmp_path / "data",
        MOUSEION_REPOS_DIR=tmp_path / "repos",
        EMBEDDING_MODEL="missing-model",
    )

    with pytest.raises(EmbeddingError, match="Embedding backend is not ready"):
        Embedder(settings)._ensure_ready_sync()
