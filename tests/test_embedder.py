from __future__ import annotations

from pathlib import Path
from typing import Any

from mouseion.config import Settings
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
