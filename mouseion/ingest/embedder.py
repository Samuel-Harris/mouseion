from __future__ import annotations

from dataclasses import dataclass

import anyio
import ollama

from mouseion.config import Settings
from mouseion.errors import EmbeddingError

EMBEDDING_DIMENSIONS = 768


@dataclass(slots=True)
class Embedder:
    settings: Settings

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await anyio.to_thread.run_sync(self._embed_many_sync, texts)

    def _embed_many_sync(self, texts: list[str]) -> list[list[float]]:
        client = ollama.Client(host=self.settings.ollama_host)
        try:
            response = client.embed(model=self.settings.embedding_model, input=texts)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"Embedding backend failed for model {self.settings.embedding_model!r}: {exc}"
            ) from exc

        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError(
                "Embedding backend returned invalid batch response "
                f"for {self.settings.embedding_model!r}"
            )

        vectors: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSIONS:
                raise EmbeddingError(
                    "Embedding backend returned invalid vector dimensions "
                    f"for {self.settings.embedding_model!r}"
                )
            vectors.append([float(value) for value in embedding])
        return vectors
