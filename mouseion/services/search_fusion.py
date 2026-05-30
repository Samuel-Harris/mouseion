from __future__ import annotations

from dataclasses import dataclass

LEXICAL_FAST_PATH_SCORE = 18.0


@dataclass(slots=True)
class RankedHit:
    chunk_id: int
    score: float
    rank: int
    source: str = ""


@dataclass(slots=True)
class FusedHit:
    chunk_id: int
    score: float
    match: str


@dataclass(frozen=True, slots=True)
class FusionResult:
    hits: list[FusedHit]
    requires_vector: bool


@dataclass(slots=True)
class HybridFusionPolicy:
    top_k: int

    def exact_lexical(self, hits: list[RankedHit]) -> FusionResult:
        if not hits:
            return FusionResult([], requires_vector=True)
        return FusionResult(
            [FusedHit(hit.chunk_id, hit.score, "lexical") for hit in hits],
            requires_vector=False,
        )

    def lexical_fast_path(self, hits: list[RankedHit]) -> FusionResult:
        if not hits or hits[0].score < LEXICAL_FAST_PATH_SCORE:
            return FusionResult([], requires_vector=True)
        return FusionResult(
            [
                FusedHit(hit.chunk_id, hit.score, "lexical")
                for hit in hits[: self.top_k]
            ],
            requires_vector=False,
        )

    def hybrid(self, lexical_hits: list[RankedHit], vector_hits: list[RankedHit]) -> FusionResult:
        return FusionResult(
            hybrid_fuse(lexical_hits, vector_hits, top_k=self.top_k),
            requires_vector=False,
        )


def rrf_fuse(hit_lists: list[list[RankedHit]], k: int) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for hits in hit_lists:
        for hit in hits:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_fuse(
    lexical_hits: list[RankedHit],
    vector_hits: list[RankedHit],
    *,
    top_k: int,
) -> list[FusedHit]:
    scores: dict[int, float] = {}
    sources: dict[int, set[str]] = {}

    for hit in lexical_hits:
        lexical_score = 2.0 + min(hit.score, 30.0) / 8.0 + 1.0 / hit.rank
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + lexical_score
        sources.setdefault(hit.chunk_id, set()).add("lexical")

    for hit in vector_hits:
        if not lexical_hits and hit.score < 0.78:
            continue
        semantic_score = max(0.0, hit.score - 0.70) * 2.0 + 0.25 / hit.rank
        if semantic_score <= 0:
            continue
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + semantic_score
        sources.setdefault(hit.chunk_id, set()).add("semantic")

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            item[1],
            "lexical" in sources.get(item[0], set()),
            item[0] * -1,
        ),
        reverse=True,
    )
    return [
        FusedHit(chunk_id=chunk_id, score=score, match="+".join(sorted(sources[chunk_id])))
        for chunk_id, score in ranked[:top_k]
    ]
