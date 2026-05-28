from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from mouseion.domain.models import SearchFilter, normalize_tags
from mouseion.services.search import RankedHit, rrf_fuse
from mouseion.support.utils import canonical_text_hash, deterministic_edge_id


def test_normalize_tags_deduplicates_and_lowercases() -> None:
    assert normalize_tags([" AI ", "ai", "Papers", ""]) == ["ai", "papers"]


def test_search_filter_normalizes_tags() -> None:
    assert SearchFilter(tags=[" Research ", "research"]).tags == ["research"]


def test_hash_canonicalizes_whitespace() -> None:
    assert canonical_text_hash("a  b\nc") == canonical_text_hash("a b c")


def test_rrf_fuse_combines_rank_sources() -> None:
    fused = rrf_fuse(
        [
            [RankedHit("a", 0.9, 1), RankedHit("b", 0.7, 2)],
            [RankedHit("b", 3.0, 1), RankedHit("c", 1.0, 2)],
        ],
        k=60,
    )
    assert fused[0][0] == "b"


def test_deterministic_edge_id_is_stable() -> None:
    left = uuid4()
    right = uuid4()
    edge_id = deterministic_edge_id(left, right, "related")
    assert edge_id == deterministic_edge_id(left, right, "related")
    assert isinstance(edge_id, UUID)


def test_search_filter_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        SearchFilter(type="bad")
