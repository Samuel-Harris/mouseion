from __future__ import annotations

from uuid import UUID

import pytest

from mouseion.domain.models import SearchFilter, normalize_tags
from mouseion.services.search import RankedHit, rrf_fuse
from mouseion.support.utils import canonical_text_hash


def test_normalize_tags_deduplicates_and_lowercases() -> None:
    assert normalize_tags([" AI ", "ai", "Papers", ""]) == ["ai", "papers"]


def test_search_filter_normalizes_tags() -> None:
    assert SearchFilter(tags=[" Research ", "research"]).tags == ["research"]


def test_search_filter_accepts_document_id() -> None:
    document_id = UUID("00000000-0000-0000-0000-000000000123")

    assert SearchFilter(document_id=str(document_id)).document_id == document_id


def test_hash_canonicalizes_whitespace() -> None:
    assert canonical_text_hash("a  b\nc") == canonical_text_hash("a b c")


def test_rrf_fuse_combines_rank_sources() -> None:
    fused = rrf_fuse(
        [
            [RankedHit(1, 0.9, 1), RankedHit(2, 0.7, 2)],
            [RankedHit(2, 3.0, 1), RankedHit(3, 1.0, 2)],
        ],
        k=60,
    )
    assert fused[0][0] == 2


def test_search_filter_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        SearchFilter(type="bad")  # type: ignore[arg-type]
