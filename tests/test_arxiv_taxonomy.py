from __future__ import annotations

from pathlib import Path

from utilities.arxiv.oai import OaiSet
from utilities.arxiv.taxonomy import (
    build_category_catalog_from_sets,
    category_code_from_set_spec,
    category_name_mismatches,
    load_category_catalog,
)

SEED_PATH = Path("utilities/arxiv/categories.json")


def test_category_code_is_derived_from_set_spec() -> None:
    assert category_code_from_set_spec("cs") is None
    assert category_code_from_set_spec("physics:astro-ph") == "astro-ph"
    assert category_code_from_set_spec("cs:cs:AI") == "cs.AI"
    assert category_code_from_set_spec("math:math:AT") == "math.AT"
    assert category_code_from_set_spec("physics:cond-mat:stat-mech") == "cond-mat.stat-mech"


def test_build_catalog_merges_seed_descriptions_and_blanks_new_codes() -> None:
    seed = load_category_catalog(SEED_PATH)
    sets = [
        OaiSet(spec="cs", name="Computer Science"),
        OaiSet(spec="cs:cs:AI", name="Artificial Intelligence"),
        OaiSet(spec="cs:cs:XX", name="Brand New Area"),
    ]

    catalog = build_category_catalog_from_sets(seed, sets)

    existing = catalog.resolve("cs.AI")
    assert existing is not None
    assert existing.name == "Artificial Intelligence"
    assert existing.description == seed.resolve("cs.AI").description  # type: ignore[union-attr]
    assert existing.group_slug == "computer-science"

    new_code = catalog.resolve("cs.XX")
    assert new_code is not None
    assert new_code.name == "Brand New Area"
    assert new_code.description == ""
    assert new_code.group_slug == "computer-science"


def test_derived_codes_are_a_superset_of_the_seed() -> None:
    seed = load_category_catalog(SEED_PATH)
    sets = [
        OaiSet(spec="cs", name="Computer Science"),
        OaiSet(spec="cs:cs:AI", name="Artificial Intelligence"),
        OaiSet(spec="math", name="Mathematics"),
        OaiSet(spec="math:math:AT", name="Algebraic Topology"),
        OaiSet(spec="physics", name="Physics"),
        OaiSet(spec="physics:cond-mat:stat-mech", name="Statistical Mechanics"),
    ]

    catalog = build_category_catalog_from_sets(seed, sets)

    assert set(catalog.categories) >= set(seed.categories)
    assert len(seed.categories) >= 150


def test_category_name_mismatches_report_arxiv_differences() -> None:
    seed = load_category_catalog(SEED_PATH)
    sets = [
        OaiSet(spec="cs", name="Computer Science"),
        OaiSet(spec="cs:cs:AI", name="AI (renamed)"),
    ]

    mismatches = category_name_mismatches(seed, sets)

    assert mismatches["cs.AI"] == ("Artificial Intelligence", "AI (renamed)")
