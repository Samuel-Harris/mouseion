"""arXiv category taxonomy: the human-authored seed and the ListSets-derived structure."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from utilities.arxiv.oai import OaiSet


@dataclass(frozen=True, slots=True)
class CategoryEntry:
    code: str
    name: str
    description: str
    group: str
    group_slug: str

    def metadata(self) -> dict[str, str]:
        return {
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "group_slug": self.group_slug,
        }


@dataclass(slots=True)
class CategoryCatalog:
    categories: dict[str, CategoryEntry]
    groups: dict[str, str]

    def resolve(self, code: str) -> CategoryEntry | None:
        return self.categories.get(code.lower())

    def group_matches(self, requested_groups: set[str], codes: list[str]) -> bool:
        return any(
            entry.group_slug in requested_groups
            for code in codes
            if (entry := self.resolve(code)) is not None
        )


def load_category_catalog(path: Path) -> CategoryCatalog:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    categories: dict[str, CategoryEntry] = {}
    groups: dict[str, str] = {}

    if not isinstance(raw, list):
        raise ValueError(f"Expected {path} to contain a list of arXiv category groups")

    for group_item in cast(list[object], raw):
        if not isinstance(group_item, dict):
            continue
        group_mapping = cast(dict[object, object], group_item)
        for group_name, entries in group_mapping.items():
            group_slug = slugify(str(group_name))
            groups[group_slug] = str(group_name)
            if not isinstance(entries, list):
                continue
            for entry_item in cast(list[object], entries):
                if not isinstance(entry_item, dict):
                    continue
                entry_mapping = cast(dict[object, object], entry_item)
                for code, details in entry_mapping.items():
                    if not isinstance(details, dict):
                        continue
                    details_mapping = cast(dict[object, object], details)
                    category = CategoryEntry(
                        code=str(code),
                        name=str(details_mapping.get("name") or ""),
                        description=str(details_mapping.get("description") or ""),
                        group=str(group_name),
                        group_slug=group_slug,
                    )
                    categories[category.code.lower()] = category

    return CategoryCatalog(categories=categories, groups=groups)


def build_category_catalog_from_sets(seed: CategoryCatalog, sets: list[OaiSet]) -> CategoryCatalog:
    group_names_by_spec = top_level_groups(sets)
    categories = dict(seed.categories)
    for set_spec in sets:
        code = category_code_from_set_spec(set_spec.spec)
        if code is None:
            continue
        key = code.lower()
        if key in categories:
            continue
        top_level = _set_spec_segments(set_spec.spec)[0]
        group_name = group_names_by_spec.get(top_level, top_level)
        categories[key] = CategoryEntry(
            code=code,
            name=set_spec.name,
            description="",
            group=group_name,
            group_slug=slugify(group_name),
        )

    groups = dict(seed.groups)
    for group_name in group_names_by_spec.values():
        groups.setdefault(slugify(group_name), group_name)
    return CategoryCatalog(categories=categories, groups=groups)


def top_level_groups(sets: list[OaiSet]) -> dict[str, str]:
    return {
        set_spec.spec: set_spec.name
        for set_spec in sets
        if len(_set_spec_segments(set_spec.spec)) == 1
    }


def category_code_from_set_spec(set_spec: str) -> str | None:
    segments = _set_spec_segments(set_spec)
    match segments:
        case [_, archive]:
            return archive
        case [_, archive, category]:
            return f"{archive}.{category}"
        case _:
            return None


def category_name_mismatches(
    seed: CategoryCatalog, sets: list[OaiSet]
) -> dict[str, tuple[str, str]]:
    derived_names: dict[str, str] = {}
    for set_spec in sets:
        code = category_code_from_set_spec(set_spec.spec)
        if code is not None:
            derived_names[code.lower()] = set_spec.name

    mismatches: dict[str, tuple[str, str]] = {}
    for key, entry in seed.categories.items():
        derived = derived_names.get(key)
        if derived is not None and derived != entry.name:
            mismatches[entry.code] = (entry.name, derived)
    return mismatches


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _set_spec_segments(set_spec: str) -> list[str]:
    return [segment for segment in set_spec.split(":") if segment]
