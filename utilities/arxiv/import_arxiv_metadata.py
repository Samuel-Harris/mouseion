"""Import arXiv metadata into searchable Mouseion documents.

Two modes are supported. With no ``--input`` the tool harvests arXiv changes over
OAI-PMH, deriving scope and sync position from the existing corpus. With ``--input``
it keeps the original local-JSONL behaviour for offline and backfill workflows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from mouseion.config import Settings
from mouseion.domain.models import ChunkText, DocumentType, IngestedContent, normalize_tags
from mouseion.errors import EmbeddingError, MouseionError
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.embedder import Embedder
from mouseion.services.bulk_ingest import BatchIngestItem, BulkIngestService
from mouseion.services.factory import open_services
from mouseion.storage.db import SQLiteStore
from utilities.arxiv.oai import (
    OaiError,
    OaiSet,
    fetch_oai_sets,
    iter_oai_records,
    later_datestamp,
    oai_record_to_kaggle_dict,
)
from utilities.arxiv.taxonomy import (
    CategoryCatalog,
    CategoryEntry,
    build_category_catalog_from_sets,
    category_name_mismatches,
    load_category_catalog,
    slugify,
    top_level_groups,
)

__all__ = [
    "ImportOptions",
    "ImportStats",
    "async_main",
    "category_filter_codes",
    "import_arxiv_metadata",
    "main",
    "prepare_record",
    "raw_categories_match_filter",
]

DEFAULT_CATEGORIES_JSON = Path("utilities/arxiv/categories.json")
KAGGLE_ARXIV_DOWNLOAD_URL = (
    "https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download"
)
CATEGORIES_FIELD_RE = re.compile(rb'"categories"\s*:\s*"((?:[^"\\]|\\.)*)"')

ARXIV_EARLIEST_DATESTAMP = "2005-09-16"
OAI_IMPORT_SOURCE = "oai:oaipmh.arxiv.org/oai"
ARXIV_HARVEST_CURSOR_KEY = "arxiv_harvest_cursor"
ARXIV_HARVEST_STATUS_KEY = "arxiv_harvest_status"
ARXIV_HARVEST_ERROR_KEY = "arxiv_harvest_error"
OAI_REQUEST_TIMEOUT_SECONDS = 60.0

METADATA_COMPARE_EXCLUDE = {"import_timestamp", "import_source", "authors_parsed"}


class MissingInputFileError(FileNotFoundError):
    pass


class UnknownFilterError(MouseionError):
    pass


@dataclass(frozen=True, slots=True)
class ImportOptions:
    input: Path | None = None
    categories_json: Path = DEFAULT_CATEGORIES_JSON
    groups: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    from_date: str | None = None
    limit: int | None = None
    batch_size: int = 100
    dry_run: bool = False
    progress_every: int = 1000


@dataclass(slots=True)
class PreparedPaper:
    arxiv_id: str
    source: str
    title: str
    content: str
    tags: list[str]
    metadata: dict[str, Any]
    category_codes: list[str]
    unresolved_categories: list[str]


@dataclass(slots=True)
class ImportStats:
    selected: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    deleted: int = 0
    pruned: int = 0
    unresolved_categories: Counter[str] = field(default_factory=Counter[str])
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "failed": self.failed,
            "deleted": self.deleted,
            "pruned": self.pruned,
            "unresolved_categories": dict(sorted(self.unresolved_categories.items())),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class _HarvestOutcome:
    observed_max: str | None
    deleted_ids: tuple[str, ...]


def prepare_record(
    record: dict[str, Any],
    catalog: CategoryCatalog,
    *,
    import_source: str,
    import_timestamp: str,
) -> PreparedPaper | None:
    arxiv_id = normalize_text(str(record.get("id") or ""))
    title = normalize_text(str(record.get("title") or ""))
    abstract = normalize_text(str(record.get("abstract") or ""))
    if not arxiv_id or not title or not abstract:
        return None

    raw_categories = normalize_text(str(record.get("categories") or ""))
    category_codes = raw_categories.split()
    resolved: list[CategoryEntry] = []
    unresolved: list[str] = []
    for code in category_codes:
        entry = catalog.resolve(code)
        if entry is None:
            unresolved.append(code)
        else:
            resolved.append(entry)

    tags = ["arxiv"]
    tags.extend(f"arxiv:category:{code.lower()}" for code in category_codes)
    tags.extend(f"arxiv:group:{entry.group_slug}" for entry in resolved)
    tags = normalize_tags(tags)

    group_metadata = {
        entry.group_slug: {"name": entry.group, "slug": entry.group_slug} for entry in resolved
    }
    content = f"{title}\n\n{abstract}"
    metadata = {
        "arxiv_id": arxiv_id,
        "pdf_url": f"https://export.arxiv.org/pdf/{arxiv_id}",
        "html_url": f"https://export.arxiv.org/html/{arxiv_id}",
        "categories": raw_categories,
        "category_codes": category_codes,
        "resolved_categories": [entry.metadata() for entry in resolved],
        "resolved_groups": list(group_metadata.values()),
        "unresolved_categories": unresolved,
        "authors": record.get("authors"),
        "authors_parsed": record.get("authors_parsed") or [],
        "submitter": record.get("submitter"),
        "comments": record.get("comments"),
        "journal_ref": record.get("journal-ref"),
        "doi": record.get("doi"),
        "report_number": record.get("report-no"),
        "license": record.get("license"),
        "versions": record.get("versions") or [],
        "update_date": record.get("update_date"),
        "import_source": import_source,
        "import_timestamp": import_timestamp,
    }

    return PreparedPaper(
        arxiv_id=arxiv_id,
        source=f"arxiv:{arxiv_id}",
        title=title,
        content=content,
        tags=tags,
        metadata=metadata,
        category_codes=category_codes,
        unresolved_categories=unresolved,
    )


async def import_arxiv_metadata(
    options: ImportOptions,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> ImportStats:
    start = time.monotonic()
    active_settings = settings or Settings()
    if options.input is None:
        stats = await _import_from_oai(options, settings=active_settings, embedder=embedder)
    else:
        stats = await _import_from_file(
            options.input, options, settings=active_settings, embedder=embedder
        )
    stats.elapsed_seconds = time.monotonic() - start
    return stats


async def _import_from_file(
    input_path: Path,
    options: ImportOptions,
    *,
    settings: Settings,
    embedder: Embedder | None,
) -> ImportStats:
    ensure_input_file_exists(input_path)
    catalog = load_category_catalog(options.categories_json)
    requested_groups, requested_categories = _requested_filters(options)
    requested_filter_categories = category_filter_codes(
        catalog,
        requested_groups=requested_groups,
        requested_categories=requested_categories,
    )

    async with _ingest_services(options, settings, embedder) as (_, bulk_ingest):
        intake = _build_intake(
            options,
            catalog=catalog,
            requested_groups=requested_groups,
            requested_categories=requested_categories,
            import_source=str(input_path),
            bulk_ingest=bulk_ingest,
        )
        with intake.progress_context("Scanning arXiv JSONL"), input_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if intake.limit_reached():
                    break
                line = line.strip()
                if not line:
                    intake.note_skipped()
                    continue
                if requested_filter_categories and not raw_categories_match_filter(
                    line, requested_filter_categories
                ):
                    intake.note_skipped()
                    continue
                try:
                    loaded_record: object = json.loads(line)
                except json.JSONDecodeError as exc:
                    intake.note_failure(f"line {line_number}: invalid JSON: {exc}")
                    continue
                if not isinstance(loaded_record, dict):
                    intake.note_skipped()
                    continue
                await intake.add(cast(dict[str, Any], loaded_record))
            await intake.finish()
        return intake.stats


async def _import_from_oai(
    options: ImportOptions, *, settings: Settings, embedder: Embedder | None
) -> ImportStats:
    async with (
        _ingest_services(options, settings, embedder) as (store, bulk_ingest),
        httpx.AsyncClient(follow_redirects=True, timeout=OAI_REQUEST_TIMEOUT_SECONDS) as client,
    ):
        seed_catalog = load_category_catalog(options.categories_json)
        oai_sets = await fetch_oai_sets(client)
        catalog = build_category_catalog_from_sets(seed_catalog, oai_sets)
        _warn_about_name_mismatches(seed_catalog, oai_sets)

        requested_groups, requested_categories = _requested_filters(options)
        requested_filter_categories = category_filter_codes(
            catalog,
            requested_groups=requested_groups,
            requested_categories=requested_categories,
        )
        set_specs = await _resolve_set_specs(
            oai_sets,
            catalog,
            store,
            requested_groups=requested_groups,
            requested_categories=requested_categories,
        )
        stored_cursor = await _effective_cursor(store)
        from_date = harvest_window_start(options.from_date or stored_cursor)
        intake = _build_intake(
            options,
            catalog=catalog,
            requested_groups=requested_groups,
            requested_categories=requested_categories,
            import_source=OAI_IMPORT_SOURCE,
            bulk_ingest=bulk_ingest,
        )

        if options.dry_run:
            with intake.progress_context("Planning arXiv OAI"):
                await _harvest(client, set_specs=set_specs, from_date=from_date, intake=intake)
                await intake.finish()
            return intake.stats

        assert store is not None, "a non-dry-run OAI harvest always has an open store"
        await store.set_meta(ARXIV_HARVEST_STATUS_KEY, "running")
        try:
            with intake.progress_context("Harvesting arXiv OAI"):
                outcome = await _harvest(
                    client, set_specs=set_specs, from_date=from_date, intake=intake
                )
                await intake.finish()
        except Exception as exc:
            await store.set_meta(ARXIV_HARVEST_STATUS_KEY, "error")
            await store.set_meta(ARXIV_HARVEST_ERROR_KEY, str(exc))
            raise

        await _persist_harvest(
            store,
            intake.stats,
            outcome,
            stored_cursor=stored_cursor,
            from_override=options.from_date,
            prune_scope=(
                requested_filter_categories if options.groups or options.categories else None
            ),
        )
        return intake.stats


async def _harvest(
    client: httpx.AsyncClient,
    *,
    set_specs: list[str],
    from_date: str,
    intake: _RecordIntake,
) -> _HarvestOutcome:
    observed_max: str | None = None
    deleted_ids: list[str] = []
    for set_spec in set_specs:
        async for page in iter_oai_records(client, set_spec=set_spec, from_date=from_date):
            observed_max = later_datestamp(observed_max, page.max_datestamp or "")
            deleted_ids.extend(page.deleted_ids)
            for oai_record in page.records:
                if intake.limit_reached():
                    return _HarvestOutcome(observed_max, tuple(deleted_ids))
                await intake.add(oai_record_to_kaggle_dict(oai_record))
    return _HarvestOutcome(observed_max, tuple(deleted_ids))


async def _persist_harvest(
    store: SQLiteStore,
    stats: ImportStats,
    outcome: _HarvestOutcome,
    *,
    stored_cursor: str,
    from_override: str | None,
    prune_scope: set[str] | None,
) -> None:
    stats.deleted = await _delete_withdrawn(store, outcome.deleted_ids)
    if prune_scope is not None:
        stats.pruned = await _prune_out_of_scope(store, prune_scope)
    cursor_floor = later_datestamp(stored_cursor, from_override or "") or stored_cursor
    await store.set_meta(
        ARXIV_HARVEST_CURSOR_KEY,
        later_datestamp(cursor_floor, outcome.observed_max or "") or cursor_floor,
    )
    await store.set_meta(ARXIV_HARVEST_STATUS_KEY, "complete")
    await store.set_meta(ARXIV_HARVEST_ERROR_KEY, "")


async def _resolve_set_specs(
    oai_sets: list[OaiSet],
    catalog: CategoryCatalog,
    store: SQLiteStore | None,
    *,
    requested_groups: set[str],
    requested_categories: set[str],
) -> list[str]:
    group_names = top_level_groups(oai_sets)
    spec_by_group_slug = {slugify(name): spec for spec, name in group_names.items()}
    all_specs = sorted(group_names)

    if requested_groups or requested_categories:
        resolved: set[str] = set()
        unknown: list[str] = []
        for group in requested_groups:
            spec = spec_by_group_slug.get(group)
            if spec is None:
                unknown.append(group)
            else:
                resolved.add(spec)
        for code in requested_categories:
            entry = catalog.resolve(code)
            if entry is None or entry.group_slug not in spec_by_group_slug:
                unknown.append(code)
            else:
                resolved.add(spec_by_group_slug[entry.group_slug])
        if unknown:
            raise UnknownFilterError(
                "No arXiv group or category matched the requested filter(s): "
                + ", ".join(sorted(unknown))
            )
        return sorted(resolved)

    corpus_group_slugs = await _arxiv_group_slugs(store)
    default_specs = sorted(
        spec for slug, spec in spec_by_group_slug.items() if slug in corpus_group_slugs
    )
    return default_specs or all_specs


async def _arxiv_group_slugs(store: SQLiteStore | None) -> set[str]:
    if store is None:
        return set()
    result = await store.execute("SELECT DISTINCT tag FROM tags WHERE tag LIKE 'arxiv:group:%'")
    return {str(row["tag"]).removeprefix("arxiv:group:") for row in result.rows}


async def _effective_cursor(store: SQLiteStore | None) -> str:
    if store is not None:
        stored = await store.get_meta(ARXIV_HARVEST_CURSOR_KEY)
        if stored is not None and _is_iso_date(stored):
            return stored
        seeded = await _seed_cursor_from_documents(store)
        if seeded is not None:
            return seeded
    return ARXIV_EARLIEST_DATESTAMP


async def _seed_cursor_from_documents(store: SQLiteStore) -> str | None:
    result = await store.execute(
        """
        SELECT MAX(json_extract(metadata, '$.update_date')) AS cursor
        FROM documents
        WHERE source LIKE 'arxiv:%'
        """
    )
    row = result.first()
    if row is None:
        return None
    value = row["cursor"]
    return str(value) if value else None


async def _delete_withdrawn(store: SQLiteStore, deleted_ids: tuple[str, ...]) -> int:
    if not deleted_ids:
        return 0
    sources = sorted({f"arxiv:{arxiv_id}" for arxiv_id in deleted_ids})
    return await store.delete_documents_by_source(sources)


async def _prune_out_of_scope(store: SQLiteStore, in_scope_categories: set[str]) -> int:
    codes = sorted(in_scope_categories)
    if not codes:
        return 0
    placeholders = ",".join("?" for _ in codes)
    result = await store.execute(
        f"""
        SELECT d.source
        FROM documents d
        WHERE d.source LIKE 'arxiv:%'
          AND NOT EXISTS (
            SELECT 1 FROM tags t
            WHERE t.document_id = d.id AND t.tag IN ({placeholders})
          )
        """,
        tuple(f"arxiv:category:{code}" for code in codes),
    )
    sources = [str(row["source"]) for row in result.rows]
    if not sources:
        return 0
    return await store.delete_documents_by_source(sources)


def harvest_window_start(cursor: str) -> str:
    try:
        start = date.fromisoformat(cursor) - timedelta(days=1)
    except ValueError:
        return ARXIV_EARLIEST_DATESTAMP
    return max(start, date.fromisoformat(ARXIV_EARLIEST_DATESTAMP)).isoformat()


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _requested_filters(options: ImportOptions) -> tuple[set[str], set[str]]:
    return (
        {slugify(group) for group in options.groups},
        {category.lower() for category in options.categories},
    )


def _build_intake(
    options: ImportOptions,
    *,
    catalog: CategoryCatalog,
    requested_groups: set[str],
    requested_categories: set[str],
    import_source: str,
    bulk_ingest: BulkIngestService | None,
) -> _RecordIntake:
    return _RecordIntake(
        catalog=catalog,
        requested_groups=requested_groups,
        requested_categories=requested_categories,
        import_source=import_source,
        import_timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        bulk_ingest=bulk_ingest,
        dry_run=options.dry_run,
        batch_size=options.batch_size,
        limit=options.limit,
        progress_every=options.progress_every,
    )


def _warn_about_name_mismatches(seed: CategoryCatalog, oai_sets: list[OaiSet]) -> None:
    mismatches = category_name_mismatches(seed, oai_sets)
    if not mismatches:
        return
    print(
        f"WARNING: {len(mismatches)} arXiv category name(s) differ from the seed; "
        "the seed name is retained to avoid re-embedding unchanged documents.",
        file=sys.stderr,
    )
    for code, (seed_name, derived_name) in sorted(mismatches.items()):
        print(f"  {code}: seed={seed_name!r} arXiv={derived_name!r}", file=sys.stderr)


@dataclass(slots=True)
class _RecordIntake:
    catalog: CategoryCatalog
    requested_groups: set[str]
    requested_categories: set[str]
    import_source: str
    import_timestamp: str
    bulk_ingest: BulkIngestService | None
    dry_run: bool
    batch_size: int
    limit: int | None
    progress_every: int
    stats: ImportStats = field(default_factory=ImportStats)
    _batch: list[PreparedPaper] = field(default_factory=list[PreparedPaper], init=False)
    _completed: int = field(default=0, init=False)
    _progress: Progress | None = field(default=None, init=False)
    _task: TaskID | None = field(default=None, init=False)

    def limit_reached(self) -> bool:
        return self.limit is not None and self.stats.selected >= self.limit

    def note_skipped(self) -> None:
        self.stats.skipped += 1

    def note_failure(self, message: str) -> None:
        self.stats.failed += 1
        print(message, file=sys.stderr)

    @contextmanager
    def progress_context(self, description: str) -> Generator[None]:
        if self.progress_every <= 0:
            yield
            return
        self._progress = create_progress()
        self._task = self._progress.add_task(
            description, total=None, stats=format_progress_stats(self.stats)
        )
        try:
            with self._progress:
                yield
        finally:
            self._progress = None
            self._task = None

    async def add(self, record: dict[str, Any]) -> None:
        paper = prepare_record(
            record,
            self.catalog,
            import_source=self.import_source,
            import_timestamp=self.import_timestamp,
        )
        if paper is None or not selected_by_filters(
            paper,
            self.catalog,
            requested_groups=self.requested_groups,
            requested_categories=self.requested_categories,
        ):
            self.stats.skipped += 1
        else:
            self.stats.selected += 1
            self.stats.unresolved_categories.update(paper.unresolved_categories)
            self._batch.append(paper)
            if len(self._batch) >= self.batch_size:
                await self._flush()
        self._completed += 1
        if (
            self._progress is not None
            and self._task is not None
            and self._completed % self.progress_every == 0
        ):
            self._render_progress()

    async def finish(self) -> None:
        await self._flush()
        self._render_progress()

    async def _flush(self) -> None:
        if not self._batch:
            return
        batch = self._batch
        self._batch = []
        await flush_batch(
            bulk_ingest=self.bulk_ingest,
            dry_run=self.dry_run,
            batch=batch,
            stats=self.stats,
        )

    def _render_progress(self) -> None:
        if self._progress is None or self._task is None:
            return
        self._progress.update(self._task, stats=format_progress_stats(self.stats))


async def flush_batch(
    *,
    bulk_ingest: BulkIngestService | None,
    dry_run: bool,
    batch: list[PreparedPaper],
    stats: ImportStats,
) -> None:
    items = [paper_to_batch_item(paper) for paper in batch]
    if dry_run:
        if bulk_ingest is None:
            stats.inserted += len(batch)
            return
        output = await bulk_ingest.preview(
            items,
            skip_unchanged=True,
            metadata_compare_exclude=METADATA_COMPARE_EXCLUDE,
        )
        stats.inserted += int(output["inserted"])
        stats.updated += int(output["updated"])
        stats.skipped += int(output["skipped"])
        return

    if bulk_ingest is None:
        raise RuntimeError("import batch requires open Mouseion services")

    try:
        output = await bulk_ingest.ingest(
            items,
            skip_unchanged=True,
            metadata_compare_exclude=METADATA_COMPARE_EXCLUDE,
        )
        stats.inserted += int(output["inserted"])
        stats.updated += int(output["updated"])
        stats.skipped += int(output["skipped"])
    except EmbeddingError:
        raise
    except Exception as exc:  # noqa: BLE001
        stats.failed += len(batch)
        print(f"failed to import batch of {len(batch)} papers: {exc}", file=sys.stderr)


@asynccontextmanager
async def _ingest_services(
    options: ImportOptions, settings: Settings, embedder: Embedder | None
) -> AsyncGenerator[tuple[SQLiteStore | None, BulkIngestService | None]]:
    async with AsyncExitStack() as stack:
        if options.dry_run:
            if not settings.sqlite_path.exists():
                yield None, None
                return
            store = SQLiteStore(settings)
            await store.open_readonly()
            stack.push_async_callback(store.close)
            active_embedder = embedder or Embedder(settings)
            yield store, BulkIngestService(store, Chunker(settings), active_embedder)
            return

        active_embedder = embedder or Embedder(settings)
        await ensure_embedding_backend_ready(active_embedder)
        bundle = await stack.enter_async_context(open_services(settings, embedder=active_embedder))
        yield bundle.store, bundle.bulk_ingest


async def ensure_embedding_backend_ready(embedder: object) -> None:
    ensure_ready = getattr(embedder, "ensure_ready", None)
    if ensure_ready is None:
        return
    await ensure_ready()


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def paper_to_batch_item(paper: PreparedPaper) -> BatchIngestItem:
    return BatchIngestItem(
        content=IngestedContent(
            title=paper.title,
            source=paper.source,
            type=DocumentType.DOCUMENT,
            content=paper.content,
            metadata=paper.metadata,
        ),
        tags=paper.tags,
        chunks=[ChunkText(content=paper.content, token_count=estimate_token_count(paper.content))],
    )


def selected_by_filters(
    paper: PreparedPaper,
    catalog: CategoryCatalog,
    *,
    requested_groups: set[str],
    requested_categories: set[str],
) -> bool:
    if not requested_groups and not requested_categories:
        return True
    record_categories = {code.lower() for code in paper.category_codes}
    category_match = bool(record_categories & requested_categories)
    group_match = catalog.group_matches(requested_groups, paper.category_codes)
    return category_match or group_match


def category_filter_codes(
    catalog: CategoryCatalog,
    *,
    requested_groups: set[str],
    requested_categories: set[str],
) -> set[str]:
    if not requested_groups and not requested_categories:
        return set()
    return requested_categories | {
        code for code, entry in catalog.categories.items() if entry.group_slug in requested_groups
    }


def raw_categories_match_filter(line: bytes, requested_categories: set[str]) -> bool:
    raw_categories = extract_raw_categories(line)
    if raw_categories is None:
        return False
    return any(code.lower() in requested_categories for code in raw_categories.split())


def extract_raw_categories(line: bytes) -> str | None:
    match = CATEGORIES_FIELD_RE.search(line)
    if match is None:
        return None

    raw_value = match.group(1)
    if b"\\" not in raw_value:
        return raw_value.decode("utf-8", errors="replace")

    try:
        loaded: object = json.loads(b'"' + raw_value + b'"')
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, str):
        return loaded
    return None


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def ensure_input_file_exists(path: Path) -> None:
    if path.is_file():
        return
    raise MissingInputFileError(
        f"arXiv metadata JSONL file not found: {path}\n"
        "Please download the Kaggle arXiv metadata dataset from "
        f"{KAGGLE_ARXIV_DOWNLOAD_URL} and place it at that path, "
        "or omit --input to harvest arXiv changes over OAI-PMH."
    )


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.fields[stats]}"),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    )


def format_progress_stats(stats: ImportStats) -> str:
    return (
        f"selected={stats.selected} inserted={stats.inserted} updated={stats.updated} "
        f"skipped={stats.skipped} failed={stats.failed} deleted={stats.deleted}"
    )


def parse_args(argv: list[str] | None = None) -> ImportOptions:
    parser = argparse.ArgumentParser(description="Harvest arXiv metadata into Mouseion documents.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Import a local Kaggle-style JSONL file instead of harvesting OAI-PMH.",
    )
    parser.add_argument("--categories-json", type=Path, default=DEFAULT_CATEGORIES_JSON)
    parser.add_argument("--groups", nargs="+", default=())
    parser.add_argument("--categories", nargs="+", default=())
    parser.add_argument(
        "--from",
        dest="from_date",
        default=None,
        help="Override the OAI start date (YYYY-MM-DD).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.from_date is not None and not _is_iso_date(args.from_date):
        parser.error("--from must be a YYYY-MM-DD date")

    return ImportOptions(
        input=args.input,
        categories_json=args.categories_json,
        groups=tuple(args.groups),
        categories=tuple(args.categories),
        from_date=args.from_date,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        progress_every=args.progress_every,
    )


async def async_main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> int:
    options = parse_args(argv)
    try:
        stats = await import_arxiv_metadata(options, settings=settings, embedder=embedder)
    except MissingInputFileError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (EmbeddingError, OaiError, UnknownFilterError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(stats.as_dict(), indent=2, sort_keys=True))
    return 0 if stats.failed == 0 else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
