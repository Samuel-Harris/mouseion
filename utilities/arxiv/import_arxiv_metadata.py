"""Import Kaggle arXiv metadata JSONL records as searchable Mouseion documents."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from mouseion.config import Settings
from mouseion.domain.models import ChunkText, DocumentType, IngestedContent, normalize_tags
from mouseion.errors import EmbeddingError
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.embedder import Embedder
from mouseion.services.bulk_ingest import (
    BatchIngestItem,
    BulkIngestService,
)
from mouseion.services.factory import open_services
from mouseion.storage.db import SQLiteStore

DEFAULT_INPUT = Path("raw_data/raw-kaggle-arxiv-metadata-2026-05-29.json")
DEFAULT_CATEGORIES_JSON = Path("utilities/arxiv/categories.json")
KAGGLE_ARXIV_DOWNLOAD_URL = (
    "https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download"
)
CATEGORIES_FIELD_RE = re.compile(rb'"categories"\s*:\s*"((?:[^"\\]|\\.)*)"')


class MissingInputFileError(FileNotFoundError):
    pass


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


@dataclass(frozen=True, slots=True)
class ImportOptions:
    input: Path = DEFAULT_INPUT
    categories_json: Path = DEFAULT_CATEGORIES_JSON
    groups: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
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
    unresolved_categories: Counter[str] = field(default_factory=Counter[str])
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "failed": self.failed,
            "unresolved_categories": dict(sorted(self.unresolved_categories.items())),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


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
    stats = ImportStats()
    settings = settings or Settings()
    ensure_input_file_exists(options.input)
    catalog = load_category_catalog(options.categories_json)
    requested_groups = {slugify(group) for group in options.groups}
    requested_categories = {category.lower() for category in options.categories}
    requested_filter_categories = category_filter_codes(
        catalog,
        requested_groups=requested_groups,
        requested_categories=requested_categories,
    )
    import_timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    import_source = str(options.input)

    dry_store: SQLiteStore | None = None
    bulk_ingest: BulkIngestService | None = None
    active_embedder = embedder
    progress = create_progress() if options.progress_every > 0 else None
    scan_task: TaskID | None = None
    input_size = options.input.stat().st_size

    async with AsyncExitStack() as stack:
        if options.dry_run:
            if settings.sqlite_path.exists():
                dry_store = SQLiteStore(settings)
                await dry_store.open_readonly()
                stack.push_async_callback(dry_store.close)
                dry_embedder = active_embedder or Embedder(settings)
                bulk_ingest = BulkIngestService(
                    dry_store,
                    Chunker(settings),
                    dry_embedder,
                )
        else:
            active_embedder = active_embedder or Embedder(settings)
            await ensure_embedding_backend_ready(active_embedder)
            services = await stack.enter_async_context(
                open_services(settings, embedder=active_embedder)
            )
            bulk_ingest = services.bulk_ingest

        batch: list[PreparedPaper] = []
        progress_context = progress if progress is not None else nullcontext()
        with progress_context:
            if progress is not None:
                scan_task = progress.add_task(
                    "Scanning arXiv JSONL",
                    total=input_size,
                    stats=format_progress_stats(stats),
                )
            pending_bytes = 0
            with options.input.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line_size = len(line)
                    pending_bytes += line_size
                    if options.progress_every > 0 and line_number % options.progress_every == 0:
                        update_progress(
                            progress,
                            scan_task,
                            pending_bytes=pending_bytes,
                            stats=stats,
                        )
                        pending_bytes = 0

                    if options.limit is not None and stats.selected >= options.limit:
                        break
                    line = line.strip()
                    if not line:
                        stats.skipped += 1
                        continue
                    if requested_filter_categories and not raw_categories_match_filter(
                        line, requested_filter_categories
                    ):
                        stats.skipped += 1
                        continue
                    try:
                        loaded_record: object = json.loads(line)
                    except json.JSONDecodeError as exc:
                        stats.failed += 1
                        print(f"line {line_number}: invalid JSON: {exc}", file=sys.stderr)
                        continue
                    if not isinstance(loaded_record, dict):
                        stats.skipped += 1
                        continue
                    record = cast(dict[str, Any], loaded_record)

                    paper = prepare_record(
                        record,
                        catalog,
                        import_source=import_source,
                        import_timestamp=import_timestamp,
                    )
                    if paper is None:
                        stats.skipped += 1
                        continue
                    if not selected_by_filters(
                        paper,
                        catalog,
                        requested_groups=requested_groups,
                        requested_categories=requested_categories,
                    ):
                        stats.skipped += 1
                        continue

                    stats.selected += 1
                    stats.unresolved_categories.update(paper.unresolved_categories)
                    batch.append(paper)

                    if len(batch) >= options.batch_size:
                        await flush_batch(
                            bulk_ingest=bulk_ingest,
                            dry_run=options.dry_run,
                            batch=batch,
                            stats=stats,
                        )
                        batch = []
                        update_progress(
                            progress,
                            scan_task,
                            pending_bytes=pending_bytes,
                            stats=stats,
                        )
                        pending_bytes = 0
                    if options.limit is not None and stats.selected >= options.limit:
                        break

            if batch:
                await flush_batch(
                    bulk_ingest=bulk_ingest,
                    dry_run=options.dry_run,
                    batch=batch,
                    stats=stats,
                )
            if pending_bytes:
                update_progress(
                    progress,
                    scan_task,
                    pending_bytes=pending_bytes,
                    stats=stats,
                )
            else:
                update_progress(
                    progress,
                    scan_task,
                    pending_bytes=0,
                    stats=stats,
                )
    stats.elapsed_seconds = time.monotonic() - start
    return stats


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
            metadata_compare_exclude={"import_timestamp"},
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
            metadata_compare_exclude={"import_timestamp"},
        )
        stats.inserted += int(output["inserted"])
        stats.updated += int(output["updated"])
        stats.skipped += int(output["skipped"])
    except EmbeddingError:
        raise
    except Exception as exc:  # noqa: BLE001
        stats.failed += len(batch)
        print(f"failed to import batch of {len(batch)} papers: {exc}", file=sys.stderr)


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


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def ensure_input_file_exists(path: Path) -> None:
    if path.is_file():
        return
    raise MissingInputFileError(
        f"arXiv metadata JSONL file not found: {path}\n"
        "Please download the Kaggle arXiv metadata dataset from "
        f"{KAGGLE_ARXIV_DOWNLOAD_URL} and place it at that path, "
        "or pass --input with the downloaded JSONL file path."
    )


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(binary_units=True),
        TextColumn("{task.fields[stats]}"),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    )


def update_progress(
    progress: Progress | None,
    scan_task: TaskID | None,
    *,
    pending_bytes: int,
    stats: ImportStats,
) -> None:
    if progress is None or scan_task is None:
        return
    if pending_bytes:
        progress.advance(scan_task, pending_bytes)
    progress.update(scan_task, stats=format_progress_stats(stats))


def format_progress_stats(stats: ImportStats) -> str:
    return (
        f"selected={stats.selected} inserted={stats.inserted} updated={stats.updated} "
        f"skipped={stats.skipped} failed={stats.failed}"
    )


def parse_args(argv: list[str] | None = None) -> ImportOptions:
    parser = argparse.ArgumentParser(
        description="Import Kaggle arXiv metadata JSONL into Mouseion documents."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--categories-json", type=Path, default=DEFAULT_CATEGORIES_JSON)
    parser.add_argument("--groups", nargs="+", default=())
    parser.add_argument("--categories", nargs="+", default=())
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

    return ImportOptions(
        input=args.input,
        categories_json=args.categories_json,
        groups=tuple(args.groups),
        categories=tuple(args.categories),
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
    except EmbeddingError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(stats.as_dict(), indent=2, sort_keys=True))
    return 0 if stats.failed == 0 else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
