"""Import Kaggle arXiv metadata JSONL records as searchable Mouseion documents."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import apsw
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
from mouseion.domain.models import (
    BatchIngestItem,
    ChunkText,
    DocumentType,
    IngestedContent,
    normalize_tags,
)
from mouseion.errors import EmbeddingError
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.embedder import Embedder
from mouseion.ingest.ingestor import Ingestor
from mouseion.ingest.repo import RepoService
from mouseion.services.exporter import Exporter
from mouseion.services.graph import GraphService
from mouseion.services.search import SearchService
from mouseion.services.service import EdgePolicy, MouseionService
from mouseion.storage.db import SQLiteStore, _fetch_one
from mouseion.support.utils import canonical_text_hash, json_loads

DEFAULT_INPUT = Path("raw_data/raw-kaggle-arxiv-metadata-2026-05-29.json")
DEFAULT_CATEGORIES_JSON = Path("utilities/arxiv/categories.json")
KAGGLE_ARXIV_DOWNLOAD_URL = (
    "https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download"
)


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
    edge_policy: EdgePolicy = "recompute-after-insert"


@dataclass(slots=True)
class PreparedPaper:
    arxiv_id: str
    source: str
    title: str
    content: str
    content_hash: str
    tags: list[str]
    metadata: dict[str, Any]
    category_codes: list[str]
    unresolved_categories: list[str]


@dataclass(frozen=True, slots=True)
class ExistingDocumentSnapshot:
    id: str
    source: str
    content_hash: str
    tags: list[str]
    metadata: dict[str, Any]


@dataclass(slots=True)
class ImportStats:
    selected: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    edges_created: int = 0
    unresolved_categories: Counter[str] = field(default_factory=Counter)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "failed": self.failed,
            "edges_created": self.edges_created,
            "unresolved_categories": dict(sorted(self.unresolved_categories.items())),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


class DryRunDocumentLookup:
    def __init__(self, sqlite_path: Path) -> None:
        self.connection: apsw.Connection | None = None
        if sqlite_path.exists():
            self.connection = apsw.Connection(str(sqlite_path), flags=apsw.SQLITE_OPEN_READONLY)

    def exists(self, source: str) -> bool:
        if self.connection is None:
            return False
        row = _fetch_one(
            self.connection,
            "SELECT id FROM documents WHERE type = ? AND source = ? LIMIT 1",
            (str(DocumentType.DOCUMENT), source),
        )
        return row is not None

    def existing_sources(self, sources: list[str]) -> set[str]:
        if self.connection is None or not sources:
            return set()
        return set(self.existing_snapshots(sources))

    def existing_snapshots(self, sources: list[str]) -> dict[str, ExistingDocumentSnapshot]:
        if self.connection is None or not sources:
            return {}
        snapshots: dict[str, ExistingDocumentSnapshot] = {}
        for source_batch in batched(sources, 900):
            placeholders = ",".join("?" for _ in source_batch)
            rows = self.connection.cursor().execute(
                f"""
                {_existing_snapshot_select_sql()}
                WHERE d.type = ? AND d.source IN ({placeholders})
                """,
                (str(DocumentType.DOCUMENT), *source_batch),
            )
            snapshots.update(snapshot_from_row(row) for row in rows)
        return snapshots

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def load_category_catalog(path: Path) -> CategoryCatalog:
    raw = json.loads(path.read_text(encoding="utf-8"))
    categories: dict[str, CategoryEntry] = {}
    groups: dict[str, str] = {}

    if not isinstance(raw, list):
        raise ValueError(f"Expected {path} to contain a list of arXiv category groups")

    for group_item in raw:
        if not isinstance(group_item, dict):
            continue
        for group_name, entries in group_item.items():
            group_slug = slugify(str(group_name))
            groups[group_slug] = str(group_name)
            if not isinstance(entries, list):
                continue
            for entry_item in entries:
                if not isinstance(entry_item, dict):
                    continue
                for code, details in entry_item.items():
                    if not isinstance(details, dict):
                        continue
                    category = CategoryEntry(
                        code=str(code),
                        name=str(details.get("name") or ""),
                        description=str(details.get("description") or ""),
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
        content_hash=canonical_text_hash(content),
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
    import_timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    import_source = str(options.input)

    store: SQLiteStore | None = None
    service: MouseionService | None = None
    dry_lookup: DryRunDocumentLookup | None = None
    active_embedder = embedder
    progress = create_progress() if options.progress_every > 0 else None
    scan_task: TaskID | None = None
    input_size = options.input.stat().st_size

    if options.dry_run:
        dry_lookup = DryRunDocumentLookup(settings.sqlite_path)
    else:
        active_embedder = active_embedder or Embedder(settings)
        await ensure_embedding_backend_ready(active_embedder)
        store = SQLiteStore(settings)
        await store.open()
        service = MouseionService(
            store,
            Ingestor(settings),
            Chunker(settings),
            active_embedder,
            SearchService(store, active_embedder, settings.rrf_k),
            GraphService(store, settings),
            RepoService(settings),
            Exporter(settings, store),
        )

    try:
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
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        stats.failed += 1
                        print(f"line {line_number}: invalid JSON: {exc}", file=sys.stderr)
                        continue
                    if not isinstance(record, dict):
                        stats.skipped += 1
                        continue

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
                            service=service,
                            dry_lookup=dry_lookup,
                            batch=batch,
                            stats=stats,
                            edge_policy=_batch_edge_policy(options.edge_policy),
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
                    service=service,
                    dry_lookup=dry_lookup,
                    batch=batch,
                    stats=stats,
                    edge_policy=_batch_edge_policy(options.edge_policy),
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
            if (
                service is not None
                and options.edge_policy == "recompute-after-insert"
                and stats.inserted + stats.updated > 0
            ):
                recompute = await service.recompute_edges()
                stats.edges_created = int(recompute["edges_created"])
    finally:
        if store is not None:
            await store.close()
        if dry_lookup is not None:
            dry_lookup.close()

    stats.elapsed_seconds = time.monotonic() - start
    return stats


async def flush_batch(
    *,
    service: MouseionService | None,
    dry_lookup: DryRunDocumentLookup | None,
    batch: list[PreparedPaper],
    stats: ImportStats,
    edge_policy: EdgePolicy,
) -> None:
    if dry_lookup is not None:
        existing = dry_lookup.existing_snapshots([paper.source for paper in batch])
        for paper in batch:
            snapshot = existing.get(paper.source)
            if snapshot is None:
                stats.inserted += 1
            elif paper_matches_snapshot(paper, snapshot):
                stats.skipped += 1
            else:
                stats.updated += 1
        return

    if service is None:
        raise RuntimeError("import batch requires an open Mouseion service")

    try:
        output = await service.batch_ingest(
            [paper_to_batch_item(paper) for paper in batch],
            edge_policy=edge_policy,
            skip_unchanged=True,
            metadata_compare_exclude={"import_timestamp"},
        )
        stats.inserted += int(output["inserted"])
        stats.updated += int(output["updated"])
        stats.skipped += int(output["skipped"])
        stats.edges_created += int(output["edges_created"])
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


def batched[T](items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


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


def _batch_edge_policy(edge_policy: EdgePolicy) -> EdgePolicy:
    if edge_policy == "recompute-after-insert":
        return "skip"
    return edge_policy


def _existing_snapshot_select_sql() -> str:
    return """
    SELECT d.id,
           d.source,
           d.content_hash,
           d.metadata,
           COALESCE(
             (
               SELECT json_group_array(tag)
               FROM (SELECT tag FROM tags WHERE document_id = d.id ORDER BY tag)
             ),
             '[]'
           ) AS tags
    FROM documents d
    """


def snapshot_from_row(row: tuple[Any, ...]) -> tuple[str, ExistingDocumentSnapshot]:
    source = str(row[1])
    return (
        source,
        ExistingDocumentSnapshot(
            id=str(row[0]),
            source=source,
            content_hash=str(row[2]),
            metadata=json_loads(str(row[3] or "")),
            tags=list(json.loads(str(row[4] or "[]"))),
        ),
    )


def paper_matches_snapshot(paper: PreparedPaper, snapshot: ExistingDocumentSnapshot) -> bool:
    return (
        paper.content_hash == snapshot.content_hash
        and sorted(paper.tags) == sorted(snapshot.tags)
        and comparable_metadata(paper.metadata) == comparable_metadata(snapshot.metadata)
    )


def comparable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "import_timestamp"}


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
    parser.add_argument(
        "--edge-policy",
        choices=("skip", "incremental", "recompute-after-insert"),
        default="recompute-after-insert",
        help="How to create similarity graph edges after bulk ingest.",
    )
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
        edge_policy=args.edge_policy,
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
