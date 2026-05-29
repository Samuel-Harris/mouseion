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
from uuid import uuid4

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
from mouseion.domain.models import DocumentType, normalize_tags
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.embedder import Embedder
from mouseion.storage.db import SQLiteStore, _embedding_blob, _fetch_one, _to_db_timestamp
from mouseion.support.utils import canonical_text_hash, json_dumps

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


@dataclass(slots=True)
class ImportStats:
    selected: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    unresolved_categories: Counter[str] = field(default_factory=Counter)
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
    resolved = [entry for code in category_codes if (entry := catalog.resolve(code)) is not None]
    unresolved = [code for code in category_codes if catalog.resolve(code) is None]

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
    dry_lookup: DryRunDocumentLookup | None = None
    chunker: Chunker | None = None
    active_embedder = embedder
    progress = create_progress() if options.progress_every > 0 else None
    scan_task: TaskID | None = None
    selected_task: TaskID | None = None
    input_size = options.input.stat().st_size

    if options.dry_run:
        dry_lookup = DryRunDocumentLookup(settings.sqlite_path)
    else:
        store = SQLiteStore(settings)
        await store.open()
        chunker = Chunker(settings)
        active_embedder = active_embedder or Embedder(settings)

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
                            selected_task,
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

                    if options.dry_run:
                        if dry_lookup is not None and dry_lookup.exists(paper.source):
                            stats.updated += 1
                        else:
                            stats.inserted += 1
                    else:
                        batch.append(paper)
                        if len(batch) >= options.batch_size:
                            await flush_batch(store, chunker, active_embedder, batch, stats)
                            batch = []

                    update_progress(
                        progress,
                        scan_task,
                        selected_task,
                        pending_bytes=pending_bytes,
                        stats=stats,
                    )
                    pending_bytes = 0
                    if options.limit is not None and stats.selected >= options.limit:
                        break

            if batch:
                await flush_batch(store, chunker, active_embedder, batch, stats)
            if pending_bytes:
                update_progress(
                    progress,
                    scan_task,
                    selected_task,
                    pending_bytes=pending_bytes,
                    stats=stats,
                )
            else:
                update_progress(
                    progress,
                    scan_task,
                    selected_task,
                    pending_bytes=0,
                    stats=stats,
                )
    finally:
        if store is not None:
            await store.close()
        if dry_lookup is not None:
            dry_lookup.close()

    stats.elapsed_seconds = time.monotonic() - start
    return stats


async def flush_batch(
    store: SQLiteStore | None,
    chunker: Chunker | None,
    embedder: Embedder | None,
    batch: list[PreparedPaper],
    stats: ImportStats,
) -> None:
    if store is None or chunker is None or embedder is None:
        raise RuntimeError("import batch requires an open store, chunker, and embedder")

    try:
        chunk_rows_by_paper = [chunker.chunk(paper.content) for paper in batch]
        flat_chunk_texts = [chunk.content for chunks in chunk_rows_by_paper for chunk in chunks]
        embeddings = await embedder.embed_many(flat_chunk_texts)
        embedded_batches: list[list[tuple[str, int, list[float]]]] = []
        offset = 0
        for chunks in chunk_rows_by_paper:
            count = len(chunks)
            chunk_embeddings = embeddings[offset : offset + count]
            offset += count
            embedded_batches.append(
                [
                    (chunk.content, chunk.token_count, embedding)
                    for chunk, embedding in zip(chunks, chunk_embeddings, strict=True)
                ]
            )
        inserted, updated = await bulk_upsert(store, batch, embedded_batches)
        stats.inserted += inserted
        stats.updated += updated
    except Exception as exc:  # noqa: BLE001
        stats.failed += len(batch)
        print(f"failed to import batch of {len(batch)} papers: {exc}", file=sys.stderr)


async def bulk_upsert(
    store: SQLiteStore,
    papers: list[PreparedPaper],
    chunk_batches: list[list[tuple[str, int, list[float]]]],
) -> tuple[int, int]:
    now = datetime.now(tz=UTC)

    def run(conn: apsw.Connection) -> tuple[int, int]:
        inserted = 0
        updated = 0
        for paper, chunks in zip(papers, chunk_batches, strict=True):
            existing = _fetch_one(
                conn,
                "SELECT id FROM documents WHERE type = ? AND source = ? LIMIT 1",
                (str(DocumentType.DOCUMENT), paper.source),
            )
            if existing is None:
                document_id = str(uuid4())
                inserted += 1
                conn.execute(
                    """
                    INSERT INTO documents(
                        id, type, title, source, content_hash, created_at, updated_at, metadata
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        str(DocumentType.DOCUMENT),
                        paper.title,
                        paper.source,
                        paper.content_hash,
                        _to_db_timestamp(now),
                        _to_db_timestamp(now),
                        json_dumps(paper.metadata),
                    ),
                )
            else:
                document_id = str(existing["id"])
                updated += 1
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM tags WHERE document_id = ?", (document_id,))
                conn.execute(
                    """
                    UPDATE documents
                    SET type = ?,
                        title = ?,
                        source = ?,
                        content_hash = ?,
                        updated_at = ?,
                        metadata = ?
                    WHERE id = ?
                    """,
                    (
                        str(DocumentType.DOCUMENT),
                        paper.title,
                        paper.source,
                        paper.content_hash,
                        _to_db_timestamp(now),
                        json_dumps(paper.metadata),
                        document_id,
                    ),
                )

            for tag in paper.tags:
                conn.execute(
                    "INSERT OR IGNORE INTO tags(document_id, tag) VALUES(?, ?)",
                    (document_id, tag),
                )

            for chunk_index, (content, token_count, embedding) in enumerate(chunks):
                conn.execute(
                    """
                    INSERT INTO chunks(document_id, content, chunk_index, token_count, created_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (document_id, content, chunk_index, token_count, _to_db_timestamp(now)),
                )
                chunk_id = conn.last_insert_rowid()
                conn.execute(
                    "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES(?, ?)",
                    (chunk_id, _embedding_blob(embedding)),
                )

        return inserted, updated

    result = await store.write(run)
    return int(result[0]), int(result[1])


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
    selected_task: TaskID | None,
    *,
    pending_bytes: int,
    stats: ImportStats,
) -> None:
    if progress is None or scan_task is None:
        return
    if pending_bytes:
        progress.advance(scan_task, pending_bytes)
    progress.update(scan_task, stats=format_progress_stats(stats))
    if selected_task is not None:
        progress.update(selected_task, completed=stats.selected, stats=format_progress_stats(stats))


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


async def async_main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    try:
        stats = await import_arxiv_metadata(options)
    except MissingInputFileError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(stats.as_dict(), indent=2, sort_keys=True))
    return 0 if stats.failed == 0 else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
