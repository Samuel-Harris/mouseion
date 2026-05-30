from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mouseion.config import Settings
from mouseion.ingest.chunker import Chunker
from mouseion.ingest.embedder import Embedder
from mouseion.ingest.ingestor import Ingestor
from mouseion.ingest.repo import RepoService
from mouseion.services.bulk_ingest import BulkIngestService
from mouseion.services.document_reader import DocumentReader
from mouseion.services.exporter import Exporter
from mouseion.services.search import SearchService
from mouseion.services.service import MouseionService
from mouseion.storage.db import SQLiteStore


@dataclass(slots=True)
class ServiceBundle:
    store: SQLiteStore
    service: MouseionService
    bulk_ingest: BulkIngestService


@asynccontextmanager
async def open_services(
    settings: Settings | None = None, *, embedder: Any | None = None
) -> AsyncGenerator[ServiceBundle]:
    active_settings = settings or Settings()
    active_settings.ensure_directories()
    store = SQLiteStore(active_settings)
    await store.open()
    try:
        active_embedder = embedder or Embedder(active_settings)
        chunker = Chunker(active_settings)
        searcher = SearchService(store, active_embedder, active_settings.rrf_k)
        document_reader = DocumentReader(store, searcher)
        service = MouseionService(
            store=store,
            ingestor=Ingestor(active_settings),
            chunker=chunker,
            embedder=active_embedder,
            searcher=searcher,
            repos=RepoService(active_settings),
            exporter=Exporter(active_settings, store),
            document_reader=document_reader,
        )
        yield ServiceBundle(
            store=store,
            service=service,
            bulk_ingest=BulkIngestService(store, chunker, active_embedder),
        )
    finally:
        await store.close()
