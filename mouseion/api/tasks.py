from __future__ import annotations

import asyncio
from contextlib import suppress

from mouseion.config import Settings
from mouseion.services.graph import GraphService
from mouseion.support.logging_config import get_logger


class BackgroundTasks:
    def __init__(self, settings: Settings, graph: GraphService) -> None:
        self.settings = settings
        self.graph = graph
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        if self.settings.similar_edge_recompute_hours > 0:
            self._tasks.append(asyncio.create_task(self._similarity_recompute_loop()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task

    async def _similarity_recompute_loop(self) -> None:
        logger = get_logger(__name__)
        interval = self.settings.similar_edge_recompute_hours * 60 * 60
        while True:
            await asyncio.sleep(interval)
            try:
                await self.graph.recompute_all()
            except Exception:
                logger.exception("background_similarity_recompute_failed")
