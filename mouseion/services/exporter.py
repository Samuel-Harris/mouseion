from __future__ import annotations

from dataclasses import dataclass

from mouseion.config import Settings
from mouseion.domain.models import utc_now
from mouseion.storage.db import SQLiteStore
from mouseion.support.utils import safe_filename, timestamp_for_path


@dataclass(slots=True)
class Exporter:
    settings: Settings
    store: SQLiteStore

    async def export(self) -> dict[str, str | int]:
        timestamp = utc_now()
        export_path = self.settings.export_dir / timestamp_for_path(timestamp)
        export_path.mkdir(parents=True, exist_ok=True)
        documents, total = await self.store.list_documents(
            type_filter="all", limit=100_000, offset=0
        )
        for index, document in enumerate(documents, start=1):
            chunks = await self.store.get_chunks_for_document(document.id)
            filename = f"{index:04d}-{safe_filename(document.title, str(document.id))}.md"
            path = export_path / filename
            frontmatter = [
                "---",
                f"id: {document.id}",
                f"type: {document.type}",
                f"title: {document.title}",
                f"source: {document.source}",
                f"content_hash: {document.content_hash}",
                f"created_at: {document.created_at.isoformat()}",
                f"updated_at: {document.updated_at.isoformat()}",
                f"tags: {document.tags}",
                "---",
                "",
            ]
            body = "\n\n".join(chunk.content for chunk in chunks)
            path.write_text("\n".join(frontmatter) + body + "\n", encoding="utf-8")
        return {"export_path": str(export_path), "document_count": total}
