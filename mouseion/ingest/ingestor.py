from __future__ import annotations

import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import fitz
import httpx
import magic
import trafilatura

from mouseion.config import Settings
from mouseion.domain.models import DocumentType, IngestedContent
from mouseion.errors import MouseionError, ScannedPDFError, UnsupportedFileTypeError
from mouseion.support.utils import file_hash, safe_filename

MIN_PDF_CHARS = 500


@dataclass(slots=True)
class Ingestor:
    settings: Settings

    async def fetch_url(self, url: str) -> IngestedContent:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
        extracted = trafilatura.extract(
            response.text,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            url=url,
        )
        if not extracted or not extracted.strip():
            raise MouseionError(f"No readable text could be extracted from URL: {url}")
        title = _title_from_html(response.text) or _title_from_url(url)
        return IngestedContent(
            title=title,
            source=url,
            type=DocumentType.URL,
            content=extracted.strip(),
            metadata={"url": url, "content_type": response.headers.get("content-type", "")},
        )

    async def read_file(self, file_path: str) -> IngestedContent:
        source_path = Path(file_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"File does not exist: {source_path}")
        mime = self._detect_mime(source_path)
        suffix = source_path.suffix.lower()
        if mime == "application/pdf" or suffix == ".pdf":
            content = self._read_pdf(source_path)
        elif mime in {"text/markdown", "text/plain"} or suffix in {".md", ".markdown", ".txt"}:
            content = source_path.read_text(encoding="utf-8")
        elif mime in {"text/html", "application/xhtml+xml"} or suffix in {".html", ".htm"}:
            html = source_path.read_text(encoding="utf-8", errors="replace")
            extracted = trafilatura.extract(html, output_format="markdown", include_tables=True)
            if not extracted:
                raise MouseionError(
                    f"No readable text could be extracted from HTML file: {source_path}"
                )
            content = extracted
        else:
            raise UnsupportedFileTypeError(f"Unsupported file type {mime!r} for {source_path}")

        stored_path = self._copy_original(source_path)
        return IngestedContent(
            title=source_path.stem,
            source=str(source_path),
            type=DocumentType.FILE,
            content=content.strip(),
            metadata={
                "original_path": str(source_path),
                "stored_path": str(stored_path),
                "mime_type": mime,
                "file_hash": file_hash(source_path),
            },
        )

    def memory(self, content: str) -> IngestedContent:
        trimmed = content.strip()
        if not trimmed:
            raise MouseionError("Memory content cannot be empty")
        return IngestedContent(
            title=trimmed.splitlines()[0][:80] or "Memory",
            source="memory",
            type=DocumentType.MEMORY,
            content=trimmed,
            metadata={},
        )

    def _detect_mime(self, path: Path) -> str:
        try:
            return magic.from_file(str(path), mime=True)
        except Exception:
            guessed, _ = mimetypes.guess_type(path)
            return guessed or "application/octet-stream"

    def _read_pdf(self, path: Path) -> str:
        chunks = []
        with fitz.open(path) as document:
            for page in document:
                chunks.append(page.get_text("text"))
        text = "\n\n".join(chunks).strip()
        if len(text) < MIN_PDF_CHARS:
            raise ScannedPDFError(
                f"PDF has only {len(text)} extracted characters; OCR is out of scope: {path}"
            )
        return text

    def _copy_original(self, source_path: Path) -> Path:
        self.settings.files_dir.mkdir(parents=True, exist_ok=True)
        destination = self.settings.files_dir / (
            f"{file_hash(source_path)[:16]}-{safe_filename(source_path.name, 'file')}"
        )
        if not destination.exists():
            shutil.copy2(source_path, destination)
        return destination


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").split("/")[-1]
    return path or parsed.netloc or url


def _title_from_html(html: str) -> str | None:
    lower = html.lower()
    start = lower.find("<title>")
    end = lower.find("</title>")
    if start == -1 or end == -1 or end <= start:
        return None
    return html[start + len("<title>") : end].strip()
