from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from urllib.parse import unquote, urlparse

import fitz
import httpx
import magic
import trafilatura

from mouseion.config import Settings
from mouseion.domain.models import DocumentType, IngestedContent
from mouseion.errors import MouseionError, ScannedPDFError, UnsupportedFileTypeError
from mouseion.support.utils import file_hash, safe_filename

MIN_PDF_CHARS = 500
URL_FETCH_MAX_ATTEMPTS = 3
URL_FETCH_BACKOFF_SECONDS = 0.5
URL_FETCH_RETRY_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class Ingestor:
    settings: Settings

    async def fetch_url(self, url: str) -> IngestedContent:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await _get_url_with_retries(client, url)
        content_type = _content_type(response.headers.get("content-type", ""))
        if _is_supported_url_file(url, content_type):
            return self._read_url_file(url, response.content, response.headers, content_type)
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

    def _read_url_file(
        self, url: str, content: bytes, headers: httpx.Headers, content_type: str
    ) -> IngestedContent:
        if not content:
            raise MouseionError(f"Downloaded file was empty: {url}")
        stored_path, digest = self._store_url_file(url, content, headers, content_type)
        suffix = stored_path.suffix.lower()
        if content_type == "application/pdf" or suffix == ".pdf":
            extracted = self._read_pdf(stored_path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported URL file type {content_type!r}: {url}")

        title = _title_from_filename(stored_path.name) or _title_from_url(url)
        return IngestedContent(
            title=title,
            source=url,
            type=DocumentType.URL,
            content=extracted.strip(),
            metadata={
                "url": url,
                "content_type": headers.get("content-type", ""),
                "stored_path": str(stored_path),
                "mime_type": content_type,
                "file_hash": digest,
            },
        )

    def _store_url_file(
        self, url: str, content: bytes, headers: httpx.Headers, content_type: str
    ) -> tuple[Path, str]:
        self.settings.files_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        filename = _filename_from_response(url, headers, content_type)
        destination = self.settings.files_dir / (
            f"{digest[:16]}-{safe_filename(filename, 'download')}"
        )
        if not destination.exists():
            destination.write_bytes(content)
        return destination, digest


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").split("/")[-1]
    return path or parsed.netloc or url


def _title_from_filename(filename: str) -> str:
    name = filename
    if "-" in name:
        prefix, candidate = name.split("-", 1)
        if len(prefix) == 16 and all(character in "0123456789abcdef" for character in prefix):
            name = candidate
    return Path(name).stem


def _title_from_html(html: str) -> str | None:
    lower = html.lower()
    start = lower.find("<title>")
    end = lower.find("</title>")
    if start == -1 or end == -1 or end <= start:
        return None
    return html[start + len("<title>") : end].strip()


def _content_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _is_supported_url_file(url: str, content_type: str) -> bool:
    suffix = Path(urlparse(url).path).suffix.lower()
    return content_type == "application/pdf" or suffix == ".pdf"


def _filename_from_response(url: str, headers: httpx.Headers, content_type: str) -> str:
    filename = _filename_from_content_disposition(headers.get("content-disposition", ""))
    if not filename:
        parsed = urlparse(url)
        filename = unquote(Path(parsed.path).name)
    if not filename:
        filename = "download"

    suffix = Path(filename).suffix
    if content_type == "application/pdf" and suffix.lower() != ".pdf":
        filename = f"{filename}.pdf"
    elif not suffix:
        extension = mimetypes.guess_extension(content_type) or ""
        filename = f"{filename}{extension}"
    return filename


def _filename_from_content_disposition(value: str) -> str | None:
    if not value:
        return None
    message = Message()
    message["content-disposition"] = value
    filename = message.get_filename()
    return Path(filename).name if filename else None


async def _get_url_with_retries(client: httpx.AsyncClient, url: str) -> httpx.Response:
    for attempt in range(URL_FETCH_MAX_ATTEMPTS):
        try:
            response = await client.get(url)
            if _is_retryable_status_code(response.status_code):
                if _is_final_url_fetch_attempt(attempt):
                    response.raise_for_status()
                await _sleep_before_url_retry(attempt)
                continue

            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            retryable_status = _is_retryable_status_code(exc.response.status_code)
            if not retryable_status or _is_final_url_fetch_attempt(attempt):
                raise
            await _sleep_before_url_retry(attempt)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError):
            if _is_final_url_fetch_attempt(attempt):
                raise
            await _sleep_before_url_retry(attempt)

    raise RuntimeError("URL fetch retry loop exited unexpectedly")


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in URL_FETCH_RETRY_STATUS_CODES


def _is_final_url_fetch_attempt(attempt: int) -> bool:
    return attempt >= URL_FETCH_MAX_ATTEMPTS - 1


async def _sleep_before_url_retry(attempt: int) -> None:
    await asyncio.sleep(URL_FETCH_BACKOFF_SECONDS * (2**attempt))
