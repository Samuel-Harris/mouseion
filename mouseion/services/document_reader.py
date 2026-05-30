from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from mouseion.domain.models import (
    Chunk,
    Document,
    DocumentOutlineInput,
    ReadDocumentInput,
    SearchDocumentInput,
    SearchFilter,
)
from mouseion.errors import DocumentNotFoundError, InvalidCursorError
from mouseion.services.search import SearchService
from mouseion.storage.db import SQLiteStore

JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ReadCursor:
    chunk_index: int
    char_offset: int


@dataclass(frozen=True, slots=True)
class _ReadPage:
    text: str
    next: _ReadCursor | None
    end: _ReadCursor
    chunks_returned: int


@dataclass(slots=True)
class DocumentReader:
    store: SQLiteStore
    searcher: SearchService

    async def read_document(self, input: ReadDocumentInput) -> JsonDict:
        document = await self._require_document(input.document_id)
        stats = await self.store.document_chunk_stats(input.document_id)
        cursor = _decode_read_cursor(input.cursor)
        chunks = await self.store.get_text_chunks_for_document(
            input.document_id,
            start_chunk_index=cursor.chunk_index,
            limit=input.max_chunks + 1,
        )
        if input.cursor is not None:
            _validate_read_cursor(cursor, chunks, stats.total_chunks)
        page = _page_document_text(
            chunks, cursor, max_chars=input.max_chars, max_chunks=input.max_chunks
        )
        next_cursor = _encode_read_cursor(page.next) if page.next is not None else None
        return {
            "document": _document_payload(document, include_metadata=input.include_metadata),
            "text": page.text,
            "next_cursor": next_cursor,
            "pagination": {
                "cursor": input.cursor,
                "next_cursor": next_cursor,
                "max_chars": input.max_chars,
                "max_chunks": input.max_chunks,
                "chunks_returned": page.chunks_returned,
                "start_chunk_index": cursor.chunk_index,
                "start_char_offset": cursor.char_offset,
                "end_chunk_index": page.end.chunk_index,
                "end_char_offset": page.end.char_offset,
                "total_chunks": stats.total_chunks,
                "total_chars": stats.total_chars,
                "total_tokens": stats.total_tokens,
            },
        }

    async def document_outline(self, input: DocumentOutlineInput) -> JsonDict:
        document = await self._require_document(input.document_id)
        stats = await self.store.document_chunk_stats(input.document_id)
        chunks = await self.store.get_text_chunks_for_document(input.document_id)
        return {
            "document": _document_summary(document),
            "title": document.title,
            "source": document.source,
            "total_chunks": stats.total_chunks,
            "total_chars": stats.total_chars,
            "total_tokens": stats.total_tokens,
            "headings": _derive_headings(chunks),
        }

    async def search_document(self, input: SearchDocumentInput) -> JsonDict:
        await self._require_document(input.document_id)
        return await self.searcher.search(
            input.query,
            top_k=input.top_k,
            filter=SearchFilter(document_id=input.document_id),
        )

    async def _require_document(self, document_id: UUID) -> Document:
        document = await self.store.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        return document


def _document_payload(document: Document, *, include_metadata: bool) -> JsonDict:
    payload = document.model_dump(mode="json")
    if not include_metadata:
        payload.pop("metadata", None)
    return payload


def _document_summary(document: Document) -> JsonDict:
    return {
        "id": str(document.id),
        "type": str(document.type),
        "title": document.title,
        "source": document.source,
        "tags": document.tags,
    }


def _decode_read_cursor(value: str | None) -> _ReadCursor:
    if value is None:
        return _ReadCursor(chunk_index=0, char_offset=0)
    try:
        padded = value + ("=" * (-len(value) % 4))
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        decoded: object = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise InvalidCursorError("Invalid document cursor: expected base64url JSON.") from None
    if not isinstance(decoded, dict):
        raise InvalidCursorError("Invalid document cursor: expected an object.")
    payload = cast(dict[str, object], decoded)
    chunk_index = payload.get("chunk_index")
    char_offset = payload.get("char_offset")
    if (
        not isinstance(chunk_index, int)
        or isinstance(chunk_index, bool)
        or not isinstance(char_offset, int)
        or isinstance(char_offset, bool)
        or chunk_index < 0
        or char_offset < 0
    ):
        raise InvalidCursorError(
            "Invalid document cursor: chunk_index and char_offset must be non-negative integers."
        )
    return _ReadCursor(chunk_index=chunk_index, char_offset=char_offset)


def _encode_read_cursor(cursor: _ReadCursor) -> str:
    payload = {"chunk_index": cursor.chunk_index, "char_offset": cursor.char_offset}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _validate_read_cursor(
    cursor: _ReadCursor, chunks: list[Chunk], total_chunks: int
) -> None:
    if total_chunks == 0:
        if cursor != _ReadCursor(chunk_index=0, char_offset=0):
            raise InvalidCursorError("Invalid document cursor: document has no chunks.")
        return
    if not chunks or chunks[0].chunk_index != cursor.chunk_index:
        raise InvalidCursorError("Invalid document cursor: chunk_index is outside document.")
    if cursor.char_offset > len(chunks[0].content):
        raise InvalidCursorError("Invalid document cursor: char_offset is outside chunk.")


def _page_document_text(
    chunks: list[Chunk], cursor: _ReadCursor, *, max_chars: int, max_chunks: int
) -> _ReadPage:
    page_chunks = chunks[:max_chunks]
    parts: list[str] = []
    remaining = max_chars
    next_cursor: _ReadCursor | None = None
    end = cursor
    chunks_returned = 0

    for chunk in page_chunks:
        offset = cursor.char_offset if chunk.chunk_index == cursor.chunk_index else 0
        if offset > len(chunk.content):
            raise InvalidCursorError("Invalid document cursor: char_offset is outside chunk.")
        available = len(chunk.content) - offset
        if available <= 0:
            chunks_returned += 1
            end = _ReadCursor(chunk_index=chunk.chunk_index, char_offset=len(chunk.content))
            continue
        separator = "\n\n" if parts else ""
        if separator:
            if remaining <= len(separator):
                next_cursor = _ReadCursor(chunk_index=chunk.chunk_index, char_offset=offset)
                break
            parts.append(separator)
            remaining -= len(separator)
        take = min(available, remaining)
        parts.append(chunk.content[offset : offset + take])
        remaining -= take
        chunks_returned += 1
        end = _ReadCursor(chunk_index=chunk.chunk_index, char_offset=offset + take)
        if take < available:
            next_cursor = _ReadCursor(chunk_index=chunk.chunk_index, char_offset=offset + take)
            break

    if next_cursor is None and len(chunks) > max_chunks:
        next_chunk = chunks[max_chunks]
        next_cursor = _ReadCursor(chunk_index=next_chunk.chunk_index, char_offset=0)

    return _ReadPage(
        text="".join(parts),
        next=next_cursor,
        end=end,
        chunks_returned=chunks_returned,
    )


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<number>(?:\d+|[IVXLCDM]+)(?:\.\d+)*[.)]?)\s+"
    r"(?P<title>[A-Z][\w ,:;&/()'\"-]{2,120})$",
    flags=re.IGNORECASE,
)
_SMALL_TITLE_WORDS = {"a", "an", "and", "as", "by", "for", "in", "of", "on", "or", "the", "to"}


def _derive_headings(chunks: list[Chunk]) -> list[JsonDict]:
    headings: list[JsonDict] = []
    for chunk in chunks:
        line_offset = 0
        lines = chunk.content.splitlines(keepends=True)
        for index, line in enumerate(lines):
            raw_line = line.rstrip("\r\n")
            stripped = raw_line.strip()
            previous_blank = index == 0 or not lines[index - 1].strip()
            next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
            classified = _classify_heading(
                stripped, previous_blank=previous_blank, next_blank=next_blank
            )
            if classified is not None:
                level, text = classified
                headings.append(
                    {
                        "level": level,
                        "text": text,
                        "chunk_index": chunk.chunk_index,
                        "char_offset": line_offset + raw_line.index(stripped),
                    }
                )
            line_offset += len(line)
    return headings


def _classify_heading(
    text: str, *, previous_blank: bool, next_blank: bool
) -> tuple[int, str] | None:
    if not text or len(text) > 140:
        return None
    markdown = _MARKDOWN_HEADING_RE.match(text)
    if markdown is not None:
        return len(markdown.group(1)), markdown.group(2).strip()
    numbered = _NUMBERED_HEADING_RE.match(text)
    if numbered is not None:
        number = numbered.group("number").rstrip(".)")
        level = min(6, number.count(".") + 1)
        return level, text
    if _looks_like_title_line(text, previous_blank=previous_blank, next_blank=next_blank):
        return 2, text
    return None


def _looks_like_title_line(text: str, *, previous_blank: bool, next_blank: bool) -> bool:
    if not (previous_blank and next_blank):
        return False
    if len(text) > 80 or text.endswith((".", ",", ";", "?", "!")):
        return False
    if text.startswith(("-", "*", "+", ">")) or "://" in text:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text)
    if not words or len(words) > 10:
        return False
    if text.isupper() and len(words) <= 8:
        return True
    significant = [word for word in words if word.lower() not in _SMALL_TITLE_WORDS]
    if not significant:
        return False
    title_like = sum(1 for word in significant if word[:1].isupper())
    return title_like >= max(1, len(significant) - 1)
