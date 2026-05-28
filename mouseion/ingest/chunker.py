from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from mouseion.config import Settings
from mouseion.domain.models import ChunkText


@dataclass(slots=True)
class Chunker:
    settings: Settings
    _encoding: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def chunk(self, text: str) -> list[ChunkText]:
        sections = self._split_structure(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue
            section_tokens = self.count_tokens(section)
            if section_tokens > self.settings.chunk_max_tokens:
                self._flush(current, chunks)
                current = []
                current_tokens = 0
                chunks.extend(self._force_split(section))
                continue
            would_exceed = current_tokens + section_tokens > self.settings.chunk_target_tokens
            if current and would_exceed:
                self._flush(current, chunks)
                current = [section]
                current_tokens = section_tokens
            else:
                current.append(section)
                current_tokens += section_tokens

        self._flush(current, chunks)
        merged = self._merge_small_chunks(chunks)
        return [ChunkText(content=chunk, token_count=self.count_tokens(chunk)) for chunk in merged]

    def _split_structure(self, text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n")
        blocks = re.split(r"\n(?=#{1,6}\s)|\n{2,}", normalized)
        expanded: list[str] = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if self.count_tokens(block) <= self.settings.chunk_target_tokens:
                expanded.append(block)
                continue
            expanded.extend(self._split_sentences(block))
        return expanded

    def _split_sentences(self, text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?])\s+", text)
        return [piece.strip() for piece in pieces if piece.strip()]

    def _force_split(self, text: str) -> list[str]:
        token_ids = self._encoding.encode(text)
        chunks: list[str] = []
        max_tokens = self.settings.chunk_max_tokens
        for start in range(0, len(token_ids), max_tokens):
            piece = self._encoding.decode(token_ids[start : start + max_tokens]).strip()
            if piece:
                chunks.append(piece)
        return chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []
        merged: list[str] = []
        pending = chunks[0]
        for chunk in chunks[1:]:
            pending_tokens = self.count_tokens(pending)
            chunk_tokens = self.count_tokens(chunk)
            if (
                pending_tokens < self.settings.chunk_min_tokens
                and pending_tokens + chunk_tokens <= self.settings.chunk_max_tokens
            ):
                pending = f"{pending}\n\n{chunk}"
            else:
                merged.append(pending)
                pending = chunk
        merged.append(pending)
        return merged

    @staticmethod
    def _flush(current: list[str], chunks: list[str]) -> None:
        if current:
            chunks.append("\n\n".join(current).strip())
