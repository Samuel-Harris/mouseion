from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator


class DocumentType(StrEnum):
    DOCUMENT = "document"
    MEMORY = "memory"
    URL = "url"
    FILE = "file"


class Document(BaseModel):
    id: UUID
    type: DocumentType
    title: str
    source: str
    content_hash: str
    created_at: datetime
    updated_at: datetime
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: int
    document_id: UUID
    content: str
    chunk_index: int
    embedding: list[float] = Field(default_factory=list)
    token_count: int
    created_at: datetime


class RelatedDocument(BaseModel):
    document: Document
    label: str | None = None
    note: str | None = None
    created_at: datetime | None = None


class SearchFilter(BaseModel):
    tags: list[str] | None = None
    type: DocumentType | None = None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return normalize_tags(value)


class AddUrlInput(BaseModel):
    url: HttpUrl
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def normalize_tag_values(cls, value: list[str]) -> list[str]:
        return normalize_tags(value)


class AddFileInput(BaseModel):
    file_path: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def normalize_file_tags(cls, value: list[str]) -> list[str]:
        return normalize_tags(value)


class AddMemoryInput(BaseModel):
    content: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def normalize_memory_tags(cls, value: list[str]) -> list[str]:
        return normalize_tags(value)


class AddRepoInput(BaseModel):
    repo_url: str
    name: str | None = None


class SearchInput(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    include_graph_neighbours: bool = False
    filter: SearchFilter | None = None


class GetDocumentInput(BaseModel):
    document_id: UUID


class ListInput(BaseModel):
    type: Literal["document", "memory", "url", "all"] = "all"
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class RelateInput(BaseModel):
    from_id: UUID
    to_id: UUID
    label: str | None = None
    note: str | None = None


class DeleteInput(BaseModel):
    id: UUID


class IngestedContent(BaseModel):
    title: str
    source: str
    type: DocumentType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkText(BaseModel):
    content: str
    token_count: int


def normalize_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = tag.strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def utc_now() -> datetime:
    return datetime.now(tz=UTC)
