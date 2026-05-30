from __future__ import annotations

from functools import cached_property
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.stdlib import BoundLogger


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mouseion_host: str = Field(default="127.0.0.1", alias="MOUSEION_HOST")
    mouseion_port: int = Field(default=7778, alias="MOUSEION_PORT")
    mouseion_data_dir: Path = Field(default=Path("./data"), alias="MOUSEION_DATA_DIR")
    mouseion_repos_dir: Path = Field(default=Path("./repos"), alias="MOUSEION_REPOS_DIR")
    mouseion_mcp_description: str | None = Field(default=None, alias="MOUSEION_MCP_DESCRIPTION")
    ollama_host: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_HOST")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    chunk_target_tokens: int = Field(default=512, alias="CHUNK_TARGET_TOKENS")
    chunk_max_tokens: int = Field(default=1024, alias="CHUNK_MAX_TOKENS")
    chunk_min_tokens: int = Field(default=100, alias="CHUNK_MIN_TOKENS")
    similarity_threshold: float = Field(default=0.82, alias="SIMILARITY_THRESHOLD")
    similarity_top_k: int = Field(default=10, alias="SIMILARITY_TOP_K")
    similar_edge_recompute_hours: float = Field(default=24, alias="SIMILAR_EDGE_RECOMPUTE_HOURS")
    rrf_k: int = Field(default=60, alias="RRF_K")
    vector_search_mode: Literal["exact", "quantized"] = Field(
        default="exact", alias="MOUSEION_VECTOR_SEARCH_MODE"
    )
    vector_quantization_qbits: Literal[2, 3, 4] = Field(
        default=4, alias="MOUSEION_VECTOR_QUANTIZATION_QBITS"
    )
    vector_quantize_preload: bool = Field(
        default=False, alias="MOUSEION_VECTOR_QUANTIZE_PRELOAD"
    )
    vector_quantize_max_memory: str = Field(
        default="30MB", alias="MOUSEION_VECTOR_QUANTIZE_MAX_MEMORY"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("mouseion_data_dir", "mouseion_repos_dir", mode="before")
    @classmethod
    def expand_path(cls, value: Any) -> Path:
        return Path(value).expanduser()

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("vector_search_mode", mode="before")
    @classmethod
    def normalize_vector_search_mode(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("vector_quantize_max_memory", mode="before")
    @classmethod
    def normalize_vector_quantize_max_memory(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip().upper()
        if not stripped:
            raise ValueError("MOUSEION_VECTOR_QUANTIZE_MAX_MEMORY must not be empty")
        return stripped

    @field_validator("mouseion_mcp_description")
    @classmethod
    def normalize_mcp_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_ranges(self) -> Settings:
        if self.mouseion_port < 1 or self.mouseion_port > 65535:
            raise ValueError("MOUSEION_PORT must be between 1 and 65535")
        if self.chunk_min_tokens < 1:
            raise ValueError("CHUNK_MIN_TOKENS must be positive")
        if self.chunk_target_tokens < self.chunk_min_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS must be >= CHUNK_MIN_TOKENS")
        if self.chunk_max_tokens < self.chunk_target_tokens:
            raise ValueError("CHUNK_MAX_TOKENS must be >= CHUNK_TARGET_TOKENS")
        if not (0 < self.similarity_threshold <= 1):
            raise ValueError("SIMILARITY_THRESHOLD must be in (0, 1]")
        if self.similarity_top_k < 1:
            raise ValueError("SIMILARITY_TOP_K must be positive")
        if self.rrf_k < 1:
            raise ValueError("RRF_K must be positive")
        return self

    @property
    def sqlite_path(self) -> Path:
        return self.mouseion_data_dir / "mouseion.db"

    @property
    def files_dir(self) -> Path:
        return self.mouseion_data_dir / "files"

    @property
    def export_dir(self) -> Path:
        return self.mouseion_data_dir / "export"

    @cached_property
    def absolute_data_dir(self) -> Path:
        return self.mouseion_data_dir.resolve()

    def is_loopback_host(self) -> bool:
        if self.mouseion_host in {"localhost", "::1"}:
            return True
        try:
            return ip_address(self.mouseion_host).is_loopback
        except ValueError:
            return False

    def warn_if_non_loopback(self, logger: BoundLogger) -> None:
        if not self.is_loopback_host():
            logger.warning(
                "mouseion_host_not_loopback",
                mouseion_host=self.mouseion_host,
                message="No authentication is enabled; non-loopback binding exposes the daemon.",
            )

    def ensure_directories(self) -> None:
        self.mouseion_data_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.mouseion_repos_dir.mkdir(parents=True, exist_ok=True)
