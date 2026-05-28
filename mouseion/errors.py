from __future__ import annotations


class MouseionError(Exception):
    """Base exception for user-facing mouseion errors."""


class ScannedPDFError(MouseionError):
    """Raised when a PDF has too little extractable text and OCR is required."""


class DocumentNotFoundError(MouseionError):
    """Raised when a document id does not exist."""


class UnsupportedFileTypeError(MouseionError):
    """Raised when a file type cannot be ingested."""


class EmbeddingError(MouseionError):
    """Raised when the embedding backend is unavailable or returns invalid data."""
