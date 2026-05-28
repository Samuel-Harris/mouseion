from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from mouseion.config import Settings
from mouseion.errors import MouseionError, ScannedPDFError
from mouseion.ingest.ingestor import Ingestor


def test_memory_content_validation(tmp_path: Path) -> None:
    ingestor = Ingestor(
        Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    )
    with pytest.raises(MouseionError):
        ingestor.memory("   ")


def test_pdf_scanned_error_threshold(tmp_path: Path) -> None:
    ingestor = Ingestor(
        Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    )
    pdf = tmp_path / "blank.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf)
    document.close()

    with pytest.raises(ScannedPDFError):
        ingestor._read_pdf(pdf)
