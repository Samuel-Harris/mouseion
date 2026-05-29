from __future__ import annotations

from pathlib import Path

import fitz
import httpx
import pytest

import mouseion.ingest.ingestor as ingestor_module
from mouseion.config import Settings
from mouseion.domain.models import DocumentType
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


async def test_fetch_url_ingests_direct_pdf_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ingestor = Ingestor(
        Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    )
    url = "https://export.arxiv.org/pdf/2605.26115"
    pdf = _pdf_bytes("Direct PDF URL support " * 80)

    class FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def get(self, requested_url: str) -> httpx.Response:
            return httpx.Response(
                200,
                content=pdf,
                headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", requested_url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    content = await ingestor.fetch_url(url)

    assert content.type == DocumentType.URL
    assert content.source == url
    assert content.title == "2605.26115"
    assert "Direct PDF URL support" in content.content
    assert content.metadata["mime_type"] == "application/pdf"
    assert Path(content.metadata["stored_path"]).is_file()


async def test_fetch_url_retries_retryable_status_with_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ingestor = Ingestor(
        Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    )
    url = "https://example.test/paper.pdf"
    sleeps: list[int] = []
    calls = _patch_async_client(
        monkeypatch,
        [
            _response(503, url=url),
            _response(
                200,
                url=url,
                content=_pdf_bytes("Retried PDF URL support " * 80),
                headers={"content-type": "application/pdf"},
            ),
        ],
    )

    async def fake_sleep(attempt: int) -> None:
        sleeps.append(attempt)

    monkeypatch.setattr(ingestor_module, "_sleep_before_url_retry", fake_sleep)

    content = await ingestor.fetch_url(url)

    assert calls == [url, url]
    assert sleeps == [0]
    assert content.title == "paper"
    assert "Retried PDF URL support" in content.content


async def test_fetch_url_retries_transport_error_with_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ingestor = Ingestor(
        Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    )
    url = "https://example.test/paper.pdf"
    sleeps: list[int] = []
    request = httpx.Request("GET", url)
    calls = _patch_async_client(
        monkeypatch,
        [
            httpx.ConnectError("temporary connection failure", request=request),
            _response(
                200,
                url=url,
                content=_pdf_bytes("Transport retry PDF support " * 80),
                headers={"content-type": "application/pdf"},
            ),
        ],
    )

    async def fake_sleep(attempt: int) -> None:
        sleeps.append(attempt)

    monkeypatch.setattr(ingestor_module, "_sleep_before_url_retry", fake_sleep)

    content = await ingestor.fetch_url(url)

    assert calls == [url, url]
    assert sleeps == [0]
    assert "Transport retry PDF support" in content.content


async def test_fetch_url_does_not_retry_non_retryable_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ingestor = Ingestor(
        Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    )
    url = "https://example.test/missing"
    calls = _patch_async_client(monkeypatch, [_response(404, url=url)])

    async def unexpected_sleep(attempt: int) -> None:
        raise AssertionError(f"unexpected retry sleep for attempt {attempt}")

    monkeypatch.setattr(ingestor_module, "_sleep_before_url_retry", unexpected_sleep)

    with pytest.raises(httpx.HTTPStatusError):
        await ingestor.fetch_url(url)

    assert calls == [url]


def _pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    lines = [text[index : index + 80] for index in range(0, len(text), 80)]
    for index, line in enumerate(lines[:20]):
        page.insert_text((72, 72 + index * 24), line)
    return bytes(document.tobytes())


def _response(
    status_code: int,
    *,
    url: str,
    content: bytes = b"",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=content,
        headers=headers or {},
        request=httpx.Request("GET", url),
    )


def _patch_async_client(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[httpx.Response | httpx.RequestError]
) -> list[str]:
    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def get(self, requested_url: str) -> httpx.Response:
            calls.append(requested_url)
            outcome = outcomes.pop(0)
            if isinstance(outcome, httpx.RequestError):
                raise outcome
            return outcome

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return calls
