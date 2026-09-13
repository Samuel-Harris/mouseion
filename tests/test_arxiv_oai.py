from __future__ import annotations

import httpx
import pytest
from arxiv_oai_fixtures import list_records_page, list_sets_page, record

import utilities.arxiv.oai as oai_module
from utilities.arxiv.oai import (
    OaiError,
    RequestPacer,
    _get_with_retries,
    fetch_oai_sets,
    iter_oai_records,
    oai_record_to_kaggle_dict,
    parse_oai_page,
)

OAI_URL = "https://oaipmh.arxiv.org/oai"


class _StaticClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    async def get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        self.calls.append(dict(params or {}))
        return self._responses.pop(0)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _response(
    content: bytes = b"", *, status_code: int = 200, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=content,
        headers=headers or {},
        request=httpx.Request("GET", OAI_URL),
    )


def _patch_no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(oai_module, "_sleep", fake_sleep)
    return sleeps


def test_parse_oai_page_maps_arxivraw_to_kaggle_shape() -> None:
    xml = list_records_page(
        [
            record(
                "1601.04794",
                datestamp="2026-09-01",
                set_specs=("cs:cs:CC",),
                fields={
                    "title": "Concentration Inequalities",
                    "abstract": "A new framework for phase transitions.",
                    "authors": "Changqing Liu and Bo Zhang",
                    "submitter": "Changqing Liu",
                    "categories": "cs.CC",
                    "comments": "57 pages",
                },
                versions=(
                    ("v1", "Tue, 19 Jan 2016 04:10:52 GMT"),
                    ("v2", "Tue, 25 Aug 2026 16:38:03 GMT"),
                ),
            )
        ]
    )

    page = parse_oai_page(xml)

    assert page.max_datestamp == "2026-09-01"
    assert page.resumption_token is None
    assert page.deleted_ids == []
    assert len(page.records) == 1
    payload = oai_record_to_kaggle_dict(page.records[0])
    assert payload["id"] == "1601.04794"
    assert payload["title"] == "Concentration Inequalities"
    assert payload["abstract"] == "A new framework for phase transitions."
    assert payload["authors"] == "Changqing Liu and Bo Zhang"
    assert payload["submitter"] == "Changqing Liu"
    assert payload["categories"] == "cs.CC"
    assert payload["comments"] == "57 pages"
    assert payload["versions"] == [
        {"version": "v1", "created": "Tue, 19 Jan 2016 04:10:52 GMT"},
        {"version": "v2", "created": "Tue, 25 Aug 2026 16:38:03 GMT"},
    ]
    assert payload["update_date"] == "2026-09-01"
    assert "journal-ref" not in payload


def test_parse_oai_page_collects_deleted_ids_and_latest_datestamp() -> None:
    xml = list_records_page(
        [
            record("1111.0001", datestamp="2026-09-01"),
            record("2222.0002", datestamp="2026-09-03", deleted=True),
        ]
    )

    page = parse_oai_page(xml)

    assert [parsed.arxiv_id for parsed in page.records] == ["1111.0001"]
    assert page.deleted_ids == ["2222.0002"]
    assert page.max_datestamp == "2026-09-03"


def test_parse_oai_page_treats_no_records_match_as_empty() -> None:
    xml = (
        b'<?xml version="1.0"?><OAI-PMH><error code="noRecordsMatch">'
        b"No matches for the requested query.</error></OAI-PMH>"
    )

    page = parse_oai_page(xml)

    assert page.records == []
    assert page.deleted_ids == []
    assert page.resumption_token is None
    assert page.max_datestamp is None


def test_parse_oai_page_raises_on_other_oai_errors() -> None:
    xml = b'<?xml version="1.0"?><OAI-PMH><error code="badVerb">Bad verb</error></OAI-PMH>'

    with pytest.raises(OaiError, match="badVerb"):
        parse_oai_page(xml)


def test_parse_oai_page_rejects_doctype_payloads() -> None:
    xml = b'<?xml version="1.0"?><!DOCTYPE OAI-PMH [<!ENTITY x "y">]><OAI-PMH/>'

    with pytest.raises(OaiError, match="DOCTYPE"):
        parse_oai_page(xml)


async def test_iter_oai_records_follows_resumption_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_no_sleep(monkeypatch)
    client = _StaticClient(
        [
            _response(
                list_records_page([record("1111.0001")], token="verb%3DListRecords%26skip%3D1")
            ),
            _response(list_records_page([record("2222.0002")])),
        ]
    )

    pages = [page async for page in iter_oai_records(client, set_spec="cs", from_date="2026-09-01")]

    assert len(pages) == 2
    assert client.calls[0] == {
        "verb": "ListRecords",
        "metadataPrefix": "arXivRaw",
        "set": "cs",
        "from": "2026-09-01",
    }
    assert client.calls[1] == {
        "verb": "ListRecords",
        "resumptionToken": "verb%3DListRecords%26skip%3D1",
    }


async def test_fetch_oai_sets_parses_set_specs_and_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_no_sleep(monkeypatch)
    client = _StaticClient(
        [
            _response(
                list_sets_page([("cs", "Computer Science"), ("physics:astro-ph", "Astrophysics")])
            )
        ]
    )

    sets = await fetch_oai_sets(client)

    assert [(item.spec, item.name) for item in sets] == [
        ("cs", "Computer Science"),
        ("physics:astro-ph", "Astrophysics"),
    ]


async def test_get_with_retries_backs_off_on_retryable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _patch_no_sleep(monkeypatch)
    client = _StaticClient(
        [
            _response(b"", status_code=503, headers={"retry-after": "not-a-number"}),
            _response(list_records_page([record("1111.0001")])),
        ]
    )

    response = await _get_with_retries(client, base_url=OAI_URL, params={"verb": "ListRecords"})

    assert response.status_code == 200
    assert sleeps == [0.5]
    assert len(client.calls) == 2


async def test_get_with_retries_honours_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _patch_no_sleep(monkeypatch)
    client = _StaticClient(
        [
            _response(b"", status_code=503, headers={"retry-after": "7"}),
            _response(list_records_page([record("1111.0001")])),
        ]
    )

    await _get_with_retries(client, base_url=OAI_URL, params={"verb": "ListRecords"})

    assert sleeps == [7.0]


async def test_request_pacer_enforces_three_second_spacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr(oai_module, "_monotonic", clock.monotonic)
    monkeypatch.setattr(oai_module, "_sleep", fake_sleep)

    pacer = RequestPacer()
    await pacer.wait()
    await pacer.wait()
    await pacer.wait()

    assert sleeps == [3.0, 3.0]
