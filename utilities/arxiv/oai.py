"""OAI-PMH transport and parsing for the arXiv metadata harvester."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

import httpx

from mouseion.errors import MouseionError

OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
OAI_METADATA_PREFIX = "arXivRaw"
OAI_REQUEST_INTERVAL_SECONDS = 3.0
OAI_MAX_ATTEMPTS = 3
OAI_BACKOFF_SECONDS = 0.5
OAI_RETRY_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
OAI_IDENTIFIER_PREFIX = "oai:arXiv.org:"

ARXIV_RAW_TEXT_FIELDS = (
    "title",
    "abstract",
    "authors",
    "submitter",
    "categories",
    "comments",
    "journal-ref",
    "doi",
    "report-no",
    "license",
)


class OaiError(MouseionError):
    """Raised when arXiv's OAI-PMH endpoint fails or returns an unusable response."""


@dataclass(frozen=True, slots=True)
class OaiRecord:
    arxiv_id: str
    datestamp: str
    fields: dict[str, str]
    versions: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class OaiPage:
    records: list[OaiRecord]
    deleted_ids: list[str]
    resumption_token: str | None
    max_datestamp: str | None


@dataclass(frozen=True, slots=True)
class OaiSet:
    spec: str
    name: str


@dataclass(frozen=True, slots=True)
class OaiSetPage:
    sets: list[OaiSet]
    resumption_token: str | None


def oai_record_to_kaggle_dict(record: OaiRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": record.arxiv_id,
        "versions": [dict(version) for version in record.versions],
        "update_date": record.datestamp,
    }
    payload.update(record.fields)
    return payload


def parse_oai_page(xml_bytes: bytes) -> OaiPage:
    root = _parse_root(xml_bytes)
    error = _child(root, "error")
    if error is not None:
        code = error.get("code", "")
        if code == "noRecordsMatch":
            return OaiPage(records=[], deleted_ids=[], resumption_token=None, max_datestamp=None)
        raise OaiError(_error_message(error))

    list_records = _child(root, "ListRecords")
    if list_records is None:
        raise OaiError("OAI response did not contain a ListRecords element")

    records: list[OaiRecord] = []
    deleted_ids: list[str] = []
    max_datestamp: str | None = None
    for record_element in _children(list_records, "record"):
        header = _child(record_element, "header")
        datestamp = _text(_child(header, "datestamp")) if header is not None else ""
        max_datestamp = later_datestamp(max_datestamp, datestamp)
        if header is not None and header.get("status", "") == "deleted":
            arxiv_id = _arxiv_id_from_identifier(_text(_child(header, "identifier")))
            if arxiv_id:
                deleted_ids.append(arxiv_id)
            continue
        parsed = _parse_record(record_element, datestamp)
        if parsed is not None:
            records.append(parsed)

    token = _text(_child(list_records, "resumptionToken")) or None
    return OaiPage(
        records=records,
        deleted_ids=deleted_ids,
        resumption_token=token,
        max_datestamp=max_datestamp,
    )


def parse_oai_sets_page(xml_bytes: bytes) -> OaiSetPage:
    root = _parse_root(xml_bytes)
    error = _child(root, "error")
    if error is not None:
        raise OaiError(_error_message(error))

    list_sets = _child(root, "ListSets")
    if list_sets is None:
        raise OaiError("OAI response did not contain a ListSets element")

    sets = [
        OaiSet(
            spec=_text(_child(set_element, "setSpec")),
            name=_text(_child(set_element, "setName")),
        )
        for set_element in _children(list_sets, "set")
    ]
    token = _text(_child(list_sets, "resumptionToken")) or None
    return OaiSetPage(sets=sets, resumption_token=token)


async def iter_oai_records(
    client: httpx.AsyncClient,
    *,
    set_spec: str,
    from_date: str,
    until_date: str | None = None,
    base_url: str = OAI_BASE_URL,
) -> AsyncIterator[OaiPage]:
    pacer = RequestPacer()
    params: dict[str, str] = {
        "verb": "ListRecords",
        "metadataPrefix": OAI_METADATA_PREFIX,
        "set": set_spec,
        "from": from_date,
    }
    if until_date is not None:
        params["until"] = until_date
    while True:
        await pacer.wait()
        response = await _get_with_retries(client, base_url=base_url, params=params)
        page = parse_oai_page(response.content)
        yield page
        if page.resumption_token is None:
            return
        params = {"verb": "ListRecords", "resumptionToken": page.resumption_token}


async def fetch_oai_sets(
    client: httpx.AsyncClient, *, base_url: str = OAI_BASE_URL
) -> list[OaiSet]:
    pacer = RequestPacer()
    params: dict[str, str] = {"verb": "ListSets"}
    sets: list[OaiSet] = []
    while True:
        await pacer.wait()
        response = await _get_with_retries(client, base_url=base_url, params=params)
        page = parse_oai_sets_page(response.content)
        sets.extend(page.sets)
        if page.resumption_token is None:
            return sets
        params = {"verb": "ListSets", "resumptionToken": page.resumption_token}


class RequestPacer:
    def __init__(self, interval: float = OAI_REQUEST_INTERVAL_SECONDS) -> None:
        self._interval = interval
        self._next_allowed_at = 0.0

    async def wait(self) -> None:
        now = _monotonic()
        if now < self._next_allowed_at:
            await _sleep(self._next_allowed_at - now)
            now = _monotonic()
        self._next_allowed_at = now + self._interval


async def _get_with_retries(
    client: httpx.AsyncClient, *, base_url: str, params: dict[str, str]
) -> httpx.Response:
    for attempt in range(OAI_MAX_ATTEMPTS):
        try:
            response = await client.get(base_url, params=params)
            if _is_retryable_status_code(response.status_code) and not _is_final_attempt(attempt):
                await _sleep_before_retry(response, attempt)
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if not _is_retryable_status_code(exc.response.status_code) or _is_final_attempt(
                attempt
            ):
                raise
            await _sleep_before_retry(exc.response, attempt)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError):
            if _is_final_attempt(attempt):
                raise
            await _sleep_before_retry(None, attempt)

    raise OaiError("OAI request retry loop exited unexpectedly")


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in OAI_RETRY_STATUS_CODES


def _is_final_attempt(attempt: int) -> bool:
    return attempt >= OAI_MAX_ATTEMPTS - 1


async def _sleep_before_retry(response: httpx.Response | None, attempt: int) -> None:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                delay = max(0.0, float(retry_after))
            except ValueError:
                delay = None
            if delay is not None:
                await _sleep(delay)
                return
    await _sleep(OAI_BACKOFF_SECONDS * (2**attempt))


def _monotonic() -> float:
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _parse_root(xml_bytes: bytes) -> ElementTree.Element:
    if b"<!doctype" in xml_bytes.lower():
        raise OaiError("OAI response contained a DOCTYPE declaration and was rejected")
    try:
        return ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        raise OaiError(f"Could not parse OAI XML response: {exc}") from exc


def _parse_record(record_element: ElementTree.Element, datestamp: str) -> OaiRecord | None:
    metadata = _child(record_element, "metadata")
    arxiv_raw = _child(metadata, "arXivRaw") if metadata is not None else None
    if arxiv_raw is None:
        return None

    fields: dict[str, str] = {}
    for name in ARXIV_RAW_TEXT_FIELDS:
        element = _child(arxiv_raw, name)
        if element is not None:
            fields[name] = _text(element)

    versions = tuple(
        {
            "version": version_element.get("version", ""),
            "created": _text(_child(version_element, "date")),
        }
        for version_element in _children(arxiv_raw, "version")
    )
    arxiv_id = _strip_all_whitespace(_text(_child(arxiv_raw, "id")))
    if not arxiv_id:
        return None
    return OaiRecord(arxiv_id=arxiv_id, datestamp=datestamp, fields=fields, versions=versions)


def _error_message(error: ElementTree.Element) -> str:
    code = error.get("code", "") or "unknown"
    message = _text(error)
    return f"OAI error {code}: {message}" if message else f"OAI error {code}"


def _arxiv_id_from_identifier(identifier: str) -> str:
    value = identifier.strip()
    if value.startswith(OAI_IDENTIFIER_PREFIX):
        return _strip_all_whitespace(value[len(OAI_IDENTIFIER_PREFIX) :])
    if ":" in value:
        return _strip_all_whitespace(value.rsplit(":", 1)[-1])
    return _strip_all_whitespace(value)


def _strip_all_whitespace(value: str) -> str:
    return "".join(value.split())


def later_datestamp(current: str | None, candidate: str) -> str | None:
    if not candidate:
        return current
    if current is None or candidate > current:
        return candidate
    return current


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child
    return None


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _text(element: ElementTree.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()
