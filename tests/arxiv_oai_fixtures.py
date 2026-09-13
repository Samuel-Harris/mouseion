"""OAI-PMH XML builders shared by the arXiv importer tests."""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape

OAI_PMH_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"
ARXIV_RAW_NAMESPACE = "http://arxiv.org/OAI/arXivRaw/"


def oai_page(body: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<OAI-PMH xmlns="{OAI_PMH_NAMESPACE}">\n{body}\n</OAI-PMH>\n'
    ).encode()


def list_sets_page(sets: list[tuple[str, str]], *, token: str | None = None) -> bytes:
    elements = "".join(
        f"<set><setSpec>{escape(spec)}</setSpec><setName>{escape(name)}</setName></set>"
        for spec, name in sets
    )
    return oai_page(f"<ListSets>{elements}{_token(token)}</ListSets>")


def list_records_page(records: list[dict[str, Any]], *, token: str | None = None) -> bytes:
    elements = "".join(_record(record) for record in records)
    return oai_page(f"<ListRecords>{elements}{_token(token)}</ListRecords>")


def record(
    arxiv_id: str,
    *,
    datestamp: str = "2026-09-01",
    set_specs: tuple[str, ...] = ("cs:cs:AI",),
    fields: dict[str, str] | None = None,
    versions: tuple[tuple[str, str], ...] = (("v1", "Mon, 1 Jan 2024 00:00:00 GMT"),),
    deleted: bool = False,
) -> dict[str, Any]:
    return {
        "id": arxiv_id,
        "datestamp": datestamp,
        "set_specs": set_specs,
        "fields": fields
        or {
            "title": f"Title for {arxiv_id}",
            "abstract": f"Abstract for {arxiv_id}.",
            "authors": "Ada Lovelace",
            "categories": "cs.AI",
            "comments": "12 pages",
        },
        "versions": [{"version": version, "created": created} for version, created in versions],
        "deleted": deleted,
    }


def _record(data: dict[str, Any]) -> str:
    status = ' status="deleted"' if data.get("deleted") else ""
    set_specs = "".join(
        f"<setSpec>{escape(str(spec))}</setSpec>" for spec in data.get("set_specs", ())
    )
    identifier = escape("oai:arXiv.org:" + str(data["id"]))
    datestamp = escape(str(data.get("datestamp", "")))
    header = (
        f"<header{status}>"
        f"<identifier>{identifier}</identifier>"
        f"<datestamp>{datestamp}</datestamp>"
        f"{set_specs}</header>"
    )
    if data.get("deleted"):
        return f"<record>{header}</record>"

    versions = "".join(
        f'<version version="{escape(str(version["version"]))}">'
        f"<date>{escape(str(version['created']))}</date></version>"
        for version in data.get("versions", ())
    )
    fields = "".join(
        f"<{name}>{escape(str(value))}</{name}>" for name, value in data.get("fields", {}).items()
    )
    metadata = (
        f'<arXivRaw xmlns="{ARXIV_RAW_NAMESPACE}">'
        f"<id>{escape(str(data['id']))}</id>{versions}{fields}</arXivRaw>"
    )
    return f"<record>{header}<metadata>{metadata}</metadata></record>"


def _token(token: str | None) -> str:
    return "" if token is None else f"<resumptionToken>{escape(token)}</resumptionToken>"
