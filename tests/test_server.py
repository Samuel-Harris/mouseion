from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount, Route

from mouseion.api.mcp_tools import build_mcp
from mouseion.api.server import create_app
from mouseion.domain.models import (
    DocumentOutlineInput,
    ReadDocumentInput,
    SearchDocumentInput,
)


def test_build_mcp_uses_custom_description() -> None:
    mcp = build_mcp({}, description="Custom corpus description")

    assert mcp.instructions == "Custom corpus description"


def test_mcp_tool_registry_excludes_export() -> None:
    tools = asyncio.run(build_mcp({}).list_tools())
    tool_names = {tool.name for tool in tools}

    assert "mouseion_export" not in tool_names
    assert {
        "mouseion_add_url",
        "mouseion_add_file",
        "mouseion_add_memory",
        "mouseion_add_repo",
        "mouseion_search",
        "mouseion_read_document",
        "mouseion_document_outline",
        "mouseion_search_document",
        "mouseion_list",
        "mouseion_delete",
    } <= tool_names


def test_mcp_route_is_exposed_without_double_prefix() -> None:
    app = create_app()

    assert any(isinstance(route, Route) and route.path == "/mcp" for route in app.routes)
    assert not any(isinstance(route, Mount) and route.path == "/mcp" for route in app.routes)
    assert not any(getattr(route, "path", None) == "/mcp/mcp" for route in app.routes)


def test_vector_api_routes_are_exposed() -> None:
    app = create_app()
    paths = {getattr(route, "path", None) for route in app.routes}

    assert "/api/vector/status" in paths
    assert "/api/vector/mode" in paths
    assert "/api/vector/quantize" in paths
    assert "/api/vector/cleanup" in paths


def test_document_read_outline_and_search_routes() -> None:
    app = create_app()
    app.state.service = FakeDocumentService()
    client = TestClient(app)
    document_id = "00000000-0000-0000-0000-000000000123"

    read = client.get(
        f"/api/documents/{document_id}/read",
        params={
            "cursor": "abc",
            "max_chars": 123,
            "max_chunks": 4,
            "include_metadata": "false",
        },
    )
    outline = client.get(f"/api/documents/{document_id}/outline")
    search = client.post(
        f"/api/documents/{document_id}/search",
        json={"query": "needle", "top_k": 3},
    )

    assert read.status_code == 200
    assert read.json() == {
        "document_id": document_id,
        "cursor": "abc",
        "max_chars": 123,
        "max_chunks": 4,
        "include_metadata": False,
    }
    assert outline.status_code == 200
    assert outline.json() == {"document_id": document_id, "headings": []}
    assert search.status_code == 200
    assert search.json() == {
        "document_id": document_id,
        "query": "needle",
        "top_k": 3,
        "results": [],
    }


def test_mcp_endpoint_runs_inside_main_lifespan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MOUSEION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MOUSEION_REPOS_DIR", str(tmp_path / "repos"))
    monkeypatch.setenv("MOUSEION_MCP_DESCRIPTION", "Test corpus description")
    app = create_app()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0"},
        },
    }

    with TestClient(app, base_url="http://127.0.0.1:7778") as client:
        response = client.post(
            "/mcp",
            json=payload,
            headers={"Accept": "application/json, text/event-stream"},
        )

    assert response.status_code != 404
    assert response.status_code != 500
    assert response.json()["result"]["instructions"] == "Test corpus description"


class FakeDocumentService:
    async def read_document(self, input: ReadDocumentInput) -> dict[str, object]:
        return {
            "document_id": str(input.document_id),
            "cursor": input.cursor,
            "max_chars": input.max_chars,
            "max_chunks": input.max_chunks,
            "include_metadata": input.include_metadata,
        }

    async def document_outline(self, input: DocumentOutlineInput) -> dict[str, object]:
        return {"document_id": str(input.document_id), "headings": []}

    async def search_document(self, input: SearchDocumentInput) -> dict[str, object]:
        return {
            "document_id": str(input.document_id),
            "query": input.query,
            "top_k": input.top_k,
            "results": [],
        }
