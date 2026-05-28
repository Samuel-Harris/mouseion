from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount, Route

from mouseion.api.server import create_app


def test_mcp_route_is_exposed_without_double_prefix() -> None:
    app = create_app()

    assert any(isinstance(route, Route) and route.path == "/mcp" for route in app.routes)
    assert not any(isinstance(route, Mount) and route.path == "/mcp" for route in app.routes)
    assert not any(getattr(route, "path", None) == "/mcp/mcp" for route in app.routes)


def test_mcp_endpoint_runs_inside_main_lifespan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MOUSEION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MOUSEION_REPOS_DIR", str(tmp_path / "repos"))
    monkeypatch.setenv("SIMILAR_EDGE_RECOMPUTE_HOURS", "0")
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

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json=payload,
            headers={"Accept": "application/json, text/event-stream"},
        )

    assert response.status_code != 404
    assert response.status_code != 500
