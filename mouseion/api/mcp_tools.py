from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from mouseion.domain.models import (
    AddFileInput,
    AddMemoryInput,
    AddRepoInput,
    AddUrlInput,
    DeleteInput,
    GetDocumentInput,
    ListInput,
    RelateInput,
    SearchFilter,
    SearchInput,
)
from mouseion.services.service import MouseionService


def build_mcp(
    service_ref: dict[str, MouseionService], description: str | None = None
) -> FastMCP:
    mcp = FastMCP(
        "Mouseion",
        instructions=description,
        stateless_http=True,
        json_response=True,
    )

    def service() -> MouseionService:
        return service_ref["service"]

    @mcp.tool()
    async def mouseion_add_url(url: str, tags: list[str] | None = None) -> dict[str, Any]:
        return await service().add_url(AddUrlInput.model_validate({"url": url, "tags": tags or []}))

    @mcp.tool()
    async def mouseion_add_file(file_path: str, tags: list[str] | None = None) -> dict[str, Any]:
        return await service().add_file(AddFileInput(file_path=file_path, tags=tags or []))

    @mcp.tool()
    async def mouseion_add_memory(content: str, tags: list[str] | None = None) -> dict[str, Any]:
        return await service().add_memory(AddMemoryInput(content=content, tags=tags or []))

    @mcp.tool()
    async def mouseion_add_repo(repo_url: str, name: str | None = None) -> dict[str, Any]:
        return await service().add_repo(AddRepoInput(repo_url=repo_url, name=name))

    @mcp.tool()
    async def mouseion_search(
        query: str,
        top_k: int = 10,
        include_graph_neighbours: bool = False,
        filter: dict[str, Any] | None = None,
        search_syntax: str = "plain",
    ) -> dict[str, Any]:
        parsed_filter = SearchFilter.model_validate(filter) if filter else None
        parsed_syntax: Literal["plain", "advanced"] = (
            "advanced" if search_syntax == "advanced" else "plain"
        )
        return await service().search(
            SearchInput(
                query=query,
                top_k=top_k,
                include_graph_neighbours=include_graph_neighbours,
                filter=parsed_filter,
                search_syntax=parsed_syntax,
            )
        )

    @mcp.tool()
    async def mouseion_get_document(document_id: str) -> dict[str, Any]:
        return await service().get_document(GetDocumentInput(document_id=UUID(document_id)))

    @mcp.tool()
    async def mouseion_list(type: str = "all", limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return await service().list_documents(
            ListInput.model_validate({"type": type, "limit": limit, "offset": offset})
        )

    @mcp.tool()
    async def mouseion_relate(
        from_id: str, to_id: str, label: str | None = None, note: str | None = None
    ) -> dict[str, Any]:
        return await service().relate(
            RelateInput(from_id=UUID(from_id), to_id=UUID(to_id), label=label, note=note)
        )

    @mcp.tool()
    async def mouseion_delete(id: str) -> dict[str, Any]:
        return await service().delete(DeleteInput(id=UUID(id)))

    @mcp.tool()
    async def mouseion_recompute_edges() -> dict[str, Any]:
        return await service().recompute_edges()

    @mcp.tool()
    async def mouseion_export() -> dict[str, Any]:
        return await service().export()

    return mcp
