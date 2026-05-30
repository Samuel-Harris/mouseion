from __future__ import annotations

import contextlib
import shutil
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import uuid4

import structlog.contextvars
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mouseion.api.mcp_tools import build_mcp
from mouseion.api.tasks import BackgroundTasks
from mouseion.config import Settings
from mouseion.domain.models import (
    AddFileInput,
    AddMemoryInput,
    AddRepoInput,
    AddUrlInput,
    DeleteInput,
    GetDocumentInput,
    ListInput,
    RelateInput,
    SearchInput,
)
from mouseion.errors import MouseionError
from mouseion.services.factory import open_services
from mouseion.services.service import MouseionService
from mouseion.support.logging_config import configure_logging, get_logger

JsonDict = dict[str, Any]


class VectorModeInput(BaseModel):
    mode: str = Field(pattern="^(exact|quantized)$")
    qbits: int | None = Field(default=None, ge=2, le=4)


class VectorQuantizeInput(BaseModel):
    qbits: int = Field(ge=2, le=4)
    preload: bool = False


def create_app() -> FastAPI:
    settings = Settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    service_ref: dict[str, MouseionService] = {}
    tasks: BackgroundTasks | None = None
    mcp_app = build_mcp(
        service_ref, description=settings.mouseion_mcp_description
    ).streamable_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        nonlocal tasks
        async with mcp_app.router.lifespan_context(mcp_app):
            settings.warn_if_non_loopback(logger)
            async with open_services(settings) as services:
                service_ref["service"] = services.service
                app.state.service = services.service
                tasks = BackgroundTasks(settings, services.service.graph)
                tasks.start()
                logger.info(
                    "mouseion_daemon_started",
                    host=settings.mouseion_host,
                    port=settings.mouseion_port,
                )
                try:
                    yield
                finally:
                    await tasks.stop()
                    logger.info("mouseion_daemon_stopped")

    app = FastAPI(title="Mouseion", lifespan=lifespan)
    app.router.routes.extend(mcp_app.routes)

    web_dir = Path(__file__).parent / "web"
    app.mount("/web", StaticFiles(directory=web_dir), name="web")

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            structlog.contextvars.clear_contextvars()

    @app.exception_handler(MouseionError)
    async def mouseion_error_handler(_: Request, exc: MouseionError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/stats")
    async def api_stats(request: Request) -> JsonDict:
        return await _service(request).stats()

    @app.get("/api/documents")
    async def api_documents(type: str = "all", limit: int = 50, offset: int = 0) -> JsonDict:
        return await service_ref["service"].list_documents(
            ListInput.model_validate({"type": type, "limit": limit, "offset": offset})
        )

    @app.get("/api/documents/{document_id}")
    async def api_document(document_id: str, request: Request) -> JsonDict:
        return await _service(request).get_document(
            GetDocumentInput.model_validate({"document_id": document_id})
        )

    @app.post("/api/url")
    async def api_add_url(input: AddUrlInput, request: Request) -> JsonDict:
        return await _service(request).add_url(input)

    @app.post("/api/memory")
    async def api_add_memory(input: AddMemoryInput, request: Request) -> JsonDict:
        return await _service(request).add_memory(input)

    @app.post("/api/repo")
    async def api_add_repo(input: AddRepoInput, request: Request) -> JsonDict:
        return await _service(request).add_repo(input)

    @app.post("/api/search")
    async def api_search(input: SearchInput, request: Request) -> JsonDict:
        return await _service(request).search(input)

    @app.post("/api/relate")
    async def api_relate(input: RelateInput, request: Request) -> JsonDict:
        return await _service(request).relate(input)

    @app.delete("/api/documents/{document_id}")
    async def api_delete(document_id: str, request: Request) -> JsonDict:
        return await _service(request).delete(DeleteInput.model_validate({"id": document_id}))

    @app.post("/api/export")
    async def api_export(request: Request) -> JsonDict:
        return await _service(request).export()

    @app.post("/api/recompute_edges")
    async def api_recompute_edges(request: Request) -> JsonDict:
        return await _service(request).recompute_edges()

    @app.get("/api/vector/status")
    async def api_vector_status(request: Request) -> JsonDict:
        return await _service(request).vector_status()

    @app.post("/api/vector/mode")
    async def api_vector_mode(input: VectorModeInput, request: Request) -> JsonDict:
        if input.qbits is not None and input.qbits not in {2, 3, 4}:
            raise HTTPException(status_code=422, detail="qbits must be one of 2, 3, or 4")
        return await _service(request).set_vector_mode(input.mode, input.qbits)

    @app.post("/api/vector/quantize")
    async def api_vector_quantize(input: VectorQuantizeInput, request: Request) -> JsonDict:
        if input.qbits not in {2, 3, 4}:
            raise HTTPException(status_code=422, detail="qbits must be one of 2, 3, or 4")
        return await _service(request).quantize_vectors(input.qbits, preload=input.preload)

    @app.post("/api/vector/cleanup")
    async def api_vector_cleanup(request: Request) -> JsonDict:
        return await _service(request).cleanup_quantized_vectors()

    @app.post("/api/files")
    async def api_add_file(
        request: Request,
        file: Annotated[UploadFile, File()],
        tags: Annotated[str, Form()] = "",
    ) -> JsonDict:
        upload_dir = settings.files_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / f"{uuid4()}-{Path(file.filename or 'upload').name}"
        with destination.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        tag_values = [tag.strip() for tag in tags.split(",") if tag.strip()]
        return await _service(request).add_file(
            AddFileInput(file_path=str(destination), tags=tag_values)
        )

    return app


def _service(request: Request) -> MouseionService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Mouseion service is not ready")
    return cast(MouseionService, service)
