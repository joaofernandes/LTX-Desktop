"""FastAPI app factory decoupled from runtime bootstrap side effects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from _routes._errors import HTTPError
from _routes.generation import router as generation_router
from _routes.health import router as health_router
from _routes.ic_lora import router as ic_lora_router
from _routes.image_gen import router as image_gen_router
from _routes.inputs import router as inputs_router
from _routes.models import router as models_router
from _routes.suggest_gap_prompt import router as suggest_gap_prompt_router
from _routes.retake import router as retake_router
from _routes.runtime_policy import router as runtime_policy_router
from _routes.settings import router as settings_router
from logging_policy import log_http_error, log_unhandled_exception
from state import init_state_service

if TYPE_CHECKING:
    from app_handler import AppHandler

_web_mode = os.environ.get("LTX_USE_GGUF", "0") == "1" or os.environ.get("LTX_WEB_MODE", "0") == "1"

DEFAULT_ALLOWED_ORIGINS: list[str] = (
    ["*"]
    if _web_mode
    else [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
)


def create_app(
    *,
    handler: "AppHandler",
    allowed_origins: list[str] | None = None,
    title: str = "LTX-2 Video Generation Server",
) -> FastAPI:
    """Create a configured FastAPI app bound to the provided handler."""
    init_state_service(handler)

    app = FastAPI(title=title)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or DEFAULT_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def _route_http_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPError):
            log_http_error(request, exc)
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
        return JSONResponse(status_code=500, content={"error": str(exc)})

    async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            return JSONResponse(status_code=422, content={"error": str(exc)})
        return JSONResponse(status_code=422, content={"error": str(exc)})

    async def _route_generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log_unhandled_exception(request, exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPError, _route_http_error_handler)
    app.add_exception_handler(Exception, _route_generic_error_handler)

    app.include_router(health_router)
    app.include_router(generation_router)
    app.include_router(models_router)
    app.include_router(settings_router)
    app.include_router(image_gen_router)
    app.include_router(suggest_gap_prompt_router)
    app.include_router(retake_router)
    app.include_router(ic_lora_router)
    app.include_router(runtime_policy_router)
    app.include_router(inputs_router)

    # In web mode, serve outputs, inputs and the built frontend SPA
    if _web_mode:
        app_data_dir = Path(os.environ.get("LTX_APP_DATA_DIR", Path(__file__).parent))

        outputs_dir = app_data_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

        inputs_dir = app_data_dir / "input"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/inputs", StaticFiles(directory=str(inputs_dir)), name="inputs")

        frontend_dist = Path(__file__).parent.parent / "dist"
        if frontend_dist.exists():
            app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app
