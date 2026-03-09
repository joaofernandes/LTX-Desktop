"""Route handlers for /api/inputs/upload — web-mode image upload endpoint."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api", tags=["inputs"])


def _inputs_dir() -> Path:
    app_data = os.environ.get("LTX_APP_DATA_DIR", str(Path(__file__).parent.parent))
    d = Path(app_data) / "input"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("/inputs/upload")
async def upload_input_image(file: UploadFile = File(...)) -> JSONResponse:
    """Upload an image file to the server's input directory.

    Returns:
        { "url": "/inputs/<filename>", "path": "<absolute server path>" }
    """
    suffix = Path(file.filename or "image.png").suffix or ".png"
    filename = f"{uuid.uuid4().hex}{suffix}"
    dest = _inputs_dir() / filename
    content = await file.read()
    dest.write_bytes(content)
    return JSONResponse({"url": f"/inputs/{filename}", "path": str(dest)})
