import os
import mimetypes
import urllib.parse
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import CAPTURED_DIR

router = APIRouter(prefix="/media", tags=["media"])

@router.get("/{path:path}")
def media(path: str):
    base = os.path.abspath(CAPTURED_DIR)
    if not os.path.exists(base):
        raise HTTPException(status_code=404, detail="captured_images folder not found")

    decoded = urllib.parse.unquote(path).replace("/", os.sep)
    full = os.path.abspath(os.path.join(base, decoded))

    # prevent directory traversal
    if not full.startswith(base):
        raise HTTPException(status_code=403, detail="forbidden")

    if not os.path.exists(full) or not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="file not found")

    mime, _ = mimetypes.guess_type(full)
    return FileResponse(full, media_type=mime or "application/octet-stream",
                        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=3600"})
