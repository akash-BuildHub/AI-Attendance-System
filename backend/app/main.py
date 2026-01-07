import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import ensure_dirs, BASE_URL
from app.routes.camera import router as camera_router, legacy_router as camera_legacy_router, compat_router as camera_compat_router
from app.routes.training import router as training_router
from app.routes.attendance import router as attendance_router
from app.routes.media import router as media_router

ensure_dirs()

app = FastAPI(title="Grow AI - YOLOv5Face + ArcFace + ByteTrack")

cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(camera_router)
app.include_router(camera_legacy_router)
app.include_router(camera_compat_router)
app.include_router(training_router)
app.include_router(attendance_router)
app.include_router(media_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "base_url": BASE_URL,
        "model": "YOLOv5-Face + ArcFace + ByteTrack",
        "notes": [
            "Train first: POST /train",
            "Start camera: POST /cameras/start",
            "Stream: GET /cameras/stream/{camera_id}",
            "Media: /media/<relative_path_inside_images>"
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}
