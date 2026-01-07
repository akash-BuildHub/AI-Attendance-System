import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DEFAULT_PERSON_IMAGES_DIR = DATA_DIR / "person_images"
LEGACY_PERSON_IMAGES_DIR = BASE_DIR / "Image Data" / "Person Images"
PERSON_IMAGES_DIR = LEGACY_PERSON_IMAGES_DIR if os.path.isdir(LEGACY_PERSON_IMAGES_DIR) else DEFAULT_PERSON_IMAGES_DIR
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PROTOTYPES_PATH = EMBEDDINGS_DIR / "prototypes.pkl"

IMAGES_DIR = DATA_DIR / "images"
CAPTURED_DIR = IMAGES_DIR
CAPTURED_KNOWN_DIR = CAPTURED_DIR / "known"
CAPTURED_UNKNOWN_DIR = CAPTURED_DIR / "unknown"

VIDEOS_DIR = DATA_DIR / "videos"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Public URL for clickable links (must be server IP, not localhost)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

# Default settings (overridden by optimal_config.json if present)
DEFAULTS = {
    "cosine_threshold": 0.48,          # ArcFace cosine similarity threshold
    "min_face_size": 70,               # reject tiny faces
    "min_det_conf": 0.45,              # YOLO det confidence
    "capture_cooldown_sec": 8,         # per-track capture cooldown
    "stream_fps": 15,                  # MJPEG output
    "record_fps": 20
}

def load_runtime_config() -> dict:
    cfg = DEFAULTS.copy()
    if os.path.exists("optimal_config.json"):
        try:
            with open("optimal_config.json", "r") as f:
                raw = json.load(f)
            # keep your old keys if present
            if "tolerance" in raw:
                cfg["cosine_threshold"] = float(raw["tolerance"])
            if "min_face_size" in raw:
                cfg["min_face_size"] = int(raw["min_face_size"])
            if "min_detection_confidence" in raw:
                cfg["min_det_conf"] = float(raw["min_detection_confidence"])
        except Exception:
            pass
    return cfg

RUNTIME = load_runtime_config()

def sanitize_camera_name(name: str) -> str:
    cleaned = re.sub(r"[^\w]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "camera"

def ensure_dirs():
    os.makedirs(PERSON_IMAGES_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(CAPTURED_KNOWN_DIR, exist_ok=True)
    os.makedirs(CAPTURED_UNKNOWN_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
