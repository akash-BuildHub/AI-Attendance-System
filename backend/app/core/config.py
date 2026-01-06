import json
import os

DATA_DIR = "data"
PERSON_IMAGES_DIR = os.path.join(DATA_DIR, "person_images")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
PROTOTYPES_PATH = os.path.join(EMBEDDINGS_DIR, "prototypes.pkl")

CAPTURED_DIR = os.path.join(DATA_DIR, "captured_images")
CAPTURED_KNOWN_DIR = os.path.join(CAPTURED_DIR, "known")
CAPTURED_UNKNOWN_DIR = os.path.join(CAPTURED_DIR, "unknown")

VIDEOS_DIR = os.path.join(DATA_DIR, "videos")

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

def ensure_dirs():
    os.makedirs(PERSON_IMAGES_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(CAPTURED_KNOWN_DIR, exist_ok=True)
    os.makedirs(CAPTURED_UNKNOWN_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
