import json, os, re
from pathlib import Path

# Fix: Use correct parent directory navigation
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PERSON_IMAGES_DIR = DATA_DIR / "person_images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PROTOTYPES_PATH = EMBEDDINGS_DIR / "prototypes.pkl"

CAPTURED_DIR = DATA_DIR / "captured_images"
CAPTURED_KNOWN_DIR = CAPTURED_DIR / "known"
CAPTURED_UNKNOWN_DIR = CAPTURED_DIR / "unknown"

# Required by media / attendance
IMAGES_DIR = CAPTURED_DIR

VIDEOS_DIR = DATA_DIR / "videos"

# Create all necessary directories
for d in [
    PERSON_IMAGES_DIR, EMBEDDINGS_DIR,
    CAPTURED_KNOWN_DIR, CAPTURED_UNKNOWN_DIR,
    VIDEOS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

DEFAULTS = {
    "cosine_threshold": 0.48,
    "min_face_size": 70,
    "min_det_conf": 0.45,
    "capture_cooldown_sec": 8,
    "stream_fps": 15,
    "record_fps": 20,
}

def load_runtime_config():
    """Load runtime configuration from optimal_config.json if available"""
    cfg = DEFAULTS.copy()
    path = BASE_DIR / "optimal_config.json"
    if path.exists():
        try:
            with open(path) as f:
                raw = json.load(f)
            cfg.update({
                "cosine_threshold": raw.get("tolerance", cfg["cosine_threshold"]),
                "min_face_size": raw.get("min_face_size", cfg["min_face_size"]),
                "min_det_conf": raw.get("min_detection_confidence", cfg["min_det_conf"]),
            })
        except Exception as e:
            print(f"⚠️ Error loading optimal_config.json: {e}")
    return cfg

RUNTIME = load_runtime_config()

def sanitize_camera_name(name: str) -> str:
    """Sanitize camera name to be filesystem-safe"""
    return re.sub(r"[^\w]+", "_", name).strip("_") or "camera"