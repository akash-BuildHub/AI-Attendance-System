import os
import cv2
from datetime import datetime
from app.core.config import IMAGES_DIR, sanitize_camera_name

def save_capture(face_bgr, label: str, camera_name: str):
    safe_camera_name = sanitize_camera_name(camera_name)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S_%f")[:-3]

    label_dir = "unknown" if label == "Unknown" else "known"
    # ✅ Save directly to known/unknown folders (no camera subfolders)
    folder = os.path.join(str(IMAGES_DIR), label_dir)
    os.makedirs(folder, exist_ok=True)
    filename = f"{safe_camera_name}_{label}_{date_str}_{time_str}.jpg"
    path = os.path.join(folder, filename)

    cv2.imwrite(path, face_bgr)
    return path, now