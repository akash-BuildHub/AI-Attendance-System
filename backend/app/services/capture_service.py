import os
import cv2
from datetime import datetime
from app.core.config import CAPTURED_KNOWN_DIR, CAPTURED_UNKNOWN_DIR

def save_capture(face_bgr, label: str, camera_name: str):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S_%f")[:-3]

    if label == "Unknown":
        folder = CAPTURED_UNKNOWN_DIR
        filename = f"{camera_name}_unknown_{date_str}_{time_str}.jpg"
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
    else:
        folder = os.path.join(CAPTURED_KNOWN_DIR, label)
        os.makedirs(folder, exist_ok=True)
        filename = f"{camera_name}_{label}_{date_str}_{time_str}.jpg"
        path = os.path.join(folder, filename)

    cv2.imwrite(path, face_bgr)
    return path, now
