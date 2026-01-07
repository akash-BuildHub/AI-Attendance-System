import os
import cv2
from datetime import datetime
from app.core.config import VIDEOS_DIR, sanitize_camera_name

class VideoRecorder:
    _session_counts = {}

    def __init__(self):
        self.writer = None
        self.path = None
        self.session_index = None

    def start(self, first_frame, camera_name: str, fps: int = 20):
        safe = sanitize_camera_name(camera_name)
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        key = (safe, date_dir)
        count = self._session_counts.get(key, 0) + 1
        self._session_counts[key] = count
        self.session_index = count
        camera_dir = os.path.join(VIDEOS_DIR, safe, date_dir)
        os.makedirs(camera_dir, exist_ok=True)
        self.path = os.path.join(camera_dir, f"session_{count}.mp4")

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, (w, h))

    def write(self, frame):
        if self.writer:
            self.writer.write(frame)

    def stop(self):
        if self.writer:
            self.writer.release()
        self.writer = None
        return self.path
