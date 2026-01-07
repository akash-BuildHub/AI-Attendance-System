import os
import cv2
from datetime import datetime
from app.core.config import VIDEOS_DIR, sanitize_camera_name

class VideoRecorder:
    def __init__(self):
        self.writer = None
        self.path = None

    def start(self, first_frame, camera_name: str, fps: int = 20):
        safe = sanitize_camera_name(camera_name)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        camera_dir = os.path.join(VIDEOS_DIR, safe)
        os.makedirs(camera_dir, exist_ok=True)
        self.path = os.path.join(camera_dir, f"{safe}_{ts}.mp4")

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
