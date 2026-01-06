import os
import cv2
from datetime import datetime
from app.core.config import VIDEOS_DIR

class VideoRecorder:
    def __init__(self):
        self.writer = None
        self.path = None

    def start(self, first_frame, camera_name: str, fps: int = 20):
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe = camera_name.replace("/", "_").replace("\\", "_").strip() or "Camera"
        self.path = os.path.join(VIDEOS_DIR, f"{safe}_{ts}.mp4")

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
