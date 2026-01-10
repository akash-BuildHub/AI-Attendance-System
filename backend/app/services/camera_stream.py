import cv2
from app.core.state import app_state


class CameraStream:
    """
    Supports:
    - Webcam (0, 1, ...)
    - RTSP streams (rtsp://user:pass@ip:554/...)
    """

    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.running = False

    def start(self):
        with app_state.camera_lock:
            if self.running:
                return

            if isinstance(self.source, str) and self.source.startswith("rtsp"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap or not self.cap.isOpened():
                self.cap = None
                raise RuntimeError(f"Cannot open camera source: {self.source}")

            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            self.running = True
            app_state.camera_running = True

    def read(self):
        if not self.running or not self.cap:
            return None

        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def stop(self):
        with app_state.camera_lock:
            self.running = False
            app_state.camera_running = False

            if self.cap:
                self.cap.release()
                self.cap = None
