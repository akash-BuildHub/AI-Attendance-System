import cv2
import time
import threading
import numpy as np
from typing import Optional

# Better RTSP stability
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|stimeout;5000000|rw_timeout;5000005"
)

class CameraStream:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        except:
            pass
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        if not self.cap or not self.cap.isOpened():
            self.running = False
            return

        # drop initial buffer
        for _ in range(8):
            self.cap.grab()

        while self.running:
            if not self.cap.grab():
                time.sleep(0.02)
                continue
            ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            with self.lock:
                self.latest_frame = frame
            time.sleep(0.001)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap = None
        self.latest_frame = None
        self.thread = None
