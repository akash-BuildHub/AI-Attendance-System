import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from app.core.config import RUNTIME
from app.core.state import state
from app.services.camera_stream import CameraStream
from app.services.video_recorder import VideoRecorder
from app.services.detector_yolo import YoloFaceDetector
from app.services.embedder_arcface import ArcFaceEmbedder, expand_bbox_for_head_neck
from app.services.tracker_bytetrack import ByteTrackerService
from app.services.matcher import match_embedding
from app.services.capture_service import save_capture
from app.services.attendance_service import log_to_sheet
from app.services.training_service import load_prototypes

logger = logging.getLogger(__name__)

@dataclass
class CameraSession:
    camera_id: int
    camera_name: str
    rtsp_url: str
    stream: CameraStream
    recorder: VideoRecorder
    detector: YoloFaceDetector
    embedder: ArcFaceEmbedder
    tracker: ByteTrackerService
    prototypes: dict
    running: bool = False
    last_capture_by_track: Dict[int, float] = field(default_factory=dict)
    last_label_by_track: Dict[int, str] = field(default_factory=dict)
    last_label_time_by_track: Dict[int, float] = field(default_factory=dict)
    logged_labels: set = field(default_factory=set)
    latest_annotated: Optional[np.ndarray] = None
    latest_frame: Optional[np.ndarray] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    last_frame_time: float = 0.0
    stream_restarts: int = 0

    def frame_generator(self):
        while self.running and not self.stop_event.is_set():
            frame = self.latest_annotated or self.latest_frame
            if frame is None:
                time.sleep(0.03)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

class CameraSessionManager:
    def __init__(self):
        self.sessions: Dict[int, CameraSession] = {}
        self.lock = threading.Lock()
        self.frame_timeout_sec = 5.0
        self.max_stream_restarts = 3

    def _require_camera_id(self, camera_id: int) -> int:
        try:
            camera_id = int(camera_id)
        except (TypeError, ValueError):
            raise RuntimeError("cameraId must be an integer")
        return camera_id

    def start_camera(self, camera_id: int, camera_name: str, rtsp_url: str) -> CameraSession:
        camera_id = self._require_camera_id(camera_id)
        with self.lock:
            existing = self.sessions.get(camera_id)
            if existing:
                if existing.running:
                    return existing
                existing.camera_name = camera_name
                if rtsp_url:
                    existing.rtsp_url = rtsp_url
                self._reset_session(existing)
                self._start_session(existing)
                return existing

            session = self._create_session(camera_id, camera_name, rtsp_url)
            self.sessions[camera_id] = session
            self._start_session(session)
            return session

    def stop_camera(self, camera_id: int) -> Optional[str]:
        camera_id = self._require_camera_id(camera_id)
        with self.lock:
            sess = self.sessions.get(camera_id)
            if not sess:
                return None
            path = self._stop_session(sess, remove=False)
            return path

    def resume_camera(self, camera_id: int) -> Optional[CameraSession]:
        camera_id = self._require_camera_id(camera_id)
        with self.lock:
            sess = self.sessions.get(camera_id)
            if not sess:
                return None
            if sess.running:
                return sess
            self._reset_session(sess)
            self._start_session(sess)
            return sess

    def get(self, camera_id: int) -> Optional[CameraSession]:
        try:
            camera_id = self._require_camera_id(camera_id)
        except RuntimeError:
            return None
        return self.sessions.get(camera_id)

    def frame_generator(self, camera_id: int):
        sess = self.get(camera_id)
        if not sess:
            return iter(())
        return sess.frame_generator()

    def list_cameras(self):
        return [
            {
                "camera_id": sess.camera_id,
                "camera_name": sess.camera_name,
                "rtsp_url": sess.rtsp_url,
                "running": sess.running,
            }
            for sess in self.sessions.values()
        ]

    def _create_session(self, camera_id: int, camera_name: str, rtsp_url: str) -> CameraSession:
        prototypes = load_prototypes()
        return CameraSession(
            camera_id=camera_id,
            camera_name=camera_name,
            rtsp_url=rtsp_url,
            stream=CameraStream(rtsp_url),
            recorder=VideoRecorder(),
            detector=YoloFaceDetector(),
            embedder=ArcFaceEmbedder(),
            tracker=ByteTrackerService(),
            prototypes=prototypes,
            running=False,
        )

    def _reset_session(self, session: CameraSession) -> None:
        session.stream = CameraStream(session.rtsp_url)
        session.recorder = VideoRecorder()
        session.detector = YoloFaceDetector()
        session.embedder = ArcFaceEmbedder()
        session.tracker = ByteTrackerService()
        session.prototypes = load_prototypes()
        session.last_capture_by_track.clear()
        session.last_label_by_track.clear()
        session.last_label_time_by_track.clear()
        session.logged_labels.clear()
        session.latest_annotated = None
        session.latest_frame = None
        session.stop_event.clear()
        session.last_frame_time = 0.0
        session.stream_restarts = 0

    def _start_session(self, session: CameraSession):
        if session.running:
            return
        session.running = True
        session.stream.start()
        session.thread = threading.Thread(
            target=self._process_loop, args=(session,), daemon=True
        )
        session.thread.start()

    def _stop_session(self, session: CameraSession, remove: bool) -> Optional[str]:
        session.running = False
        session.stop_event.set()
        try:
            session.stream.stop()
        except Exception:
            logger.exception("Failed stopping camera stream")
        path = None
        try:
            path = session.recorder.stop()
        except Exception:
            logger.exception("Failed stopping recorder")
        if session.thread and session.thread.is_alive():
            session.thread.join(timeout=2.0)
        if remove:
            self.sessions.pop(session.camera_id, None)
        return path

    def _restart_stream(self, session: CameraSession):
        session.stream_restarts += 1
        if session.stream_restarts > self.max_stream_restarts:
            logger.error("Camera %s exceeded restart limit", session.camera_id)
            session.running = False
            session.stop_event.set()
            return
        logger.warning("Restarting camera stream for %s", session.camera_id)
        try:
            session.stream.stop()
        except Exception:
            logger.exception("Failed to stop stream during restart")
        session.stream = CameraStream(session.rtsp_url)
        session.stream.start()
        session.last_frame_time = time.time()

    def _process_loop(self, session: CameraSession):
        threshold = float(RUNTIME["cosine_threshold"])
        min_face = int(RUNTIME["min_face_size"])
        min_conf = float(RUNTIME["min_det_conf"])
        capture_cooldown = float(RUNTIME["capture_cooldown_sec"])
        unknown_margin = 0.05
        try:
            while session.running and not session.stop_event.is_set():
                frame = session.stream.read()
                now = time.time()
                if frame is None:
                    if session.last_frame_time and (now - session.last_frame_time) > self.frame_timeout_sec:
                        self._restart_stream(session)
                    time.sleep(0.02)
                    continue

                session.last_frame_time = now
                session.latest_frame = frame

                if session.recorder.writer is None:
                    session.recorder.start(frame, session.camera_name, fps=RUNTIME["record_fps"])

                session.recorder.write(frame)

                detections = session.detector.detect(frame, conf=min_conf)
                if not detections:
                    session.latest_annotated = frame
                    continue

                xyxy = []
                confs = []
                for d in detections:
                    x1, y1, x2, y2 = d["xyxy"]
                    if (x2 - x1) < min_face or (y2 - y1) < min_face:
                        continue
                    xyxy.append([x1, y1, x2, y2])
                    confs.append(d["conf"])

                if not xyxy:
                    session.latest_annotated = frame
                    continue

                tracks = session.tracker.update(np.array(xyxy, dtype=int), np.array(confs, dtype=float))

                h, w = frame.shape[:2]

                for i in range(len(tracks)):
                    tid = int(tracks.tracker_id[i])
                    x1, y1, x2, y2 = tracks.xyxy[i].astype(int)

                    nx1, ny1, nx2, ny2 = expand_bbox_for_head_neck(x1, y1, x2, y2, w, h)
                    if (nx2 - nx1) < min_face or (ny2 - ny1) < min_face:
                        continue

                    crop = frame[ny1:ny2, nx1:nx2]
                    if crop.size == 0:
                        continue

                    emb = session.embedder.embed_face_crop(crop)
                    if emb is None:
                        continue

                    label, score = match_embedding(emb, session.prototypes, threshold=threshold)

                    last_label = session.last_label_by_track.get(tid)
                    last_label_time = session.last_label_time_by_track.get(tid, 0)
                    if label == "Unknown" and last_label and last_label != "Unknown":
                        if (now - last_label_time) < 2.0 and score >= (threshold - unknown_margin):
                            label = last_label

                    session.last_label_by_track[tid] = label
                    session.last_label_time_by_track[tid] = now

                    if label != "Unknown" and label not in session.logged_labels:
                        last_capture = session.last_capture_by_track.get(tid, 0)
                        if (now - last_capture) >= capture_cooldown and score >= threshold:
                            session.last_capture_by_track[tid] = now
                            img_path, _dt = save_capture(crop, label, session.camera_name)
                            log_to_sheet(session.camera_name, label, img_path)
                            session.logged_labels.add(label)
                    elif label == "Unknown":
                        last_capture = session.last_capture_by_track.get(tid, 0)
                        if (now - last_capture) >= capture_cooldown:
                            session.last_capture_by_track[tid] = now
                            save_capture(crop, label, session.camera_name)

                    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), color, 2)
                    text = f"{label} ({score:.2f}) ID:{tid}"
                    cv2.putText(frame, text, (nx1, max(25, ny1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                session.latest_annotated = frame
        except Exception:
            logger.exception("Processing loop failed for camera %s", session.camera_id)
        finally:
            session.running = False
            try:
                session.recorder.stop()
            except Exception:
                logger.exception("Failed stopping recorder on exit")

# ✅ SINGLE INSTANCE - NO MERGE CONFLICTS
if state.session_manager is None:
    state.session_manager = CameraSessionManager()

manager = state.session_manager