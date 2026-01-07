import time
import cv2
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.core.config import RUNTIME
from app.services.camera_stream import CameraStream
from app.services.video_recorder import VideoRecorder
from app.services.detector_yolo import YoloFaceDetector
from app.services.embedder_arcface import ArcFaceEmbedder, expand_bbox_for_head_neck
from app.services.tracker_bytetrack import ByteTrackerService
from app.services.matcher import match_embedding
from app.services.capture_service import save_capture
from app.services.attendance_service import log_to_sheet
from app.services.training_service import load_prototypes

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
    latest_annotated: Optional[any] = None

class SessionManager:
    def __init__(self):
        self.sessions: Dict[int, CameraSession] = {}

    def start(self, camera_id: int, camera_name: str, rtsp_url: str):
        # Load prototypes (trained embeddings)
        prototypes = load_prototypes()
        if camera_id in self.sessions:
            self.stop(camera_id)

        sess = CameraSession(
            camera_id=camera_id,
            camera_name=camera_name,
            rtsp_url=rtsp_url,
            stream=CameraStream(rtsp_url),
            recorder=VideoRecorder(),
            detector=YoloFaceDetector(),
            embedder=ArcFaceEmbedder(),
            tracker=ByteTrackerService(),
            prototypes=prototypes,
            running=True
        )

        sess.stream.start()
        self.sessions[camera_id] = sess
        return sess

    def stop(self, camera_id: int):
        sess = self.sessions.get(camera_id)
        if not sess:
            return None
        sess.running = False
        sess.stream.stop()
        video_path = sess.recorder.stop()
        self.sessions.pop(camera_id, None)
        return video_path

    def get(self, camera_id: int) -> Optional[CameraSession]:
        return self.sessions.get(camera_id)

    def process_once(self, camera_id: int):
        sess = self.sessions.get(camera_id)
        if not sess or not sess.running:
            return None

        frame = sess.stream.read()
        if frame is None:
            return None

        # Start recording when first frame arrives
        if sess.recorder.writer is None:
            sess.recorder.start(frame, sess.camera_name, fps=RUNTIME["record_fps"])

        sess.recorder.write(frame)

        detections = sess.detector.detect(frame, conf=RUNTIME["min_det_conf"])
        if not detections:
            sess.latest_annotated = frame
            return {"faces": []}

        xyxy = []
        confs = []
        for d in detections:
            x1,y1,x2,y2 = d["xyxy"]
            # filter tiny faces early
            if (x2-x1) < RUNTIME["min_face_size"] or (y2-y1) < RUNTIME["min_face_size"]:
                continue
            xyxy.append([x1,y1,x2,y2])
            confs.append(d["conf"])

        if not xyxy:
            sess.latest_annotated = frame
            return {"faces": []}

        import numpy as np
        tracks = sess.tracker.update(np.array(xyxy, dtype=int), np.array(confs, dtype=float))

        faces_out = []
        h, w = frame.shape[:2]

        for i in range(len(tracks)):
            tid = int(tracks.tracker_id[i])
            x1,y1,x2,y2 = tracks.xyxy[i].astype(int)

            # expand bbox to include head+neck
            nx1, ny1, nx2, ny2 = expand_bbox_for_head_neck(x1,y1,x2,y2,w,h)
            crop = frame[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                continue

            emb = sess.embedder.embed_face_crop(crop)
            if emb is None:
                continue

            label, score = match_embedding(emb, sess.prototypes, threshold=RUNTIME["cosine_threshold"])

            # capture ONCE per track (cooldown)
            now = time.time()
            last = sess.last_capture_by_track.get(tid, 0)
            if now - last >= RUNTIME["capture_cooldown_sec"]:
                sess.last_capture_by_track[tid] = now
                img_path, _dt = save_capture(crop, label, sess.camera_name)
                log_to_sheet(sess.camera_name, label, img_path)

            # draw bbox
            color = (0,255,0) if label != "Unknown" else (0,0,255)
            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), color, 2)
            text = f"{label} ({score:.2f}) ID:{tid}"
            cv2.putText(frame, text, (nx1, max(25, ny1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            faces_out.append({
                "track_id": tid,
                "label": label,
                "score": float(score),
                "bbox": [int(nx1), int(ny1), int(nx2), int(ny2)]
            })

        sess.latest_annotated = frame
        return {"faces": faces_out}

manager = SessionManager()
