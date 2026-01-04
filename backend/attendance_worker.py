import cv2
import time
import json
import threading
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Tuple
import os
import urllib.parse

from face_store import face_store
from google_api import google_api


class AttendanceWorker:
    def __init__(self, camera_id: int, camera_name: str, rtsp_url: str):
        self.camera_id = camera_id
        self.camera_name = (camera_name or f"Camera_{camera_id}").strip() or f"Camera_{camera_id}"
        self.rtsp_url = urllib.parse.unquote(rtsp_url)

        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.cap: Optional[cv2.VideoCapture] = None

        # shared frames
        self.latest_frame: Optional[np.ndarray] = None      # annotated + resized for stream
        self._raw_frame: Optional[np.ndarray] = None        # raw full frame for recording/cropping
        self.frame_lock = threading.Lock()

        # Video recording
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.temp_videos_dir = "temp_videos"
        os.makedirs(self.temp_videos_dir, exist_ok=True)
        self.local_video_path: Optional[str] = None

        # Attendance tracking
        self.attendance_log: Dict[str, datetime] = {}
        self.cooldown_seconds = 30

        # Performance tuning
        self.stream_width = 960
        self.downscale_factor = 0.25
        self.recognition_interval_sec = 0.20  # run face recognition every 200ms

        # Recognition tuning - increased tolerance for better CCTV recognition
        self.tolerance = 0.50  # Increased from 0.45

        self.fps = 0.0
        self.frame_count = 0
        self.start_time = None

        print(f"🎥 Initialized camera: {self.camera_name}")
        print(f"📡 RTSP URL: {self.rtsp_url}")

    def start(self):
        if self.running:
            return False
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"🚀 Started worker for {self.camera_name}")
        return True

    def stop(self):
        if not self.running:
            return

        print(f"🛑 Stopping {self.camera_name}...")
        self.running = False

        # hard-release cap to break blocking reads
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
        self.video_writer = None

        self._upload_video()
        self.attendance_log.clear()

        with self.frame_lock:
            self.latest_frame = None
            self._raw_frame = None

        print(f"✅ Stopped {self.camera_name}")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def get_status(self) -> Dict[str, any]:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "running": self.running,
            "camera_name": self.camera_name,
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count,
            "people_tracked": len(self.attendance_log),
            "uptime_seconds": round(elapsed, 1) if elapsed else 0,
            "cooldown": self.cooldown_seconds
        }

    def _run(self):
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        # low-latency settings
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        except:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        except:
            pass

        if not self.cap.isOpened():
            print(f"❌ Failed to open: {self.camera_name}")
            self.running = False
            return

        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 0
        if fps <= 0 or fps > 60:
            fps = 20  # robust fallback

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        print(f"📹 Camera: {width}x{height} @ {fps} FPS")
        self._start_video_recording(width, height, fps)

        self.start_time = time.time()
        self.frame_count = 0
        last_recognition = 0.0
        last_fps_calc = time.time()
        frames_since_calc = 0

        annotated_cache: Optional[np.ndarray] = None

        while self.running:
            # grab/retrieve pattern for better RTSP performance
            try:
                grabbed = self.cap.grab()
                if not grabbed:
                    time.sleep(0.05)
                    continue
                ret, frame = self.cap.retrieve()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue
            except Exception:
                time.sleep(0.05)
                continue

            now_ts = time.time()

            # keep raw frame fresh
            with self.frame_lock:
                self._raw_frame = frame

            # record video (raw, full res)
            if self.video_writer and self.video_writer.isOpened():
                try:
                    self.video_writer.write(frame)
                except:
                    pass

            # run recognition at fixed interval on snapshot
            if (now_ts - last_recognition) >= self.recognition_interval_sec:
                snap = frame.copy()
                annotated_cache = self._process_frame(snap)  # returns annotated full frame
                last_recognition = now_ts

            # stream frame: if recognition didn't run, reuse last annotated frame
            out = annotated_cache if annotated_cache is not None else frame
            with self.frame_lock:
                self.latest_frame = self._resize_for_stream(out)

            self.frame_count += 1
            frames_since_calc += 1

            if (time.time() - last_fps_calc) >= 2.0:
                elapsed = time.time() - last_fps_calc
                self.fps = frames_since_calc / elapsed if elapsed > 0 else 0.0
                frames_since_calc = 0
                last_fps_calc = time.time()

            # tiny sleep to reduce CPU but keep low latency
            time.sleep(0.001)

        # cleanup
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass
        self.cap = None

        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
        self.video_writer = None

        print(f"🧹 Cleaned up: {self.camera_name}")

    def _resize_for_stream(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] <= self.stream_width:
            return frame
        ratio = self.stream_width / frame.shape[1]
        new_height = int(frame.shape[0] * ratio)
        return cv2.resize(frame, (self.stream_width, new_height), interpolation=cv2.INTER_AREA)

    def _start_video_recording(self, width: int, height: int, fps: int):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cam = self.camera_name.replace("/", "_").replace("\\", "_")
        filename = f"{safe_cam}_{ts}.mp4"
        self.local_video_path = os.path.join(self.temp_videos_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.local_video_path, fourcc, fps, (width, height))

        if self.video_writer.isOpened():
            print(f"🎬 Started recording: {filename}")
        else:
            print("❌ Failed to start recording")
            self.video_writer = None

    def _upload_video(self):
        if not self.local_video_path or not os.path.exists(self.local_video_path):
            return

        size = os.path.getsize(self.local_video_path)
        if size < 1024:
            try:
                os.remove(self.local_video_path)
            except:
                pass
            self.local_video_path = None
            return

        try:
            if google_api.is_authenticated():
                link = google_api.upload_video_segment(self.camera_name, self.local_video_path)
                if link:
                    print(f"✅ Video uploaded to Drive: {self.camera_name} -> {link}")
                else:
                    print(f"❌ Video upload failed (no link returned): {self.camera_name}")
            else:
                print("⚠️ Google not authenticated, skipping video upload")

            try:
                os.remove(self.local_video_path)
            except:
                pass

            self.local_video_path = None

        except Exception as e:
            print(f"❌ Error uploading video: {e}")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns annotated frame.
        Also logs attendance + uploads captured faces.
        """
        try:
            small = cv2.resize(
                frame, (0, 0),
                fx=self.downscale_factor,
                fy=self.downscale_factor,
                interpolation=cv2.INTER_AREA
            )
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            faces = face_store.recognize_faces(rgb_small, tolerance=self.tolerance, model="hog", top_k=6)
            scale = 1.0 / self.downscale_factor

            for face in faces:
                name = face["name"]
                left, top, right, bottom = face["location"]
                is_known = face["is_known"]
                conf = float(face.get("confidence", 0.0))

                # scale bbox back
                left = int(left * scale)
                top = int(top * scale)
                right = int(right * scale)
                bottom = int(bottom * scale)

                # draw
                self._draw_face_info(frame, name, conf, is_known, (left, top, right, bottom))

                # crop safely
                h, w = frame.shape[:2]
                left_c = max(0, left)
                top_c = max(0, top)
                right_c = min(w, right)
                bottom_c = min(h, bottom)
                if right_c <= left_c or bottom_c <= top_c:
                    continue

                face_crop = frame[top_c:bottom_c, left_c:right_c]

                if is_known:
                    now = datetime.now()
                    last_seen = self.attendance_log.get(name)
                    if last_seen and (now - last_seen).total_seconds() < self.cooldown_seconds:
                        continue
                    self.attendance_log[name] = now

                    # Log known face attendance
                    self._log_attendance(name, face_crop)
                else:
                    # Upload unknown face but don't log attendance
                    self._upload_unknown_face(name, face_crop)

            return frame

        except Exception as e:
            print(f"❌ Face processing error in {self.camera_name}: {e}")
            return frame

    def _log_attendance(self, name: str, face_image: np.ndarray):
        try:
            if not google_api.is_authenticated():
                return

            success, buffer = cv2.imencode(".jpg", face_image)
            if not success:
                return
            image_bytes = buffer.tobytes()

            # upload captured face
            face_image_url = google_api.upload_captured_face(
                self.camera_name,
                name,
                image_bytes
            )

            # log to sheet only for known faces
            if face_image_url:
                google_api.log_attendance(self.camera_name, name, face_image_url)
                print(f"✅ Face uploaded + logged: {name} @ {self.camera_name}")
            else:
                print(f"⚠️ Face URL empty for {name}")

        except Exception as e:
            print(f"❌ Logging error for {name}: {e}")

    def _upload_unknown_face(self, name: str, face_image: np.ndarray):
        """Upload unknown faces to Drive but don't log to sheet"""
        try:
            if not google_api.is_authenticated():
                return

            success, buffer = cv2.imencode(".jpg", face_image)
            if not success:
                return
            image_bytes = buffer.tobytes()

            # upload to Unknown folder
            face_image_url = google_api.upload_captured_face(
                self.camera_name,
                name,
                image_bytes
            )

            if face_image_url:
                print(f"📸 Unknown face uploaded: {name} @ {self.camera_name}")

        except Exception as e:
            print(f"❌ Error uploading unknown face: {e}")

    def _draw_face_info(self, frame, name: str, confidence: float, is_known: bool, box: Tuple[int, int, int, int]):
        left, top, right, bottom = box
        color = (0, 255, 0) if is_known else (0, 0, 255)

        # thicker box for visibility
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)

        label = f"{name} ({confidence:.2f})" if is_known else "Unknown"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.85
        thickness = 2

        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

        # place label above box, clamp within image
        y = top - 10
        if y - th - 10 < 0:
            y = bottom + th + 10  # move below if too high

        x1 = max(0, left)
        y1 = max(0, y - th - 10)
        x2 = min(frame.shape[1] - 1, left + tw + 12)
        y2 = min(frame.shape[0] - 1, y + 4)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, label, (x1 + 6, y2 - 6), font, scale, (255, 255, 255), thickness)

    def _log_locally(self, data: Dict[str, any]):
        try:
            safe_cam = self.camera_name.replace("/", "_").replace("\\", "_")
            log_file = f"{safe_cam}_attendance_backup.jsonl"
            log_entry = {
                **data,
                "timestamp": datetime.now().isoformat(),
                "camera_id": self.camera_id,
                "date": datetime.now().strftime("%d-%m-%Y"),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except:
            pass