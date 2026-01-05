import cv2
import time
import threading
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Tuple, List
import os
import urllib.parse
from collections import deque, defaultdict
import json

from face_store import face_store
from google_api import google_api

# 🔥 Enhanced FFmpeg options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "stimeout;5000000|"
    "rw_timeout;5000005"
)


class AttendanceWorker:
    """🔥 ARCFACE VERSION: Optimized for CCTV"""
    
    def __init__(self, camera_id: int, camera_name: str, rtsp_url: str):
        self.camera_id = camera_id
        self.camera_name = (camera_name or f"Camera_{camera_id}").strip() or f"Camera_{camera_id}"
        self.rtsp_url = urllib.parse.unquote(rtsp_url)
        
        # 🔥 FIXED: Always use environment variable
        self.base_url = os.environ.get("BASE_URL", "http://localhost:8000")
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            print(f"⚠️  WARNING: BASE_URL is set to {self.base_url}")
            print(f"   Google Sheets links will NOT work from other devices!")
            print(f"   Set with: export BASE_URL=http://YOUR_SERVER_IP:8000")
        
        self.running = False
        self.stop_event = threading.Event()
        self.rtsp_thread: Optional[threading.Thread] = None
        self.stream_thread: Optional[threading.Thread] = None
        self.recognition_thread: Optional[threading.Thread] = None
        self.adaptive_thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.raw_frame: Optional[np.ndarray] = None
        self.stream_frame: Optional[np.ndarray] = None
        self.recognition_queue = deque(maxlen=3)  # Reduced for performance
        
        self.raw_lock = threading.Lock()
        self.stream_lock = threading.Lock()
        self.recognition_lock = threading.Lock()
        self.writer_lock = threading.Lock()

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.local_video_path: Optional[str] = None

        self.attendance_log: Dict[str, datetime] = {}
        self.capture_log: Dict[str, datetime] = {}
        
        self.cooldown_seconds = 30
        self.capture_cooldown_seconds = 5

        # 🔥 FIXED: Load config with fallback
        self.load_optimal_config()
        
        self.stream_width = 960
        self.stream_fps = 15
        
        # Tracking
        self.active_tracks: Dict[str, Dict] = {}
        self.track_id_counter = 0
        self.track_timeout = 0.9
        self.iou_threshold = 0.3
        self.min_track_age_for_logging = 1.5  # Increased for stability
        
        # Performance metrics
        self.fps = 0.0
        self.frame_count = 0
        self.recognition_fps = 0.0
        self.recognition_count = 0
        self.start_time = None
        
        # Accuracy tracking
        self.false_unknown_count = 0
        self.correct_recognition_count = 0
        self.total_recognition_count = 0
        self.genuine_unknown_count = 0

        # Directories
        self.video_base_dir = "AI Attendance Videos"
        self.captured_base_dir = "Captured Images"
        
        self.captured_known_dir = os.path.join(self.captured_base_dir, "Known Images")
        self.captured_unknown_dir = os.path.join(self.captured_base_dir, "Unknown Images")
        
        os.makedirs(os.path.join(self.video_base_dir, self.camera_name), exist_ok=True)
        os.makedirs(self.captured_known_dir, exist_ok=True)
        os.makedirs(self.captured_unknown_dir, exist_ok=True)

        self.rtsp_failure_count = 0
        self.max_rtsp_failures = 30  # Reduced
        self.last_frame_time = time.time()

        print(f"🎥 Initialized camera: {self.camera_name}")
        print(f"   📁 Known faces → {self.captured_known_dir}")
        print(f"   📁 Unknown faces → {self.captured_unknown_dir}")
        print(f"   🔗 Base URL → {self.base_url}")
        print(f"   🧠 Using: ArcFace (InsightFace)")
        print(f"   ⚙️ Config: tolerance={self.tolerance}, interval={self.recognition_interval_sec}s")

    def load_optimal_config(self):
        """Load or create optimal configuration"""
        config_file = "optimal_config.json"
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                self.tolerance = config.get('tolerance', 0.47)
                self.downscale_factor = config.get('downscale_factor', 0.35)
                self.recognition_interval_sec = config.get('recognition_interval', 0.8)
                self.num_jitters = config.get('num_jitters', 0)
                self.top_k = config.get('top_k', 1)
                
                print(f"✅ Loaded config from {config_file}")
            else:
                # 🔥 Create default config for CCTV
                default_config = {
                    "tolerance": 0.47,
                    "downscale_factor": 0.35,
                    "recognition_interval": 0.8,
                    "num_jitters": 0,
                    "top_k": 1
                }
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                
                self.tolerance = 0.47
                self.downscale_factor = 0.35
                self.recognition_interval_sec = 0.8
                self.num_jitters = 0
                self.top_k = 1
                
                print(f"📝 Created default config at {config_file}")
                
        except Exception as e:
            print(f"⚠️ Config error, using defaults: {e}")
            # 🔥 CCTV-optimized defaults
            self.tolerance = 0.47
            self.downscale_factor = 0.35
            self.recognition_interval_sec = 0.8
            self.num_jitters = 0
            self.top_k = 1
        
        self.recognition_fps_target = 1.0 / self.recognition_interval_sec
        print(f"ℹ️ Recognition interval: {self.recognition_interval_sec}s ({self.recognition_fps_target:.1f} FPS)")

    def start(self):
        if self.running:
            return False
        self.running = True
        self.stop_event.clear()
        
        self.rtsp_thread = threading.Thread(target=self._rtsp_capture_loop, daemon=True, name="RTSP")
        self.stream_thread = threading.Thread(target=self._stream_render_loop, daemon=True, name="Stream")
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True, name="Recognition")
        self.adaptive_thread = threading.Thread(target=self._adaptive_control_loop, daemon=True, name="Adaptive")
        
        self.rtsp_thread.start()
        self.stream_thread.start()
        self.recognition_thread.start()
        self.adaptive_thread.start()
        
        print(f"🚀 Started worker: {self.camera_name} (ArcFace)")
        return True

    def stop(self):
        """Thread-safe stop"""
        if not self.running:
            return

        print(f"🛑 Stopping {self.camera_name}...")

        self.stop_event.set()
        self.running = False

        for thread in [self.rtsp_thread, self.stream_thread, self.recognition_thread, self.adaptive_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)

        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass
        self.cap = None

        with self.writer_lock:
            try:
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.release()
            except:
                pass
            self.video_writer = None

        self.attendance_log.clear()
        self.capture_log.clear()
        self.active_tracks.clear()
        self.recognition_queue.clear()

        with self.raw_lock:
            self.raw_frame = None
        with self.stream_lock:
            self.stream_frame = None

        print(f"✅ Cleanly stopped {self.camera_name}")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.stream_lock:
            return None if self.stream_frame is None else self.stream_frame.copy()

    def get_status(self) -> Dict[str, any]:
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        accuracy = (self.correct_recognition_count / max(1, self.total_recognition_count)) * 100
        false_unknown_rate = (self.false_unknown_count / max(1, self.total_recognition_count)) * 100
        
        return {
            "running": self.running,
            "camera_name": self.camera_name,
            "capture_fps": round(self.fps, 1),
            "recognition_fps": round(self.recognition_fps, 1),
            "frame_count": self.frame_count,
            "recognition_count": self.recognition_count,
            "people_tracked": len(self.attendance_log),
            "active_tracks": len(self.active_tracks),
            "uptime_seconds": round(elapsed, 1) if elapsed else 0,
            "model": "arcface",
            "tolerance": self.tolerance,
            "recognition_interval": self.recognition_interval_sec,
            "accuracy": round(accuracy, 1),
            "false_unknown_rate": round(false_unknown_rate, 1),
            "captures_saved": len(self.capture_log),
            "version": "production_v11_arcface"
        }

    def _rtsp_capture_loop(self):
        """THREAD 1: RTSP capture"""
        print(f"📹 Connecting to RTSP: {self.camera_name}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        except:
            pass

        if not self.cap.isOpened():
            print(f"❌ Failed to open: {self.camera_name}")
            self.running = False
            return

        # Skip initial buffered frames
        for _ in range(10):
            self.cap.grab()

        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        print(f"📹 RTSP Connected: {width}x{height} @ {fps} FPS")
        self._start_video_recording(width, height, fps)

        self.start_time = time.time()
        self.frame_count = 0
        frames_since_calc = 0
        last_fps_calc = time.time()
        last_recognition_send = 0

        while self.running and not self.stop_event.is_set():
            if not self.cap.grab():
                self.rtsp_failure_count += 1
                
                if self.rtsp_failure_count > self.max_rtsp_failures:
                    print(f"⚠️ RTSP stalled - Reconnecting...")
                    try:
                        self.cap.release()
                        time.sleep(2)
                        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        for _ in range(5):
                            self.cap.grab()
                        self.rtsp_failure_count = 0
                        print(f"✅ RTSP reconnected")
                    except Exception as e:
                        print(f"❌ Reconnect failed: {e}")
                        break
                
                time.sleep(0.02)
                continue

            self.rtsp_failure_count = 0
            
            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            now = time.time()
            self.last_frame_time = now

            with self.raw_lock:
                self.raw_frame = frame

            with self.writer_lock:
                if self.video_writer and self.video_writer.isOpened():
                    try:
                        self.video_writer.write(frame)
                    except:
                        pass

            # 🔥 FIX 4: Send frame for recognition at controlled interval (FIXED QUEUE)
            if (now - last_recognition_send) >= self.recognition_interval_sec:
                with self.recognition_lock:
                    # 🔥 Do NOT clear queue (prevents freeze)
                    if len(self.recognition_queue) < self.recognition_queue.maxlen:
                        self.recognition_queue.append(frame.copy())
                last_recognition_send = now

            self.frame_count += 1
            frames_since_calc += 1

            if (now - last_fps_calc) >= 2.0:
                elapsed = now - last_fps_calc
                self.fps = frames_since_calc / elapsed if elapsed > 0 else 0.0
                frames_since_calc = 0
                last_fps_calc = now

            time.sleep(0.001)

        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        self.cap = None

        print(f"🧹 RTSP thread stopped: {self.camera_name}")

    def _stream_render_loop(self):
        """THREAD 2: Stream rendering"""
        print(f"🎨 Stream render thread started")
        
        frame_time = 1.0 / self.stream_fps
        last_render = 0

        while self.running and not self.stop_event.is_set():
            now = time.time()
            
            if (now - last_render) < frame_time:
                time.sleep(0.005)
                continue

            with self.raw_lock:
                if self.raw_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.raw_frame.copy()

            # Show warning if stream is frozen
            if (now - self.last_frame_time) > 3.0:
                cv2.putText(frame, "RECONNECTING...", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            annotated = self._draw_tracks(frame)
            stream_frame = self._resize_for_stream(annotated)

            with self.stream_lock:
                self.stream_frame = stream_frame

            last_render = now

        print(f"🧹 Stream thread stopped: {self.camera_name}")

    def _recognition_loop(self):
        """THREAD 3: Face recognition (ArcFace)"""
        print(f"🧠 ArcFace recognition thread started")
        
        recognition_count_fps = 0
        last_fps_calc = time.time()

        while self.running and not self.stop_event.is_set():
            frame_to_process = None
            with self.recognition_lock:
                if self.recognition_queue:
                    frame_to_process = self.recognition_queue.popleft()

            if frame_to_process is None:
                time.sleep(0.05)
                continue

            self._process_faces_arcface(frame_to_process)
            
            self.recognition_count += 1
            recognition_count_fps += 1

            now = time.time()
            if (now - last_fps_calc) >= 2.0:
                elapsed = now - last_fps_calc
                self.recognition_fps = recognition_count_fps / elapsed
                recognition_count_fps = 0
                last_fps_calc = now

        print(f"🧹 Recognition thread stopped: {self.camera_name}")

    def _adaptive_control_loop(self):
        """THREAD 4: Adaptive control for ArcFace"""
        print(f"🎮 Adaptive control thread started")
        
        check_interval = 10
        
        while self.running and not self.stop_event.is_set():
            time.sleep(check_interval)
            
            try:
                fps = self.fps
                
                # Adjust recognition interval based on FPS
                if fps < 10:
                    self.recognition_interval_sec = min(1.2, self.recognition_interval_sec + 0.1)
                    print(f"⚡ Adaptive: FPS low ({fps:.1f}) → Increased interval to {self.recognition_interval_sec:.2f}s")
                
                elif fps > 20 and self.recognition_interval_sec > 0.5:
                    self.recognition_interval_sec = max(0.5, self.recognition_interval_sec - 0.05)
                    print(f"⚡ Adaptive: FPS good ({fps:.1f}) → Decreased interval to {self.recognition_interval_sec:.2f}s")
                
                # Save updated config
                try:
                    with open("optimal_config.json", "r") as f:
                        config = json.load(f)
                    config["recognition_interval"] = self.recognition_interval_sec
                    with open("optimal_config.json", "w") as f:
                        json.dump(config, f, indent=4)
                except:
                    pass
                
            except Exception as e:
                print(f"❌ Adaptive control error: {e}")
        
        print(f"🧹 Adaptive control thread stopped: {self.camera_name}")

    def _process_faces_arcface(self, frame: np.ndarray):
        """Face recognition using ArcFace"""
        try:
            # Resize for faster processing
            small = cv2.resize(
                frame, (0, 0),
                fx=self.downscale_factor,
                fy=self.downscale_factor,
                interpolation=cv2.INTER_AREA
            )
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # 🔥 ARCFACE RECOGNITION
            faces = face_store.recognize_faces(
                rgb_small,
                tolerance=self.tolerance
            )

            if not faces:
                return

            scale = 1.0 / self.downscale_factor
            current_time = time.time()

            new_detections = []
            for face in faces:
                name = face["name"]
                top, right, bottom, left = face["location"]
                is_known = face["is_known"]
                conf = float(face.get("confidence", 0.0))
                
                # 🔒 FIX 1: STRICT CONFIDENCE FILTER (prevents ear / side-face detection)
                if conf < 0.55:
                    continue

                # Scale bounding box back to original size
                face_bbox = (
                    int(left * scale),
                    int(top * scale),
                    int(right * scale),
                    int(bottom * scale)
                )
                
                # 🔒 FIX 2: Minimum face size for CCTV (prevents false captures)
                fw = face_bbox[2] - face_bbox[0]
                fh = face_bbox[3] - face_bbox[1]
                
                if fw < 70 or fh < 70:  # Increased from 45
                    continue
                
                # Basic aspect ratio check
                if fw > 0 and fh > 0:
                    aspect_ratio = fw / float(fh)
                    if aspect_ratio < 0.5 or aspect_ratio > 1.8:
                        continue

                canonical_bbox = self._expand_bbox_for_capture(face_bbox, frame.shape)
                
                new_detections.append({
                    'name': name,
                    'bbox': canonical_bbox,
                    'is_known': is_known,
                    'confidence': conf,
                    'time': current_time
                })
                
                self.total_recognition_count += 1
                if is_known and name != "Unknown":
                    self.correct_recognition_count += 1
                elif not is_known:
                    self.genuine_unknown_count += 1

            self._update_tracks(new_detections, frame, current_time)

        except Exception as e:
            print(f"❌ ArcFace recognition error: {e}")

    def _expand_bbox_for_capture(self, face_bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        left, top, right, bottom = face_bbox
        h, w = frame_shape[:2]

        face_w = right - left
        face_h = bottom - top

        # 🔥 FIX 3: Expand bbox to include full head + neck
        expand_top = int(face_h * 0.7)      # Increased from 0.5
        expand_bottom = int(face_h * 0.9)   # Increased from 0.8
        expand_side = int(face_w * 0.45)    # Increased from 0.4

        new_left = max(0, left - expand_side)
        new_right = min(w, right + expand_side)
        new_top = max(0, top - expand_top)
        new_bottom = min(h, bottom + expand_bottom)

        return new_left, new_top, new_right, new_bottom

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _update_tracks(self, detections: List[Dict], frame: np.ndarray, current_time: float):
        """Track and capture faces"""
        
        # Remove old tracks
        to_remove = []
        for track_id, track in self.active_tracks.items():
            if (current_time - track['last_seen']) > self.track_timeout:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.active_tracks[track_id]

        matched_tracks = set()
        matched_detections = set()

        # Match detections to existing tracks
        for det_idx, detection in enumerate(detections):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._calculate_iou(detection['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_iou > self.iou_threshold and best_track_id is not None:
                track = self.active_tracks[best_track_id]
                
                if detection['is_known']:
                    if track['name'] == 'Unknown' or detection['confidence'] > track['confidence']:
                        track['name'] = detection['name']
                        track['confidence'] = detection['confidence']
                        track['is_known'] = True
                    track['last_recognized'] = current_time
                elif track['name'] == 'Unknown':
                    track['name'] = detection['name']
                    track['is_known'] = False
                
                track['bbox'] = detection['bbox']
                track['last_seen'] = current_time
                
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
                
            else:
                # Create new track
                self.track_id_counter += 1
                track_id = f"track_{self.track_id_counter}"
                
                self.active_tracks[track_id] = {
                    'name': detection['name'],
                    'bbox': detection['bbox'],
                    'is_known': detection['is_known'],
                    'confidence': detection['confidence'],
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_recognized': current_time if detection['is_known'] else None,
                    'logged': False
                }
                
                matched_detections.add(det_idx)

        # Process logging for mature tracks
        for track_id, track in self.active_tracks.items():
            if track.get("logged"):
                continue

            track_age = current_time - track.get("first_seen", current_time)
            if track_age < self.min_track_age_for_logging:
                continue

            name = track["name"]
            is_known = track["is_known"]
            confidence = track.get("confidence", 0.0)

            if is_known:
                last_recognized = track.get("last_recognized")
                if not last_recognized or (current_time - last_recognized) > 1.0:
                    track["logged"] = True
                    continue

            now_dt = datetime.now()
            capture_key = name if is_known else f"Unknown_{self.camera_name}"
            last_cap = self.capture_log.get(capture_key)
            
            if last_cap and (now_dt - last_cap).total_seconds() < self.capture_cooldown_seconds:
                track["logged"] = True
                continue

            left, top, right, bottom = track["bbox"]
            # Ensure bounding box is valid
            if right <= left or bottom <= top:
                track["logged"] = True
                continue
                
            face_crop = frame[top:bottom, left:right]
            
            # 🔒 FIX 5: Reject invalid crops
            if face_crop is None or face_crop.size == 0:
                track["logged"] = True
                continue
                
            if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                track["logged"] = True
                continue

            # 🔥 Save image and get clickable URL
            image_url = self._save_captured_face(name, face_crop, is_known=is_known)
            self.capture_log[capture_key] = now_dt
            
            if is_known:
                print(f"📸 Captured KNOWN: {name} ({confidence:.2f})")
                
                # Log attendance
                last_seen = self.attendance_log.get(name)
                if not last_seen or (now_dt - last_seen).total_seconds() >= self.cooldown_seconds:
                    self.attendance_log[name] = now_dt
                    self._log_attendance(name, image_url)
            else:
                print(f"📸 Captured UNKNOWN: {name}")

            track["logged"] = True

    def _draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and labels"""
        current_time = time.time()
        
        for track_id, track in list(self.active_tracks.items()):
            if (current_time - track['last_seen']) > self.track_timeout:
                continue
            
            name = track['name']
            is_known = track['is_known']
            confidence = track.get('confidence', 0.0)
            left, top, right, bottom = track['bbox']
            
            # Draw bounding box
            color = (0, 255, 0) if is_known else (0, 0, 255)
            thickness = 2
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Prepare label
            if is_known:
                label = f"{name} ({confidence:.2f})"
            else:
                label = "Unknown"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            
            # Position label above or below box
            y = top - 10
            if y - th - 10 < 0:
                y = bottom + th + 10
            
            # Draw label background
            cv2.rectangle(frame, 
                         (left, y - th - 8), 
                         (left + tw + 8, y + 2), 
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (left + 4, y - 4), 
                       font, scale, (255, 255, 255), thickness)
        
        # Add watermark
        cv2.putText(frame, "ArcFace", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add FPS counter
        if self.fps > 0:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

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
        self.local_video_path = os.path.join(self.video_base_dir, self.camera_name, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.local_video_path, fourcc, fps, (width, height))

        if self.video_writer.isOpened():
            print(f"🎬 Recording: {filename}")
        else:
            print("❌ Failed to start recording")
            self.video_writer = None

    def _save_captured_face(self, name: str, face_image: np.ndarray, is_known: bool) -> str:
        """
        Saves image to backend folder AND returns clickable HTTP URL
        """
        safe_name = name.replace("/", "_").replace("\\", "_").strip() or "Unknown"

        # Save to local folder
        base = self.captured_known_dir if is_known else self.captured_unknown_dir
        person_dir = os.path.join(base, safe_name if is_known else "Unknown")
        os.makedirs(person_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.camera_name}_{safe_name}_{ts}.jpg"
        full_path = os.path.join(person_dir, filename)

        # Save the image
        try:
            cv2.imwrite(full_path, face_image)
        except Exception as e:
            print(f"❌ Failed to save image: {e}")
            return ""

        # Create relative path from "Captured Images" folder
        relative_path = os.path.relpath(full_path, "Captured Images")
        relative_path = relative_path.replace(os.sep, "/")
        
        # URL-encode spaces (CRITICAL!)
        encoded_path = urllib.parse.quote(relative_path)
        
        # 🔥 Final clickable link for Google Sheets
        clickable_url = f"{self.base_url}/media/{encoded_path}"
        
        return clickable_url

    def _log_attendance(self, name: str, image_url: str):
        """Log to Google Sheets with clickable URL"""
        try:
            if google_api.is_authenticated():
                success = google_api.log_attendance(self.camera_name, name, image_url)
                if success:
                    print(f"✅ Logged to Google Sheets: {name}")
                    print(f"   🔗 Image: {image_url}")
                else:
                    print(f"❌ Failed to log to Google Sheets for {name}")
            else:
                print(f"⚠️ Google Sheets not authenticated")
                
        except Exception as e:
            print(f"❌ Logging error: {e}")