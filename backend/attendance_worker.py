import cv2
import time
import json
import threading
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import os

from face_store import face_store
from google_api import google_api

class AttendanceWorker:
    """
    Worker for each camera that:
    1. Captures RTSP feed
    2. Detects and recognizes faces
    3. Logs attendance to Google Sheet
    4. Uploads captured face images to correct Drive locations
    5. Records video and uploads to Drive when stopped
    6. Provides smooth live stream
    """
    
    def __init__(self, camera_id: int, camera_name: str, rtsp_url: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        
        # Worker state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        
        # Video recording
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.temp_videos_dir = "temp_videos"
        os.makedirs(self.temp_videos_dir, exist_ok=True)
        self.local_video_path: Optional[str] = None
        
        # Attendance tracking
        self.attendance_log: Dict[str, datetime] = {}  # person_name -> last_seen_time
        self.cooldown_seconds = 30  # 30 seconds between logging same person
        
        # Face recognition settings
        self.min_confidence = 0.6
        
        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        self.faces_detected = 0
        
        # Stream optimization
        self.stream_width = 1280
        self.stream_every_n_frames = 3  # Process every 3rd frame for stream
        self._stream_counter = 0
        
        print(f"🎥 Initialized AttendanceWorker for {camera_name}")
    
    def start(self):
        """Start the worker thread."""
        if self.running:
            print(f"⚠️ Worker for {self.camera_name} is already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        print(f"🚀 Started AttendanceWorker for {self.camera_name}")
        return True
    
    def stop(self):
        """Stop the worker thread and upload video."""
        if not self.running:
            return
        
        print(f"🛑 Stopping AttendanceWorker for {self.camera_name}...")
        self.running = False
        
        # 🔁 NEW: Wait for thread to finish gracefully
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        # 🔁 NEW: Ensure video writer is released before upload
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
        
        # Upload final video segment
        self._upload_video()
        
        # 🔁 NEW: Clear attendance log for this session
        self.attendance_log.clear()
        
        print(f"✅ Stopped AttendanceWorker for {self.camera_name}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame for streaming."""
        return self.latest_frame
    
    def get_status(self) -> Dict[str, any]:
        """Get worker status."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "running": self.running,
            "camera_name": self.camera_name,
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count,
            "faces_detected": self.faces_detected,
            "people_tracked": len(self.attendance_log),
            "uptime_seconds": round(elapsed, 1) if elapsed else 0
        }
    
    # ========== Main Worker Loop ==========
    
    def _run(self):
        """Main worker loop with reconnection logic."""
        print(f"🔗 Opening RTSP: {self.rtsp_url}")
        
        # Open video capture with FFmpeg backend
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera: {self.camera_name}")
            self.running = False
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        
        print(f"📹 Camera {self.camera_name}: {width}x{height} @ {fps} FPS")
        
        # Start video recording
        self._start_video_recording(width, height, fps)
        
        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.faces_detected = 0
        
        # Reconnection counters
        consecutive_failures = 0
        max_failures = 10
        
        # Main processing loop
        while self.running:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"⚠️ Frame read failed for {self.camera_name} (attempt {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print(f"🔄 Reconnecting to {self.camera_name}...")
                    cap.release()
                    time.sleep(1)
                    
                    cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_failures = 0
                    
                    if not cap.isOpened():
                        print(f"❌ Reconnection failed for {self.camera_name}")
                        break
                
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            
            # Process frame for face recognition (every 5th frame for performance)
            if self.frame_count % 5 == 0:
                self._process_frame_for_attendance(frame)
            
            # Update stream frame (optimized for bandwidth)
            self._stream_counter += 1
            if self._stream_counter % self.stream_every_n_frames == 0:
                stream_frame = self._resize_for_stream(frame)
                self.latest_frame = stream_frame
            
            # Record frame to video
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.write(frame)
            
            # Update performance metrics
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
        
        # Cleanup
        cap.release()
        if self.video_writer:
            self.video_writer.release()
        
        print(f"🧹 Cleaned up resources for {self.camera_name}")
    
    def _resize_for_stream(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for efficient streaming."""
        if frame.shape[1] <= self.stream_width:
            return frame
        
        ratio = self.stream_width / frame.shape[1]
        new_height = int(frame.shape[0] * ratio)
        return cv2.resize(frame, (self.stream_width, new_height), interpolation=cv2.INTER_AREA)
    
    # ========== Video Recording ==========
    
    def _start_video_recording(self, width: int, height: int, fps: int):
        """Start recording a new video file."""
        if self.video_writer:
            self.video_writer.release()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.camera_name}_{timestamp}.mp4"
        self.local_video_path = os.path.join(self.temp_videos_dir, filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.local_video_path, 
            fourcc, 
            fps, 
            (width, height)
        )
        
        if self.video_writer.isOpened():
            print(f"🎬 Started video recording: {filename}")
        else:
            print(f"❌ Failed to start video recording")
            self.video_writer = None
    
    def _upload_video(self):
        """Upload recorded video to Google Drive."""
        if not self.local_video_path or not os.path.exists(self.local_video_path):
            print(f"⚠️ No video to upload for {self.camera_name}")
            return
        
        file_size = os.path.getsize(self.local_video_path)
        if file_size < 1024:  # Less than 1KB
            print(f"⚠️ Video file too small ({file_size} bytes), skipping upload")
            return
        
        try:
            if google_api.is_authenticated():
                video_url = google_api.upload_video_segment(self.camera_name, self.local_video_path)
                if video_url:
                    print(f"✅ Uploaded video for {self.camera_name}")
                else:
                    print(f"❌ Failed to upload video for {self.camera_name}")
            
            # Clean up local file
            try:
                os.remove(self.local_video_path)
                print(f"🗑️ Cleaned up local video: {self.local_video_path}")
            except Exception as e:
                print(f"⚠️ Error removing local video: {e}")
            
            self.local_video_path = None
            
        except Exception as e:
            print(f"❌ Error uploading video: {e}")
    
    # ========== Face Recognition & Attendance ==========
    
    def _process_frame_for_attendance(self, frame: np.ndarray):
        """Process frame for face recognition and attendance logging."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_store.recognize_faces(rgb_frame)
            
            for name, (left, top, right, bottom), confidence in results:
                self.faces_detected += 1
                
                if confidence < self.min_confidence:
                    name = "Unknown"
                
                now = datetime.now()
                last_seen = self.attendance_log.get(name)
                
                # Check cooldown period
                if last_seen and (now - last_seen).total_seconds() < self.cooldown_seconds:
                    self._draw_face_info(frame, name, confidence, (left, top, right, bottom))
                    continue
                
                # Update attendance log
                self.attendance_log[name] = now
                
                # Extract face crop
                face_crop = frame[top:bottom, left:right]
                
                # Log attendance (uploads face image and logs to sheet)
                self._log_attendance(name, face_crop, confidence)
                
                # Draw face info on frame
                self._draw_face_info(frame, name, confidence, (left, top, right, bottom))
                
        except Exception as e:
            print(f"❌ Error in face recognition: {e}")
    
    def _log_attendance(self, name: str, face_image: np.ndarray, confidence: float):
        """Log attendance to Google Sheets and upload face image."""
        try:
            # Encode face image
            success, buffer = cv2.imencode('.jpg', face_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                print("⚠️ Failed to encode face image")
                return
            
            image_bytes = buffer.tobytes()
            display_name = name if name != "Unknown" else "Unknown"
            face_image_url = ""
            
            # Upload face image to Google Drive
            if google_api.is_authenticated():
                face_image_url = google_api.upload_face_image(
                    self.camera_name,
                    display_name,
                    image_bytes
                )
            
            # Prepare attendance data
            attendance_data = {
                "camera": self.camera_name,
                "name": display_name,
                "face_image_url": face_image_url,
                "confidence": float(confidence),
                "status": "Recognized" if display_name != "Unknown" else "Unknown Person"
            }
            
            # Log to Google Sheet
            if google_api.is_authenticated():
                google_api.log_attendance(attendance_data)
            else:
                self._log_locally(attendance_data)
            
            print(f"👤 Logged: {display_name} @ {self.camera_name} (conf: {confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Error logging attendance: {e}")
    
    def _draw_face_info(self, frame: np.ndarray, name: str, confidence: float, box: Tuple[int, int, int, int]):
        """Draw face bounding box and information on frame."""
        left, top, right, bottom = box
        
        # Choose color based on recognition result
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Prepare label
        label = f"{name} ({confidence:.2f})"
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_label = max(top - 10, label_size[1] + 5)
        cv2.rectangle(frame, 
                     (left, top_label - label_size[1] - 5),
                     (left + label_size[0], top_label + 5),
                     color, -1)
        
        # Draw label text
        cv2.putText(frame, label,
                   (left, top_label),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _log_locally(self, data: Dict[str, any]):
        """Log attendance locally as JSON backup."""
        try:
            log_file = f"{self.camera_name}_attendance_backup.jsonl"
            log_entry = {
                **data,
                "timestamp": datetime.now().isoformat(),
                "camera_id": self.camera_id
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            print(f"⚠️ Error logging locally: {e}")