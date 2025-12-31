import cv2
import threading
import time

class CameraManager:
    def __init__(self):
        self.cameras = {}   # {camera_id: VideoCapture object}
        self.frames = {}    # {camera_id: last frame}
        self.threads = {}   # streaming threads
        self.running = {}   # camera running states

    def start_camera(self, camera_id, rtsp_url):
        if camera_id in self.running and self.running[camera_id]:
            return True  # already running
        
        print(f"🔗 Opening RTSP: {rtsp_url}")
        
        # Force FFmpeg backend - NO DECODING
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
        
        # Try to open the camera with a timeout
        max_retries = 3
        for i in range(max_retries):
            if cap.isOpened():
                break
            print(f"Retry {i+1}/{max_retries}...")
            time.sleep(0.5)
            
        if not cap.isOpened():
            print(f"Failed to open camera {camera_id} with URL: {rtsp_url}")
            return False

        self.cameras[camera_id] = cap
        self.running[camera_id] = True
        self.frames[camera_id] = None

        def capture_loop():
            while self.running.get(camera_id, False):
                try:
                    ret, frame = cap.read()
                    if ret:
                        # Resize frame for better performance
                        frame = cv2.resize(frame, (640, 480))
                        self.frames[camera_id] = frame
                    else:
                        # Camera disconnected, stop the thread
                        print(f"Camera {camera_id} disconnected")
                        self.running[camera_id] = False
                        break
                except Exception as e:
                    print(f"Error reading frame from camera {camera_id}: {e}")
                    self.running[camera_id] = False
                    break
                time.sleep(0.03)  # ~30 FPS
                
            # Clean up
            if camera_id in self.cameras:
                self.cameras[camera_id].release()
            if camera_id in self.running:
                self.running[camera_id] = False
            if camera_id in self.frames:
                self.frames[camera_id] = None

        thread = threading.Thread(target=capture_loop, daemon=True)
        thread.start()
        self.threads[camera_id] = thread

        return True

    def stop_camera(self, camera_id):
        if camera_id in self.running:
            self.running[camera_id] = False
            
        # Wait a bit for thread to stop
        time.sleep(0.1)
        
        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
            
        if camera_id in self.frames:
            del self.frames[camera_id]
            
        if camera_id in self.threads:
            del self.threads[camera_id]

    def get_frame(self, camera_id):
        return self.frames.get(camera_id, None)

    def is_camera_running(self, camera_id):
        return self.running.get(camera_id, False)