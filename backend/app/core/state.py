from threading import Lock

class AppState:
    def __init__(self):
        self.camera_running = False
        self.current_camera_name = None
        self.camera_lock = Lock()

        self.face_detection_enabled = True
        self.attendance_tracking = True

        self.session_manager = None

app_state = AppState()
state = app_state
