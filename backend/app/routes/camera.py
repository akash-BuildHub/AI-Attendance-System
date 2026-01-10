from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import cv2
import uuid

from app.services.session_manager import manager

router = APIRouter(prefix="", tags=["Camera"])  # ✅ Empty prefix for root-level endpoints


# ----------------------------
# Request Models
# ----------------------------

class CameraConnectRequest(BaseModel):
    camera_name: str
    ip: str
    username: str
    password: str
    rtsp_path: str = "/stream1"
    port: int = 554


class CameraTestRequest(BaseModel):
    """Model for /test-camera endpoint (used by frontend)"""
    rtspUrl: str


class StartFeedRequest(BaseModel):
    """Model for /start-feed endpoint"""
    cameraId: int
    cameraName: str
    rtspUrl: str


class StopFeedRequest(BaseModel):
    """Model for /stop-feed endpoint"""
    cameraId: int


# ----------------------------
# Helpers
# ----------------------------

def build_rtsp(req: CameraConnectRequest) -> str:
    """Build RTSP URL from camera credentials"""
    return f"rtsp://{req.username}:{req.password}@{req.ip}:{req.port}{req.rtsp_path}"


def test_rtsp(rtsp_url: str):
    """Test RTSP connection and raise error if fails"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("RTSP connection failed - cannot open stream")
    
    # Try to read one frame to ensure stream is actually working
    ret, _ = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("RTSP connection failed - cannot read frames")


# ----------------------------
# APIs - Frontend Integration
# ----------------------------

@router.post("/test-camera")
def test_camera(req: CameraTestRequest):
    """
    ✅ NEW ENDPOINT - Test RTSP camera connection (called by frontend)
    This matches what CameraForm.jsx expects
    
    Flow:
    1. Frontend sends RTSP URL
    2. Backend validates connection
    3. Returns success + temporary streamId
    4. Frontend uses streamId to load /stream/{id}
    """
    rtsp_url = req.rtspUrl

    if not rtsp_url:
        return {
            "success": False,
            "message": "RTSP URL is required"
        }

    # Validate RTSP URL format
    if not rtsp_url.lower().startswith("rtsp://"):
        return {
            "success": False,
            "message": "Invalid RTSP URL format. Must start with rtsp://"
        }

    # Test actual RTSP connection
    try:
        test_rtsp(rtsp_url)
    except Exception as e:
        return {
            "success": False,
            "message": f"Connection failed: {str(e)}"
        }

    # Generate temporary stream ID for preview
    stream_id = str(uuid.uuid4())

    # Start a temporary preview session
    try:
        # Generate a temporary camera ID (negative to distinguish from saved cameras)
        temp_camera_id = -abs(hash(stream_id)) % 100000
        
        manager.start_camera(
            camera_id=temp_camera_id,
            camera_name="Test Preview",
            rtsp_url=rtsp_url
        )
        
        return {
            "success": True,
            "message": "Camera connected successfully",
            "streamId": temp_camera_id  # ✅ Frontend needs this ID
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to start preview: {str(e)}"
        }


@router.post("/start-feed")
def start_feed(req: StartFeedRequest):
    """
    ✅ Start camera feed for saved camera
    Called when user clicks "Start" on a saved camera
    """
    try:
        # Stop any existing session with this ID
        manager.stop_camera(req.cameraId)
        
        # Start new session
        manager.start_camera(
            camera_id=req.cameraId,
            camera_name=req.cameraName,
            rtsp_url=req.rtspUrl
        )
        
        return {
            "success": True,
            "message": "Camera feed started",
            "streamId": req.cameraId
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to start feed: {str(e)}"
        }


@router.post("/stop-feed")
def stop_feed(req: StopFeedRequest):
    """
    ✅ Stop camera feed
    Called when user clicks "Stop" or switches cameras
    """
    try:
        success = manager.stop_camera(req.cameraId)
        
        return {
            "success": success,
            "message": "Camera feed stopped" if success else "Camera not found"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to stop feed: {str(e)}"
        }


# ----------------------------
# APIs - Original Camera Management
# ----------------------------

@router.post("/camera/connect")
def connect_camera(req: CameraConnectRequest):
    """
    Connect and start CCTV camera (original endpoint)
    This endpoint:
    1. Builds RTSP URL
    2. Tests connection
    3. Starts camera session
    4. Returns camera info for streaming
    """
    rtsp_url = build_rtsp(req)

    # 1. Test RTSP connection
    try:
        test_rtsp(rtsp_url)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Camera connection failed: {str(e)}"
        )

    # 2. Generate unique camera ID
    camera_id = max([s.camera_id for s in manager.sessions.values()], default=0) + 1

    # 3. Start camera session
    try:
        manager.start_camera(
            camera_id=camera_id,
            camera_name=req.camera_name,
            rtsp_url=rtsp_url
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start camera session: {str(e)}"
        )

    return {
        "success": True,
        "camera_id": camera_id,
        "camera_name": req.camera_name,
        "rtsp_url": rtsp_url,
        "stream_url": f"/stream/{camera_id}"
    }


@router.post("/camera/stop/{camera_id}")
def stop_camera(camera_id: int):
    """Stop a running camera"""
    success = manager.stop_camera(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return {"success": True, "message": "Camera stopped"}


@router.delete("/camera/{camera_id}")
def delete_camera(camera_id: int):
    """Remove a camera completely"""
    success = manager.remove_camera(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return {"success": True, "message": "Camera removed"}


@router.get("/camera/list")
def list_cameras():
    """List all cameras with their status"""
    return {
        "success": True,
        "data": manager.list_cameras()
    }


@router.get("/camera/{camera_id}/status")
def camera_status(camera_id: int):
    """Get detailed status of a specific camera"""
    sess = manager.sessions.get(camera_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return {
        "success": True,
        "data": {
            "camera_id": sess.camera_id,
            "camera_name": sess.camera_name,
            "rtsp_url": sess.rtsp_url,
            "running": sess.running,
            "stream_url": f"/stream/{camera_id}"
        }
    }