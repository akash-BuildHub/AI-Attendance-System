from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import time
import cv2
import urllib.parse

from app.services.session_manager import manager
from app.services.training_service import load_prototypes

router = APIRouter(prefix="/camera", tags=["camera"])

@router.post("/start")
def start_camera(payload: dict):
    camera_id = int(payload.get("cameraId", 0))
    camera_name = (payload.get("cameraName") or f"Camera_{camera_id}").strip()
    rtsp_url = payload.get("rtspUrl", "")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtspUrl required")

    rtsp_url = urllib.parse.unquote(rtsp_url)

    # Must have trained prototypes
    if not load_prototypes():
        raise HTTPException(status_code=400, detail="No trained data. Train first using POST /train")

    sess = manager.start(camera_id, camera_name, rtsp_url)
    return {"success": True, "message": "started", "data": {"cameraId": camera_id, "cameraName": sess.camera_name}}

@router.post("/stop")
def stop_camera(payload: dict):
    camera_id = int(payload.get("cameraId", 0))
    vp = manager.stop(camera_id)
    return {"success": True, "message": "stopped", "video_path": vp}

@router.get("/stream/{camera_id}")
def stream(camera_id: int):
    sess = manager.get(camera_id)
    if not sess:
        raise HTTPException(status_code=404, detail="camera not running")

    def gen():
        while True:
            sess = manager.get(camera_id)
            if not sess or not sess.running:
                break

            # process AI + update annotated frame
            manager.process_once(camera_id)

            frame = sess.latest_annotated
            if frame is None:
                time.sleep(0.03)
                continue

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
