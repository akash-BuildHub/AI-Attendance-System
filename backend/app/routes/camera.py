import time
import cv2
import urllib.parse
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.core.state import state
from app.services import session_manager as session_manager_module

if state.session_manager is None:
    state.session_manager = session_manager_module.manager

router = APIRouter(prefix="/cameras", tags=["cameras"])
legacy_router = APIRouter(prefix="/camera", tags=["camera-legacy"])
compat_router = APIRouter(tags=["camera-compat"])
feed_router = APIRouter(tags=["camera-legacy-feed"])

def _parse_camera_id(value) -> int:
    try:
        camera_id = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="cameraId must be an integer")
    return camera_id

async def _check_camera(rtsp_url: str) -> bool:
    def _capture():
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        return bool(ok and frame is not None)

    return await run_in_threadpool(_capture)

@router.post("/test")
async def test_camera(payload: dict):
    rtsp_url = payload.get("rtspUrl", "")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtspUrl required")

    rtsp_url = urllib.parse.unquote(rtsp_url)
    ok = await _check_camera(rtsp_url)
    if not ok:
        return {"success": False, "message": "Unable to connect to camera"}
    return {"success": True, "message": "Connected successfully"}

@compat_router.post("/test-camera")
async def test_camera_compat(payload: dict):
    rtsp_url = payload.get("rtsp_url") or payload.get("rtspUrl")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url required")
    return await test_camera({"rtspUrl": rtsp_url})

@router.get("")
async def list_cameras():
    return {"success": True, "data": state.session_manager.list_cameras()}

@router.post("")
async def create_camera(payload: dict):
    return await start_camera(payload)

@router.post("/start")
async def start_camera(payload: dict):
    camera_id = _parse_camera_id(payload.get("cameraId", 0))
    camera_name = (payload.get("cameraName") or f"Camera_{camera_id}").strip()
    rtsp_url = payload.get("rtspUrl", "")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtspUrl required")

    rtsp_url = urllib.parse.unquote(rtsp_url)
<<<<<<< ours
<<<<<<< ours
    try:
        sess = await run_in_threadpool(manager.start_camera, camera_id, camera_name, rtsp_url)
    except RuntimeError as exc:
        return {"success": False, "message": str(exc)}
=======
    sess = await run_in_threadpool(state.session_manager.start_camera, camera_id, camera_name, rtsp_url)
>>>>>>> theirs
=======
    sess = await run_in_threadpool(state.session_manager.start_camera, camera_id, camera_name, rtsp_url)
>>>>>>> theirs
    return {"success": True, "message": "started", "data": {"cameraId": camera_id, "cameraName": sess.camera_name}}

@router.post("/stop")
async def stop_camera(payload: dict):
<<<<<<< ours
<<<<<<< ours
    camera_id = _parse_camera_id(payload.get("cameraId", 0))
    vp = await run_in_threadpool(manager.stop_camera, camera_id)
=======
    camera_id = int(payload.get("cameraId", 0))
    vp = await run_in_threadpool(state.session_manager.stop_camera, camera_id)
>>>>>>> theirs
=======
    camera_id = int(payload.get("cameraId", 0))
    vp = await run_in_threadpool(state.session_manager.stop_camera, camera_id)
>>>>>>> theirs
    return {"success": True, "message": "stopped", "video_path": vp}

@router.post("/resume")
async def resume_camera(payload: dict):
<<<<<<< ours
    camera_id = _parse_camera_id(payload.get("cameraId", 0))
    try:
        sess = await run_in_threadpool(manager.resume_camera, camera_id)
    except RuntimeError as exc:
        return {"success": False, "message": str(exc)}
=======
    camera_id = int(payload.get("cameraId", 0))
    sess = await run_in_threadpool(state.session_manager.resume_camera, camera_id)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
    if not sess:
        raise HTTPException(status_code=404, detail="camera not found")
    return {"success": True, "message": "resumed", "data": {"cameraId": camera_id, "cameraName": sess.camera_name}}

@router.get("/stream/{camera_id}")
def stream(camera_id: int):
<<<<<<< ours
<<<<<<< ours
    camera_id = _parse_camera_id(camera_id)
    if not manager.get(camera_id):
        raise HTTPException(status_code=404, detail="camera not running")

    return StreamingResponse(
        manager.frame_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
=======
=======
>>>>>>> theirs
    sess = state.session_manager.get(camera_id)
    if not sess:
        raise HTTPException(status_code=404, detail="camera not running")

    def gen():
        while True:
            current = state.session_manager.get(camera_id)
            if not current or not current.running:
                break

            frame = current.latest_annotated or current.latest_frame
            if frame is None:
                time.sleep(0.03)
                continue

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
>>>>>>> theirs

@router.get("/preview-stream")
def preview_stream(rtspUrl: str):
    if not rtspUrl:
        raise HTTPException(status_code=400, detail="rtspUrl required")

    rtsp_url = urllib.parse.unquote(rtspUrl)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    def gen():
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        finally:
            cap.release()

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@legacy_router.post("/test-camera")
async def test_camera_alias(payload: dict):
    return await test_camera(payload)

@legacy_router.post("/start")
async def start_camera_alias(payload: dict):
    return await start_camera(payload)

@legacy_router.post("/stop")
async def stop_camera_alias(payload: dict):
    return await stop_camera(payload)

@legacy_router.post("/resume")
async def resume_camera_alias(payload: dict):
    return await resume_camera(payload)

@legacy_router.get("/stream/{camera_id}")
def stream_alias(camera_id: int):
    return stream(camera_id)

@compat_router.get("/stream/{camera_id}")
def stream_compat(camera_id: int):
    return stream(camera_id)

@legacy_router.post("/start-feed")
async def start_feed(payload: dict):
    return await start_camera(payload)

@legacy_router.post("/stop-feed")
async def stop_feed(payload: dict):
    return await stop_camera(payload)

<<<<<<< ours
<<<<<<< ours
@feed_router.post("/start-feed")
async def start_feed_root(payload: dict):
    return await start_camera(payload)

@feed_router.post("/stop-feed")
async def stop_feed_root(payload: dict):
    return await stop_camera(payload)

@feed_router.get("/stream/{camera_id}")
def stream_root(camera_id: int):
    return stream(camera_id)
=======
=======
>>>>>>> theirs
@compat_router.post("/start-feed")
async def start_feed_compat(payload: dict):
    return await start_camera(payload)

@compat_router.post("/stop-feed")
async def stop_feed_compat(payload: dict):
    return await stop_camera(payload)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
