from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import cv2
import os
import json
import time
import urllib.parse
from typing import Optional
from datetime import datetime

from face_store import face_store
from google_api import google_api
from attendance_worker import AttendanceWorker

app = FastAPI(title="Grow AI - Smart Attendance System (PRODUCTION FINAL)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workers = {}

# 🔥 CRITICAL: Set BASE_URL to your server IP
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
os.environ["BASE_URL"] = BASE_URL

@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "Grow AI Smart Attendance System (Production Final)",
        "version": "9.0 (Google Sheets Links Fixed)",
        "storage_mode": "local",
        "base_url": BASE_URL,
        "note": "Images are stored locally and served via HTTP"
    }

@app.get("/system-status")
async def system_status():
    return {
        "face_recognition": face_store.get_training_status(),
        "google_sheets": {
            "authenticated": google_api.is_authenticated(),
            "sheet_name": "AI Attendance Log"
        },
        "cameras": {
            "total": len(workers),
            "active": len([w for w in workers.values() if w.running]),
            "list": [
                {
                    "id": cam_id,
                    "name": worker.camera_name,
                    "status": "running" if worker.running else "stopped",
                    **worker.get_status()
                }
                for cam_id, worker in workers.items()
            ]
        }
    }

@app.post("/train-from-drive")
async def train_from_drive():
    """Train from LOCAL Person Images folder"""
    try:
        print("=" * 60)
        print("🧠 TRAINING FROM LOCAL PERSON IMAGES")
        print("=" * 60)
        
        result = face_store.train_from_local()
        
        if result.get("success"):
            print(f"✅ Training successful: {result['message']}")
        else:
            print(f"❌ Training failed: {result['message']}")
            
        return result
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/training-status")
async def get_training_status():
    return {
        "success": True,
        "data": face_store.get_training_status()
    }

@app.post("/test-camera")
async def test_camera(payload: dict):
    rtsp_url = payload.get("rtspUrl", "")
    
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="RTSP URL is required")
    
    rtsp_url = urllib.parse.unquote(rtsp_url)
    
    print(f"🔍 Testing camera: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    
    if not cap.isOpened():
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Camera not reachable"}
        )
    
    start_time = time.time()
    timeout = 5
    ret = False
    frame = None
    
    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.1)
    
    cap.release()
    
    if not ret:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Connected but no video stream"}
        )
    
    return {
        "success": True,
        "message": "Connected Successfully",
        "data": {
            "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
            "test_time": datetime.now().strftime("%H:%M:%S")
        }
    }

@app.post("/start-feed")
async def start_feed(payload: dict):
    try:
        camera_id = int(payload.get("cameraId", 0))
        camera_name = (
            payload.get("cameraName") or 
            payload.get("camera_name") or 
            payload.get("name") or 
            f"Camera_{camera_id}"
        ).strip()
        
        camera_name = camera_name.replace("/", "_").replace("\\", "_").strip()
        
        rtsp_url = payload.get("rtspUrl", "")
        
        if not rtsp_url:
            raise HTTPException(status_code=400, detail="RTSP URL is required")
        
        rtsp_url = urllib.parse.unquote(rtsp_url)
        
        if camera_id in workers and workers[camera_id].running:
            return {
                "success": True,
                "message": f"Camera '{camera_name}' is already running",
                "data": {
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "status": "already_running"
                }
            }
        
        if not face_store.encodings:
            print("🧠 No encodings found – training from local Person Images...")
            train_result = face_store.train_from_local()
            if not train_result.get("success"):
                return {
                    "success": False,
                    "message": "Training failed. Please add person images first.",
                    "details": train_result
                }
        else:
            print(f"✅ Using existing {len(face_store.encodings)} face encodings")
            train_result = {
                "success": True,
                "message": "Using existing trained encodings"
            }
        
        worker = AttendanceWorker(camera_id, camera_name, rtsp_url)
        if worker.start():
            workers[camera_id] = worker
            
            return {
                "success": True,
                "message": f"Started face recognition for '{camera_name}'",
                "data": {
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "status": "started",
                    "trained_faces": len(face_store.encodings),
                    "people_registered": len(set(face_store.names))
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start camera worker")
            
    except Exception as e:
        print(f"❌ Failed to start feed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start feed: {str(e)}")

@app.post("/stop-feed")
async def stop_feed(payload: dict):
    """Clean stop"""
    try:
        camera_id = int(payload.get("cameraId", 0))

        worker = workers.get(camera_id)
        if not worker:
            return {
                "success": True, 
                "message": "Already stopped", 
                "data": {"camera_id": camera_id, "status": "not_running"}
            }

        workers.pop(camera_id, None)
        worker.stop()

        return {
            "success": True,
            "message": f"Stopped face recognition for camera {camera_id}",
            "data": {"camera_id": camera_id, "status": "stopped"}
        }

    except Exception as e:
        print(f"❌ Stop error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop feed: {str(e)}")

@app.get("/camera-status/{camera_id}")
async def get_camera_status(camera_id: int):
    if camera_id in workers:
        worker = workers[camera_id]
        return {
            "success": True,
            "data": worker.get_status()
        }
    else:
        return {
            "success": False,
            "message": f"No worker found for camera {camera_id}",
            "data": None
        }

@app.get("/stream/{camera_id}")
async def stream_camera(camera_id: int):
    if camera_id not in workers:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Camera not found"}
        )

    def generate():
        last_frame_ts = time.time()

        while True:
            worker = workers.get(camera_id)

            if worker is None or not worker.running:
                print(f"📺 Stream ended for camera {camera_id}")
                break

            frame = worker.get_latest_frame()

            if frame is None:
                time.sleep(0.03)
                continue

            # 🔥 FIX 7: Improve IMAGE CLARITY in stream
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Increased from 70
            if not ok:
                time.sleep(0.01)
                continue

            last_frame_ts = time.time()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                b"Pragma: no-cache\r\n"
                b"Expires: 0\r\n\r\n"
                + buffer.tobytes() +
                b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

@app.get("/preview-stream")
async def preview_stream(rtspUrl: str):
    rtspUrl = urllib.parse.unquote(rtspUrl)
    
    def generate():
        cap = cv2.VideoCapture(rtspUrl, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                height, width = frame.shape[:2]
                if width > 1280:
                    ratio = 1280 / width
                    new_height = int(height * ratio)
                    frame = cv2.resize(frame, (1280, new_height), interpolation=cv2.INTER_AREA)
                
                # 🔥 FIX 7: Improve IMAGE CLARITY in preview
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Increased from 70
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/attendance-records")
async def get_attendance_records(
    camera: Optional[str] = None,
    date: Optional[str] = None,
    limit: int = 100
):
    try:
        if google_api.is_authenticated():
            records = google_api.get_recent_attendance(limit)
            
            filtered = []
            for record in records:
                if camera and record.get("Camera Name", "") != camera:
                    continue
                if date and record.get("Date", "") != date:
                    continue
                filtered.append(record)
            
            return {
                "success": True,
                "data": {
                    "records": filtered,
                    "count": len(filtered),
                    "source": "google_sheets"
                }
            }
        else:
            return {
                "success": False,
                "message": "Google Sheets not connected",
                "data": []
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get records: {str(e)}")

@app.get("/media/{path:path}")
async def serve_media(path: str):
    """
    🔥🔥🔥 MOST IMPORTANT ENDPOINT 🔥🔥🔥
    
    Serves images from backend/Captured Images/ folder.
    
    This is what makes Google Sheets links work!
    
    Example URL:
    http://192.168.1.20:8000/media/Known%20Images/AKASH/Cam1_AKASH_20250105_123456.jpg
    """
    import mimetypes
    
    # 🔥 Base directory is where images are stored
    base_dir = os.path.abspath("Captured Images")
    
    if not os.path.exists(base_dir):
        raise HTTPException(status_code=404, detail="Captured Images directory not found")
    
    # 🔥 Decode URL-encoded path (spaces %20 → spaces)
    decoded_path = urllib.parse.unquote(path)
    
    # 🔥 Convert forward slashes to OS-specific separators
    decoded_path = decoded_path.replace('/', os.sep)
    
    # 🔥 Construct full path
    requested_path = os.path.abspath(os.path.join(base_dir, decoded_path))
    
    # 🔥 Security: prevent directory traversal
    if not requested_path.startswith(base_dir):
        raise HTTPException(
            status_code=403, 
            detail="Access forbidden: Path outside allowed directory"
        )
    
    # 🔥 Check if file exists
    if not os.path.exists(requested_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # 🔥 Check if it's a file (not directory)
    if not os.path.isfile(requested_path):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(requested_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    
    # Return the image file
    return FileResponse(
        requested_path,
        media_type=mime_type,
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/cleanup")
async def cleanup():
    """Debug endpoint to clean up zombie workers"""
    stopped = 0
    for cam_id, worker in list(workers.items()):
        if not worker.running:
            workers.pop(cam_id, None)
            stopped += 1
    
    return {
        "success": True,
        "message": f"Cleaned up {stopped} zombie workers",
        "remaining_workers": len(workers)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 GROW AI - Smart Attendance System (PRODUCTION FINAL v9.0)")
    print("=" * 60)
    print("📁 Training from: Person Images/ (LOCAL)")
    print("📊 Logging to: Google Sheets (AI Attendance Log)")
    print("📹 Recording to: AI Attendance Videos/ (LOCAL)")
    print("📸 Captures: Captured Images/Known Images + Unknown Images")
    print(f"🔗 Base URL: {BASE_URL}")
    
    if "localhost" in BASE_URL or "127.0.0.1" in BASE_URL:
        print("❌❌❌ WARNING: BASE_URL is localhost! ❌❌❌")
        print("   Google Sheets links WILL NOT work from other devices!")
        print("   FIX: export BASE_URL=http://YOUR_SERVER_IP:8000")
        print("   Example: export BASE_URL=http://192.168.1.20:8000")
        print("   Then restart the server.")
    
    print("✅ FEATURES:")
    print("   • Images saved locally in backend folder")
    print("   • Google Sheets stores CLICKABLE HTTP links")
    print("   • Click opens image directly from backend")
    print("   • NO Google Drive upload")
    print("   • NO duplicate image creation")
    print("   • URL encoding fixes spaces in paths")
    
    # Create required directories
    os.makedirs("Person Images", exist_ok=True)
    os.makedirs("Captured Images/Known Images", exist_ok=True)
    os.makedirs("Captured Images/Unknown Images", exist_ok=True)
    os.makedirs("AI Attendance Videos", exist_ok=True)
    
    # Load encodings
    if os.path.exists("face_encodings.pkl"):
        face_store.load_encodings()
        print(f"✅ Encodings loaded: {len(face_store.encodings)} faces trained")
        print(f"👥 People registered: {len(set(face_store.names))}")
    else:
        print("⚠️ No encodings found. Train with /train-from-drive")
    
    print(f"🔗 Google Sheets: {google_api.is_authenticated()}")
    print("=" * 60)
    print("📡 Starting server on http://0.0.0.0:8000")
    print("=" * 60)
    
    # Manual test: verify the /media endpoint works
    print("\n🧪 QUICK TEST:")
    test_path = "Captured Images/Known Images/Test/test.jpg"
    if os.path.exists(test_path):
        test_url = f"{BASE_URL}/media/{urllib.parse.quote('Known Images/Test/test.jpg')}"
        print(f"   Test URL: {test_url}")
        print("   Open this in browser to verify image serving works")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)