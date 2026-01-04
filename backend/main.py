from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import os
from datetime import datetime
from typing import Dict, Optional
import json
import urllib.parse
import time

from face_store import face_store
from google_api import google_api
from attendance_worker import AttendanceWorker

app = FastAPI(title="Grow AI - Smart Attendance System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workers: Dict[int, AttendanceWorker] = {}

@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "Grow AI Smart Attendance System",
        "version": "2.0.0"
    }

@app.get("/system-status")
async def system_status():
    return {
        "face_recognition": face_store.get_training_status(),
        "google_integration": google_api.get_drive_info(),
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
@app.post("/force-train")
async def train_from_drive():
    try:
        print("🔁 Starting forced training...")
        result = face_store.train_from_google_drive()
        
        if result.get("success"):
            if os.path.exists("face_encodings.pkl"):
                print(f"✅ Training SUCCESS: {result['message']}")
            else:
                result["success"] = False
                result["message"] = "Training failed: face_encodings.pkl not created"
        else:
            print(f"❌ Training FAILED: {result['message']}")
            
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

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "training": face_store.get_training_status(),
        "face_encodings_file": os.path.exists("face_encodings.pkl")
    }

@app.post("/test-camera")
async def test_camera(payload: dict):
    rtsp_url = payload.get("rtspUrl", "")
    
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="RTSP URL is required")
    
    rtsp_url = urllib.parse.unquote(rtsp_url)
    
    print(f"🔍 Testing camera: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "message": "Camera not reachable"
            }
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
            "color_channels": frame.shape[2] if len(frame.shape) > 2 else 1
        }
    }

@app.post("/start-feed")
async def start_feed(payload: dict):
    try:
        camera_id = int(payload.get("cameraId", 0))
        
        # ✅ Accept multiple possible camera name fields from frontend
        camera_name = (
            payload.get("cameraName") or 
            payload.get("camera_name") or 
            payload.get("name") or 
            payload.get("camera") or 
            f"Camera_{camera_id}"
        ).strip()
        
        # ✅ Sanitize camera name for file system and Drive
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
        
        # ✅ Train ONLY if encodings don't exist
        if not os.path.exists("face_encodings.pkl"):
            print("🧠 No encodings found — training from Drive...")
            train_result = face_store.train_from_google_drive()
        else:
            print("✅ Using existing face_encodings.pkl")
            # Reload encodings to ensure fresh state
            if not face_store.encodings:
                face_store.load_encodings()
            train_result = {
                "success": True,
                "message": "Using existing trained encodings"
            }
        
        if not train_result.get("success"):
            print(f"⚠️ Training warning: {train_result.get('message')}")
        
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
                    "training_result": train_result.get("message", "Encodings loaded successfully")
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start camera worker")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start feed: {str(e)}")

@app.post("/stop-feed")
async def stop_feed(payload: dict):
    try:
        camera_id = int(payload.get("cameraId", 0))
        
        if camera_id in workers:
            worker = workers[camera_id]
            worker.stop()
            
            # ✅ Remove AFTER stop is complete
            workers.pop(camera_id, None)
            
            return {
                "success": True,
                "message": f"Stopped face recognition for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "status": "stopped"
                }
            }
        else:
            return {
                "success": False,
                "message": f"No active feed found for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "status": "not_found"
                }
            }
            
    except Exception as e:
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
    
    worker = workers[camera_id]
    
    def generate():
        while worker.running:
            frame = worker.get_latest_frame()
            if frame is None:
                # ✅ Prevent busy loop & reduce CPU usage
                time.sleep(0.03)
                continue
            
            # ✅ Lower JPEG quality for faster streaming
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/preview-stream")
async def preview_stream(rtspUrl: str):
    rtspUrl = urllib.parse.unquote(rtspUrl)
    
    print(f"🔗 Preview stream: {rtspUrl}")
    
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
                
                # ✅ Lower JPEG quality for preview too
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
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
            records = []
            for filename in os.listdir("."):
                if filename.endswith("_attendance_backup.jsonl"):
                    with open(filename, 'r') as f:
                        for line in f:
                            try:
                                record = json.loads(line.strip())
                                
                                if camera and record.get('camera') != camera:
                                    continue
                                if date and record.get('timestamp', '').split('T')[0] != date:
                                    continue
                                
                                records.append(record)
                            except json.JSONDecodeError:
                                continue
            
            records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return {
                "success": True,
                "data": {
                    "records": records[:limit],
                    "count": len(records[:limit]),
                    "source": "local_backup"
                }
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get records: {str(e)}")

@app.post("/cleanup")
async def cleanup(background_tasks: BackgroundTasks):
    def cleanup_task():
        for camera_id, worker in list(workers.items()):
            try:
                print(f"Stopping worker for camera {camera_id}...")
                worker.stop()
            except Exception as e:
                print(f"Error stopping worker {camera_id}: {e}")
        
        workers.clear()
        
        temp_dirs = ["temp_faces", "temp_images", "temp_videos"]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                print(f"Cleaning up {temp_dir}...")
                for filename in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, filename))
                    except:
                        pass
        
        print("✅ System cleanup completed")
    
    background_tasks.add_task(cleanup_task)
    
    return {
        "success": True,
        "message": "System cleanup started",
        "data": {
            "workers_stopped": len(workers),
            "status": "cleaning"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 GROW AI - Smart Attendance System")
    print("=" * 60)
    print("📁 Training from: Google Drive/Person Images/")
    print("📊 Logging to: Google Sheets (AI Attendance Log)")
    print("📹 Recording to: Google Drive/AI Attendance Videos/")
    print("📸 Face images to: Google Drive/Captured Images/")
    
    # Load encodings at startup
    if os.path.exists("face_encodings.pkl"):
        face_store.load_encodings()
        print(f"✅ Encodings loaded: {len(face_store.encodings)} faces trained")
        print(f"👥 People registered: {len(set(face_store.names))}")
    else:
        print("⚠️  No encodings found. Train with /train-from-drive")
    
    print(f"🔗 Google connected: {google_api.is_authenticated()}")
    print("=" * 60)
    print("📡 Starting server on http://0.0.0.0:8000")
    print("🎮 React UI: http://localhost:3000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)