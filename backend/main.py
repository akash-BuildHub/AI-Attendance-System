from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import os
from datetime import datetime
from typing import Dict, Optional
import json

from face_store import face_store
from google_api import google_api
from attendance_worker import AttendanceWorker

app = FastAPI(title="Grow AI - Smart Attendance System")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for camera workers
workers: Dict[int, AttendanceWorker] = {}

@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "Grow AI Smart Attendance System",
        "version": "2.0.0",
        "description": "Face recognition attendance with Google Drive/Sheets integration"
    }

@app.get("/system-status")
async def system_status():
    """Get comprehensive system status."""
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
async def train_from_drive():
    """Train face recognition system using images from Google Drive."""
    try:
        result = face_store.train_from_google_drive()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/training-status")
async def get_training_status():
    """Get current training status."""
    return {
        "success": True,
        "data": face_store.get_training_status()
    }

@app.post("/test-camera")
async def test_camera(payload: dict):
    """Test camera connectivity - used by React Test button."""
    rtsp_url = payload.get("rtspUrl", "")
    
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="RTSP URL is required")
    
    print(f"🔍 Testing camera: {rtsp_url}")
    
    # Test with FFmpeg backend
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "message": "Camera not reachable. Check IP, port, username, and password."
            }
        )
    
    # Try to read a frame with timeout
    import time
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
            content={"success": False, "message": "Connected but unable to read video stream"}
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
    """Start face recognition and recording for a camera - React Start button."""
    try:
        camera_id = int(payload.get("cameraId", 0))
        camera_name = payload.get("cameraName", f"Camera_{camera_id}")
        rtsp_url = payload.get("rtspUrl", "")
        
        if not rtsp_url:
            raise HTTPException(status_code=400, detail="RTSP URL is required")
        
        # Check if worker already exists and is running
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
        
        # 🔁 NEW: Auto train from Google Drive before starting
        print(f"🔁 Auto training from Google Drive for {camera_name}...")
        train_result = face_store.train_from_google_drive()
        
        if not train_result.get("success"):
            print(f"⚠️ Training warning: {train_result.get('message')}")
            # Still proceed - camera will use existing encodings
        
        # Create and start new worker
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
                    "training_result": train_result.get("message", "Training completed")
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start camera worker")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start feed: {str(e)}")

@app.post("/stop-feed")
async def stop_feed(payload: dict):
    """Stop face recognition and recording - React Stop button."""
    try:
        camera_id = int(payload.get("cameraId", 0))
        
        if camera_id in workers:
            worker = workers[camera_id]
            worker.stop()
            
            # Remove from workers dict
            del workers[camera_id]
            
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
    """Get status of a specific camera."""
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
    """Get MJPEG stream from a camera with face recognition overlay."""
    if camera_id not in workers:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Camera not found or not running"}
        )
    
    worker = workers[camera_id]
    
    def generate():
        while worker.running:
            frame = worker.get_latest_frame()
            if frame is not None:
                # Compress frame for streaming
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                import time
                time.sleep(0.05)  # Small delay to prevent busy waiting
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/preview-stream")
async def preview_stream(rtspUrl: str):
    """Preview stream without face recognition (for testing)."""
    
    print(f"🔗 Preview stream with URL: {rtspUrl}")
    
    def generate():
        cap = cv2.VideoCapture(rtspUrl, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for preview
                height, width = frame.shape[:2]
                if width > 1280:
                    ratio = 1280 / width
                    new_height = int(height * ratio)
                    frame = cv2.resize(frame, (1280, new_height), interpolation=cv2.INTER_AREA)
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
    """Get attendance records from Google Sheets."""
    try:
        if google_api.is_authenticated():
            records = google_api.get_recent_attendance(limit)
            
            # Filter records
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
            # Fallback to local backup files
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
    """Clean up all resources and stop all workers."""
    def cleanup_task():
        for camera_id, worker in list(workers.items()):
            try:
                print(f"Stopping worker for camera {camera_id}...")
                worker.stop()
            except Exception as e:
                print(f"Error stopping worker {camera_id}: {e}")
        
        workers.clear()
        
        # Clean up temporary directories
        temp_dirs = ["temp_videos"]
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
        "message": "System cleanup started in background",
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
    print("📁 Training from: Google Drive/Image Data/Person Images/")
    print("📊 Logging to: Google Sheets (AI Attendance Log)")
    print("📹 Recording to: Google Drive/AI Attendance Videos/")
    print("🤖 Faces trained:", len(face_store.encodings))
    print("👥 People registered:", len(set(face_store.names)))
    print("🔗 Google connected:", google_api.is_authenticated())
    print("=" * 60)
    print("📡 Starting server on http://0.0.0.0:8000")
    print("🎮 React UI: http://localhost:3000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)