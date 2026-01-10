from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services.session_manager import manager

router = APIRouter(prefix="/stream", tags=["Stream"])


@router.get("/{camera_id}")
def stream_camera(camera_id: int):
    """
    Stream live video from camera as MJPEG
    Returns multipart/x-mixed-replace stream
    """
    # Verify camera exists
    if camera_id not in manager.sessions:
        raise HTTPException(
            status_code=404, 
            detail=f"Camera {camera_id} not found"
        )
    
    # Get frame generator
    try:
        gen = manager.frame_generator(camera_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start stream: {str(e)}"
        )
    
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame"
    )