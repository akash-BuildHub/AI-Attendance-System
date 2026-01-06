from fastapi import APIRouter
from app.integrations.google_api import google_api

router = APIRouter(prefix="/attendance", tags=["attendance"])

@router.get("/records")
def records(limit: int = 100):
    if not google_api.is_authenticated():
        return {"success": False, "message": "Google Sheets not authenticated", "data": []}
    data = google_api.get_recent_attendance(limit)
    return {"success": True, "data": data}
