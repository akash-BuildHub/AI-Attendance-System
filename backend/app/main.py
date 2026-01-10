from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Grow AI Backend",
    description="AI-powered CCTV Attendance System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from app.routes import camera, attendance, media, stream, training

# Register routers
app.include_router(camera.router)
app.include_router(stream.router)
app.include_router(attendance.router)
app.include_router(media.router)
app.include_router(training.router)


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "message": "Grow AI Backend running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    """Detailed health check"""
    from app.integrations.google_api import google_api
    
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "google_sheets": google_api.is_authenticated()
        }
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting Grow AI Backend...")
    print("📡 API available at: http://localhost:8000")
    print("📚 Docs available at: http://localhost:8000/docs")
    
    # Initialize Google Sheets connection
    from app.integrations.google_api import google_api
    google_api.authenticate()


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down Grow AI Backend...")
    from app.services.session_manager import manager
    
    # Stop all running cameras
    for camera_id in list(manager.sessions.keys()):
        manager.stop_camera(camera_id)
    
    print("✅ All cameras stopped")