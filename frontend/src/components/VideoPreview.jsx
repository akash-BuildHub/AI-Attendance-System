import React, { useState, useEffect } from "react";

const VideoPreview = ({ camera, isLive, apiBase }) => {
  const [streamUrl, setStreamUrl] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (camera && isLive) {
      // For test preview (from CameraForm)
      if (camera.rtspUrl && !camera.id) {
        setStreamUrl(`${apiBase}/preview-stream?rtspUrl=${encodeURIComponent(camera.rtspUrl)}`);
      }
      // For regular camera feed
      else if (camera.id) {
        setStreamUrl(`${apiBase}/stream/${camera.id}`);
      }
      setError(null);
      setIsLoading(true);
    } else {
      setStreamUrl(null);
      setIsLoading(false);
    }
  }, [camera, isLive, apiBase]);

  return (
    <div className="video-preview">
      <div className="video-header">
        <div className="video-title">
          {camera ? camera.name : "No Camera Selected"}
        </div>
        {isLive && <div className="live-badge">LIVE</div>}
      </div>

      <div className="video-placeholder">
        {streamUrl ? (
          <>
            {isLoading && (
              <div style={{
                position: 'absolute',
                color: '#fff',
                zIndex: 1,
                padding: '10px',
                background: 'rgba(0,0,0,0.5)',
                borderRadius: '4px'
              }}>
                Loading stream...
              </div>
            )}
            <img
              src={streamUrl}
              alt="camera-live"
              style={{ 
                width: "100%", 
                height: "100%", 
                objectFit: "cover",
                backgroundColor: "#000"
              }}
              onLoad={() => setIsLoading(false)}
              onError={(e) => {
                console.error("Error loading stream:", e);
                setError("Unable to load video stream");
                setIsLoading(false);
              }}
            />
          </>
        ) : camera ? (
          <div style={{ color: "#777", padding: "20px", textAlign: "center" }}>
            {error || "Click 'Test' to check camera connection"}
            {isLive && !streamUrl && (
              <div style={{ fontSize: "12px", marginTop: "8px" }}>
                Starting face recognition feed...
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: "#777", padding: "20px", textAlign: "center" }}>
            Select a camera to preview live feed
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoPreview;