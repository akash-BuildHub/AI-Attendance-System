import React, { useState, useEffect } from "react";

const VideoPreview = ({ camera, isLive, apiBase }) => {
  const [streamUrl, setStreamUrl] = useState(null);
  const [error, setError] = useState(null);

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
    } else {
      setStreamUrl(null);
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
          <img
            src={streamUrl}
            alt="camera-live"
            style={{ 
              width: "100%", 
              height: "100%", 
              objectFit: "cover",
              backgroundColor: "#000"
            }}
            onError={(e) => {
              console.error("Error loading stream:", e);
              setError("Unable to load video stream");
            }}
          />
        ) : camera ? (
          <div style={{ color: "#777" }}>
            {error || "Click Test to check camera connection"}
          </div>
        ) : (
          <div style={{ color: "#777" }}>
            Select a camera to preview live feed
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoPreview;