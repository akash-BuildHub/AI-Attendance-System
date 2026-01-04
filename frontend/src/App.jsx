import React, { useState } from "react";
import LoginScreen from "./LoginScreen";
import Header from "./components/Header";
import AboutModal from "./components/AboutModal";
import CameraForm from "./components/CameraForm";
import CameraList from "./components/CameraList";
import VideoPreview from "./components/VideoPreview";
import "./App.css";

const API_BASE = "http://localhost:8000";

// Create a separate component for the authenticated app
const AuthenticatedApp = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [showAboutModal, setShowAboutModal] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingCamera, setEditingCamera] = useState(null);
  const [previewCamera, setPreviewCamera] = useState(null);
  const [isPreviewLive, setIsPreviewLive] = useState(false);
  const [activeStreamId, setActiveStreamId] = useState(null);

  const handleOpenAddForm = () => {
    setEditingCamera(null);
    setShowAddForm(true);
  };

  const handleSaveCamera = (formData) => {
    if (editingCamera) {
      // UPDATE existing camera
      setCameras((prev) =>
        prev.map((cam) =>
          cam.id === editingCamera.id
            ? {
                ...cam,
                name: formData.name,
                ip: formData.ip,
                username: formData.username,
                password: formData.password,
                port: formData.port,
                rtspUrl: formData.rtspUrl,
              }
            : cam
        )
      );

      // If currently selected, update preview info too
      if (selectedCamera && selectedCamera.id === editingCamera.id) {
        setSelectedCamera((prev) => ({
          ...prev,
          name: formData.name,
          ip: formData.ip,
          username: formData.username,
          password: formData.password,
          port: formData.port,
          rtspUrl: formData.rtspUrl,
        }));
      }
    } else {
      // ADD new camera
      const newCamera = {
        ...formData,
        id: Date.now(),
        isLive: false,
      };
      setCameras((prev) => [...prev, newCamera]);
    }

    setEditingCamera(null);
    setShowAddForm(false);
  };

  const handleCancelForm = () => {
    setEditingCamera(null);
    setShowAddForm(false);
  };

  const handleStartFeed = async (camera) => {
    try {
      // ✅ Sanitize camera name for file system and Drive
      const safeCameraName = camera.name.replace(/[\/\\]/g, "_").trim();

      // Stop any existing preview
      if (activeStreamId) {
        await fetch(`${API_BASE}/stop-feed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cameraId: activeStreamId }),
        });
      }

      // Start the selected camera
      const response = await fetch(`${API_BASE}/start-feed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cameraId: camera.id,
          cameraName: safeCameraName, // ✅ CRITICAL: Added camera name
          rtspUrl: camera.rtspUrl,
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        setCameras((prev) =>
          prev.map((cam) => ({
            ...cam,
            isLive: cam.id === camera.id,
          }))
        );
        setSelectedCamera({ ...camera, isLive: true });
        setActiveStreamId(camera.id);
        setPreviewCamera(camera);
        setIsPreviewLive(true);
      } else {
        alert("Failed to start camera feed: " + (data.message || "Unknown error"));
      }
    } catch (error) {
      console.error("Error starting feed:", error);
      alert("Error starting camera feed: " + error.message);
    }
  };

  const handleStopFeed = async (cameraId) => {
    try {
      const response = await fetch(`${API_BASE}/stop-feed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cameraId }),
      });

      const data = await response.json();
      
      if (data.success) {
        setCameras((prev) =>
          prev.map((cam) =>
            cam.id === cameraId ? { ...cam, isLive: false } : cam
          )
        );

        if (selectedCamera && selectedCamera.id === cameraId) {
          setSelectedCamera(null);
        }
        
        if (activeStreamId === cameraId) {
          setActiveStreamId(null);
          setPreviewCamera(null);
          setIsPreviewLive(false);
        }
      }
    } catch (error) {
      console.error("Error stopping feed:", error);
    }
  };

  const handleEditCamera = (camera) => {
    // Open right-side Add Camera section with this camera's details
    setEditingCamera(camera);
    setShowAddForm(true);
  };

  const handleDeleteCamera = async (cameraId) => {
    if (!window.confirm("Are you sure you want to delete this camera?")) return;

    try {
      // Stop the feed if it's running
      await fetch(`${API_BASE}/stop-feed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cameraId }),
      });

      setCameras((prev) => prev.filter((cam) => cam.id !== cameraId));

      if (selectedCamera && selectedCamera.id === cameraId) {
        setSelectedCamera(null);
      }
      
      if (activeStreamId === cameraId) {
        setActiveStreamId(null);
        setPreviewCamera(null);
        setIsPreviewLive(false);
      }
    } catch (error) {
      console.error("Error deleting camera:", error);
      setCameras((prev) => prev.filter((cam) => cam.id !== cameraId));
    }
  };

  // New function to handle test success
  const handleTestSuccess = async (cameraConfig, streamId) => {
    try {
      // ✅ Sanitize camera name for file system and Drive
      const safeCameraName = cameraConfig.name.replace(/[\/\\]/g, "_").trim();
      
      const sanitizedConfig = {
        ...cameraConfig,
        name: safeCameraName,
      };

      // Stop any existing preview
      if (activeStreamId) {
        await fetch(`${API_BASE}/stop-feed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cameraId: activeStreamId }),
        });
      }

      // Save camera config that passed validation
      const tempCamera = {
        ...sanitizedConfig,
        id: streamId || Date.now(),
        isLive: true, // ✅ IMPORTANT: Mark as live for consistent state
      };
      
      setPreviewCamera(tempCamera);
      setIsPreviewLive(true);
      setActiveStreamId(tempCamera.id);
    } catch (error) {
      console.error("Error handling test success:", error);
    }
  };

  return (
    <div className="app">
      <Header onAboutClick={() => setShowAboutModal(true)} />

      <AboutModal
        isOpen={showAboutModal}
        onClose={() => setShowAboutModal(false)}
      />

      <div className="main-content">
        {/* LEFT PANEL */}
        <div className="left-panel">
          <div className="section-header">
            <h2>Attached Cameras</h2>
            <div className="cameras-count">{cameras.length} cameras</div>
          </div>

          {cameras.length === 0 ? (
            <div className="empty-state">
              <p>No cameras added. Click "Add Camera" to get started.</p>
            </div>
          ) : (
            <CameraList
              cameras={cameras}
              onStartFeed={handleStartFeed}
              onStopFeed={handleStopFeed}
              onEdit={handleEditCamera}
              onDelete={handleDeleteCamera}
            />
          )}
        </div>

        {/* CENTER PREVIEW */}
        <div className="middle-panel">
          <VideoPreview
            camera={previewCamera}
            isLive={isPreviewLive}
            apiBase={API_BASE}
          />
        </div>

        {/* RIGHT PANEL - ADD/EDIT CAMERA */}
        <div className="right-panel">
          <div className="add-camera-card">
            <div className="add-camera-header">
              <h2 className="add-camera-title">
                {editingCamera ? "Edit Camera" : "Add Camera"}
              </h2>
              {!showAddForm && (
                <div
                  className="add-camera-trigger-right"
                  onClick={handleOpenAddForm}
                >
                  <div className="add-icon-right">+</div>
                </div>
              )}
            </div>

            {showAddForm && (
              <CameraForm
                onSave={handleSaveCamera}
                onCancel={handleCancelForm}
                initialData={editingCamera}
                onTestSuccess={handleTestSuccess}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Main App component with proper hook ordering
const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Fixed login credentials
  const VALID_USERNAME = "Grow";
  const VALID_PASSWORD = "123123";

  const handleLogin = (username, password) => {
    if (username === VALID_USERNAME && password === VALID_PASSWORD) {
      setIsAuthenticated(true);
    } else {
      alert("Invalid username or password");
    }
  };

  // Return either LoginScreen or AuthenticatedApp - no hooks after this return
  if (!isAuthenticated) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  return <AuthenticatedApp />;
};

export default App;