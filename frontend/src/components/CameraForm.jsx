import React, { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const CameraForm = ({ onSave, onCancel, initialData = null, onTestSuccess }) => {
  const [formData, setFormData] = useState({
    name: "",
    ip: "192.168.",
    username: "",
    password: "",
    port: "554", // Default RTSP port
    rtspUrl: "",
  });

  const [testStatus, setTestStatus] = useState(null);
  const [isTesting, setIsTesting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  useEffect(() => {
    if (initialData) {
      setFormData({
        name: initialData.name || "",
        ip: initialData.ip || "192.168.",
        username: initialData.username || "",
        password: initialData.password || "",
        port: initialData.port || "554",
        rtspUrl: initialData.rtspUrl || "",
      });
    } else {
      setFormData({
        name: "",
        ip: "192.168.",
        username: "",
        password: "",
        port: "554",
        rtspUrl: "",
      });
    }
  }, [initialData]);

  const generateRTSPUrl = (data) => {
    const { username, password, ip, port } = data;
    if (username && password && ip && port) {
      // ENCODE username and password for RTSP URL
      const encodedUsername = encodeURIComponent(username);
      const encodedPassword = encodeURIComponent(password);
      return `rtsp://${encodedUsername}:${encodedPassword}@${ip}:${port}/Streaming/Channels/101`;
    }
    return "";
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    const updated = { ...formData, [name]: value };

    if (["username", "password", "ip", "port"].includes(name)) {
      updated.rtspUrl = generateRTSPUrl(updated);
    }

    setFormData(updated);
  };

  const handleRefreshUrl = () => {
    const newUrl = generateRTSPUrl(formData);
    if (newUrl) {
      setFormData({ ...formData, rtspUrl: newUrl });
    }
  };

  const handleTest = async () => {
    const { username, password, rtspUrl, port } = formData;

    if (!username || !password || !rtspUrl) {
      setTestStatus({
        success: false,
        message: "Please enter Username, Password and RTSP URL.",
      });
      return;
    }

    // Validate port
    if (port === "8000") {
      setTestStatus({
        success: false,
        message: "Port 8000 is for the web server. Camera RTSP typically uses port 554.",
      });
      return;
    }

    if (!rtspUrl.toLowerCase().startsWith("rtsp://")) {
      setTestStatus({
        success: false,
        message: "Invalid RTSP URL. It must start with rtsp://",
      });
      return;
    }

    setIsTesting(true);
    setTestStatus({
      success: null,
      message: "Testing connection...",
    });

    try {
      const response = await fetch(`${API_BASE}/test-camera`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rtspUrl: formData.rtspUrl,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setTestStatus({
          success: true,
          message: data.message || "Connected Successfully",
        });

        if (typeof onTestSuccess === "function") {
          onTestSuccess(formData);
        }
      } else {
        setTestStatus({
          success: false,
          message: data.message || "Error: Not Connected",
        });
      }
    } catch (error) {
      setTestStatus({
        success: false,
        message: "Server error: Unable to test camera",
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Final validation
    if (formData.port === "8000") {
      alert("Port 8000 is for the web server. Camera RTSP typically uses port 554.");
      return;
    }
    
    if (!formData.name || !formData.ip || !formData.rtspUrl) {
      alert("Please fill in all required fields");
      return;
    }
    
    onSave(formData);
  };

  return (
    <form className="camera-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label>Camera Name</label>
        <input
          type="text"
          name="name"
          value={formData.name}
          onChange={handleChange}
          placeholder="Enter camera name"
          required
        />
      </div>

      <div className="form-group">
        <label>Camera IP</label>
        <input
          type="text"
          name="ip"
          value={formData.ip}
          onChange={handleChange}
          placeholder="192.168.150.108"
          required
        />
      </div>

      <div className="form-group">
        <label>Username</label>
        <input
          type="text"
          name="username"
          value={formData.username}
          onChange={handleChange}
          required
        />
      </div>

      <div className="form-group">
        <label>Password</label>
        <div className="password-input-wrapper">
          <input
            type={showPassword ? "text" : "password"}
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
          />
          <button
            type="button"
            className="password-toggle"
            onClick={() => setShowPassword((prev) => !prev)}
          >
            {showPassword ? "🔓" : "🔒"}
          </button>
        </div>
      </div>

      <div className="form-group">
        <label>Port</label>
        <input
          type="number"
          name="port"
          value={formData.port}
          onChange={handleChange}
          min="1"
          max="65535"
          placeholder="554"
          required
          style={{ backgroundColor: "#f9f9f9" }}
        />
        <div style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
          Standard RTSP port is 554. Port 8000 is for this web server.
        </div>
        {formData.port === "8000" && (
          <div style={{ color: "orange", fontSize: "12px", marginTop: "4px" }}>
            ⚠️ Warning: Port 8000 is typically for web servers, not cameras.
          </div>
        )}
      </div>

      <div className="form-group">
        <label>RTSP URL</label>
        <div className="url-group">
          <input
            type="text"
            name="rtspUrl"
            value={formData.rtspUrl}
            onChange={handleChange}
            placeholder="rtsp://admin:password@192.168.x.x:554/..."
            required
            readOnly
            style={{ backgroundColor: "#f5f5f5", cursor: "not-allowed" }}
          />
          <button
            type="button"
            className="refresh-btn"
            onClick={handleRefreshUrl}
            title="Generate RTSP URL"
          >
            <svg className="refresh-icon" viewBox="0 0 24 24">
              <path
                fill="currentColor"
                d="M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z"
              />
            </svg>
          </button>
        </div>
      </div>

      <div className={`status-message-area ${testStatus ? 'has-status' : ''}`}>
        {testStatus && (
          <div className="status-message-box">
            {testStatus.success === true ? (
              <span className="status-success">
                <span style={{ fontSize: '16px' }}>✔</span>
                {testStatus.message}
              </span>
            ) : testStatus.success === false ? (
              <span className="status-error">
                <span style={{ fontSize: '16px' }}>✘</span>
                {testStatus.message}
              </span>
            ) : (
              <span>{testStatus.message}</span>
            )}
          </div>
        )}
      </div>

      <div className="form-actions">
        <button
          type="button"
          className="test-btn"
          onClick={handleTest}
          disabled={isTesting || !formData.rtspUrl}
        >
          Test
        </button>
        <button type="button" className="cancel-btn" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="save-btn">
          Save
        </button>
      </div>
    </form>
  );
};

export default CameraForm;