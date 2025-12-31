import React, { useState, useRef, useEffect } from "react";

const CameraItem = ({ camera, onStartFeed, onStopFeed, onEdit, onDelete }) => {
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef(null);

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setShowMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleEditClick = () => {
    setShowMenu(false);
    onEdit(camera); // <- send full camera to App
  };

  const handleDeleteClick = () => {
    setShowMenu(false);
    onDelete(camera.id);
  };

  return (
    <div className="camera-item">
      <div className="camera-info">
        <div className="camera-name">{camera.name}</div>
        <div className="camera-ip">{camera.ip}</div>
      </div>
      <div className="camera-controls">
        {!camera.isLive ? (
          <button
            onClick={() => onStartFeed(camera)}
            className="control-btn start-btn"
            title="Start Feed"
          >
            ▶
          </button>
        ) : (
          <button
            onClick={() => onStopFeed(camera.id)}
            className="control-btn stop-btn"
            title="Stop Feed"
          >
            ■
          </button>
        )}

        <div className="dropdown-menu" ref={menuRef}>
          <button
            onClick={() => setShowMenu((prev) => !prev)}
            className="control-btn menu-btn"
            title="Menu"
          >
            ⋮
          </button>
          {showMenu && (
            <div className="menu-content">
              <button onClick={handleEditClick} className="menu-item edit-item">
                Edit
              </button>
              <button
                onClick={handleDeleteClick}
                className="menu-item delete-item"
              >
                Delete
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CameraItem;