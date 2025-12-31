import React from 'react';

const AboutModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>About Camera Management System</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          <p>
            This web application allows you to manage and monitor multiple IP cameras 
            in a centralized dashboard. You can add, edit, and remove cameras, 
            test connections, and view live feeds.
          </p>
          <p>
            Features include:
          </p>
          <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
            <li>Add multiple cameras with detailed configuration</li>
            <li>Auto-generate RTSP URLs for easy setup</li>
            <li>Test camera connections in real-time</li>
            <li>View live camera feeds with start/stop controls</li>
            <li>Edit or delete existing camera configurations</li>
            <li>Clean and intuitive user interface</li>
          </ul>
          <p>
            To get started, click the "Add Camera" button and fill in your camera's 
            details. Once added, you can start the live feed by clicking the green 
            button next to each camera.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AboutModal;