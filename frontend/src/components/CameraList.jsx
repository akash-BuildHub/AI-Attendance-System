import React, { useState } from 'react';
import CameraItem from './CameraItem';

const CameraList = ({ cameras, onStartFeed, onStopFeed, onEdit, onDelete }) => {
  if (cameras.length === 0) {
    return (
      <div className="empty-state">
        <p>No cameras added. Click "Add Camera" to get started.</p>
      </div>
    );
  }

  return (
    <div className="camera-list">
      {cameras.map(camera => (
        <CameraItem
          key={camera.id}
          camera={camera}
          onStartFeed={onStartFeed}
          onStopFeed={onStopFeed}
          onEdit={onEdit}
          onDelete={onDelete}
        />
      ))}
    </div>
  );
};

export default CameraList;