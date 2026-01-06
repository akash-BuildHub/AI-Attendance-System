import numpy as np
import supervision as sv

class ByteTrackerService:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections_xyxy: np.ndarray, confidences: np.ndarray):
        """
        detections_xyxy: (N,4)
        confidences: (N,)
        returns supervision.TrackedDetections
        """
        det = sv.Detections(
            xyxy=detections_xyxy,
            confidence=confidences,
            class_id=np.zeros(len(detections_xyxy), dtype=int)  # single class = face
        )
        tracks = self.tracker.update_with_detections(det)
        return tracks
