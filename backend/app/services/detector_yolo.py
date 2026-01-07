import numpy as np

class YoloFaceDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required to start a camera session") from exc
        try:
            self.model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_name}'.") from exc

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.45):
        """
        Returns: list of dict {xyxy, conf}
        """
        results = self.model(frame_bgr, verbose=False)
        out = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for xyxy, c in zip(boxes, confs):
                if float(c) < conf:
                    continue
                out.append({"xyxy": xyxy.astype(int), "conf": float(c)})
        return out
