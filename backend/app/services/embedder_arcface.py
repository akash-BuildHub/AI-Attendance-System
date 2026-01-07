import cv2
import numpy as np

class ArcFaceEmbedder:
    def __init__(self):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("insightface is required to start a camera session") from exc
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def embed_face_crop(self, face_bgr: np.ndarray):
        """
        Input: face crop BGR
        Output: normalized embedding (512,)
        """
        faces = self.app.get(face_bgr)
        if not faces:
            return None
        # choose largest face in crop
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
            reverse=True
        )
        emb = faces[0].embedding
        if emb is None:
            return None
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb

def expand_bbox_for_head_neck(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1
    h = y2 - y1
    # include head + neck
    top_expand = int(h * 0.7)
    bottom_expand = int(h * 0.9)
    side_expand = int(w * 0.45)

    nx1 = max(0, x1 - side_expand)
    nx2 = min(img_w, x2 + side_expand)
    ny1 = max(0, y1 - top_expand)
    ny2 = min(img_h, y2 + bottom_expand)
    return nx1, ny1, nx2, ny2
