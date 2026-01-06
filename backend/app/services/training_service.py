import os
import pickle
import cv2
import numpy as np

from app.core.config import PERSON_IMAGES_DIR, PROTOTYPES_PATH, EMBEDDINGS_DIR, RUNTIME
from app.services.detector_yolo import YoloFaceDetector
from app.services.embedder_arcface import ArcFaceEmbedder, expand_bbox_for_head_neck

def train_prototypes():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    det = YoloFaceDetector()
    emb = ArcFaceEmbedder()

    prototypes = {}

    if not os.path.isdir(PERSON_IMAGES_DIR):
        return {"success": False, "message": f"Missing folder: {PERSON_IMAGES_DIR}"}

    people = [p for p in os.listdir(PERSON_IMAGES_DIR) if os.path.isdir(os.path.join(PERSON_IMAGES_DIR, p))]
    if not people:
        return {"success": False, "message": "No person folders found in data/person_images"}

    for person in people:
        person_dir = os.path.join(PERSON_IMAGES_DIR, person)
        embs = []

        imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        for img_name in imgs:
            path = os.path.join(person_dir, img_name)
            img = cv2.imread(path)
            if img is None:
                continue

            faces = det.detect(img, conf=RUNTIME["min_det_conf"])
            if not faces:
                continue

            # choose largest detection
            faces = sorted(faces, key=lambda d: (d["xyxy"][2]-d["xyxy"][0])*(d["xyxy"][3]-d["xyxy"][1]), reverse=True)
            x1,y1,x2,y2 = faces[0]["xyxy"]
            h, w = img.shape[:2]
            nx1, ny1, nx2, ny2 = expand_bbox_for_head_neck(x1,y1,x2,y2,w,h)
            crop = img[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                continue

            e = emb.embed_face_crop(crop)
            if e is not None:
                embs.append(e)

        if embs:
            proto = np.mean(np.stack(embs), axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            prototypes[person] = proto

    if not prototypes:
        return {"success": False, "message": "No valid faces found. Add clearer person images."}

    with open(PROTOTYPES_PATH, "wb") as f:
        pickle.dump(prototypes, f)

    return {
        "success": True,
        "message": "Training complete",
        "data": {"people": sorted(list(prototypes.keys())), "count": len(prototypes)}
    }

def load_prototypes():
    if not os.path.exists(PROTOTYPES_PATH):
        return {}
    with open(PROTOTYPES_PATH, "rb") as f:
        return pickle.load(f)
