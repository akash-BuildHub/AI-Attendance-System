import pickle
import numpy as np
import cv2
import face_recognition
from datetime import datetime
import os
from typing import List, Dict, Any
from collections import defaultdict

from google_api import google_api

ENCODINGS_FILE = "face_encodings.pkl"


class FaceStore:
    def __init__(self):
        self.encodings: List[np.ndarray] = []
        self.names: List[str] = []
        self.face_ids: List[str] = []
        self.training_people: Dict[str, List[Dict[str, str]]] = {}

        self.load_encodings()
        self.load_training_data()

    def load_encodings(self):
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.encodings = data.get("encodings", [])
                    self.names = data.get("names", [])
                    self.face_ids = data.get("face_ids", [])
                print(f"✅ Loaded {len(self.encodings)} face encodings")
            except Exception as e:
                print(f"❌ Error loading encodings: {e}")
                self.encodings, self.names, self.face_ids = [], [], []
        else:
            print("⚠️ No existing face encodings found")

    def save_encodings(self):
        data = {"encodings": self.encodings, "names": self.names, "face_ids": self.face_ids}
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Saved {len(self.encodings)} face encodings")

    def load_training_data(self):
        if not google_api.is_authenticated():
            print("⚠️ Google API not authenticated")
            return

        try:
            print("🔄 Loading training data from Google Drive...")
            self.training_people = google_api.list_training_people()

            if not self.training_people:
                print("⚠️ WARNING: No training people found")

            for person, imgs in self.training_people.items():
                print(f"  👤 {person}: {len(imgs)} images")
            print(f"✅ Loaded training data for {len(self.training_people)} people")

        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            self.training_people = {}

    def train_from_google_drive(self) -> Dict[str, Any]:
        if not google_api.is_authenticated():
            return {"success": False, "message": "Google API not authenticated"}

        try:
            self.load_training_data()

            if not self.training_people:
                return {"success": False, "message": "No training people found in Drive"}

            print("🎯 Starting face training...")
            self.encodings, self.names, self.face_ids = [], [], []

            total_faces = 0
            trained_people = []

            for person_name, images in self.training_people.items():
                print(f"\n👤 Training: {person_name} ({len(images)} images)")

                images_ok = 0
                faces_found = 0

                for idx, image_info in enumerate(images):
                    try:
                        print(f"  📸 [{idx+1}/{len(images)}] {image_info.get('name', '?')}")
                        
                        image_bytes = google_api.download_image(image_info["id"])
                        if not image_bytes:
                            print(f"     ❌ Failed to download")
                            continue

                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if image is None:
                            print(f"     ❌ Failed to decode")
                            continue

                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # hog is faster; cnn is more accurate but slow without GPU
                        face_locations = face_recognition.face_locations(rgb, model="hog")
                        if not face_locations:
                            print(f"     ❌ No faces found")
                            continue

                        # ✅ Better stability
                        face_encs = face_recognition.face_encodings(
                            rgb,
                            face_locations,
                            num_jitters=2,
                            model="small"
                        )

                        for enc in face_encs:
                            self.encodings.append(enc)
                            self.names.append(person_name)
                            self.face_ids.append(f"train_{datetime.now().timestamp()}")
                            faces_found += 1

                        images_ok += 1
                        print(f"     ✅ Found {len(face_encs)} face(s)")

                    except Exception as e:
                        print(f"     ❌ Error: {e}")
                        continue

                if faces_found > 0:
                    trained_people.append({"name": person_name, "images": images_ok, "faces": faces_found})
                    total_faces += faces_found
                    print(f"  ✅ {person_name}: {faces_found} face(s) trained")
                else:
                    print(f"  ⚠️ {person_name}: No faces trained")

            if total_faces > 0:
                self.save_encodings()
                msg = f"Trained {total_faces} faces for {len(trained_people)} people"
                print(f"✅ Training completed: {msg}")
                
                if os.path.exists(ENCODINGS_FILE):
                    print(f"💾 Saved to: {ENCODINGS_FILE} ({os.path.getsize(ENCODINGS_FILE)} bytes)")
                else:
                    print(f"❌ ERROR: {ENCODINGS_FILE} not created!")
                
                return {"success": True, "message": msg, "data": {"total_faces": total_faces, "people_trained": len(trained_people)}}

            return {"success": False, "message": "No faces found in training images"}

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Training failed: {str(e)}"}

    @staticmethod
    def _distance_to_conf(distance: float, tolerance: float) -> float:
        """
        Confidence normalized around tolerance:
        - distance == 0 -> 1.0
        - distance == tolerance -> ~0.5
        - distance > tolerance -> < 0.5
        """
        if tolerance <= 0:
            return 0.0
        conf = 1.0 - (distance / (tolerance * 2.0))
        return float(max(0.0, min(1.0, conf)))

    def recognize_faces(
        self,
        rgb_frame: np.ndarray,
        tolerance: float = 0.50,  # Increased default tolerance for CCTV
        model: str = "hog",
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model=model)
            if not face_locations:
                return []

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model="small")

            if not self.encodings:
                return [{
                    "name": "Unknown",
                    "location": (left, top, right, bottom),
                    "confidence": 0.0,
                    "is_known": False
                } for (top, right, bottom, left) in face_locations]

            known_encs = np.array(self.encodings)

            results: List[Dict[str, Any]] = []
            eps = 1e-6

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                distances = face_recognition.face_distance(known_encs, face_encoding)

                # top-k nearest
                idxs = np.argsort(distances)[:max(1, top_k)]
                nearest = [(int(i), float(distances[i])) for i in idxs]

                # keep only those within tolerance
                inliers = [(i, d) for (i, d) in nearest if d <= tolerance]

                if not inliers:
                    results.append({
                        "name": "Unknown",
                        "location": (left, top, right, bottom),
                        "confidence": 0.0,
                        "is_known": False,
                        "distance": float(nearest[0][1]) if nearest else 1.0
                    })
                    continue

                # weighted vote
                votes = defaultdict(float)
                for i, d in inliers:
                    votes[self.names[i]] += 1.0 / (d + eps)

                best_name = max(votes.items(), key=lambda x: x[1])[0]
                best_dist = min(d for (i, d) in inliers if self.names[i] == best_name)

                results.append({
                    "name": best_name,
                    "location": (left, top, right, bottom),
                    "confidence": self._distance_to_conf(best_dist, tolerance),
                    "is_known": True,
                    "distance": best_dist
                })

            return results

        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return []

    def get_training_status(self) -> Dict[str, Any]:
        return {
            "encodings_count": len(self.encodings),
            "unique_people": len(set(self.names)),
            "people_list": sorted(set(self.names)),
            "drive_training_people": sorted(list(self.training_people.keys())) if self.training_people else [],
        }


face_store = FaceStore()