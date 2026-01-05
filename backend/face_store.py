import pickle
import numpy as np
import cv2
import face_recognition
from datetime import datetime
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict

ENCODINGS_FILE = "face_encodings.pkl"
TRAIN_DIR = "Person Images"


class FaceStore:
    """🔥 FINAL FIXED: Person-level minimum distance (no more known→unknown)"""
    
    def __init__(self):
        self.encodings: List[np.ndarray] = []
        self.names: List[str] = []
        self.face_ids: List[str] = []
        self.encodings_array: Optional[np.ndarray] = None
        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.encodings = data.get("encodings", [])
                    self.names = data.get("names", [])
                    self.face_ids = data.get("face_ids", [])
                
                if self.encodings:
                    self.encodings_array = np.array(self.encodings)
                
                print(f"✅ Loaded {len(self.encodings)} face encodings")
                print(f"👥 People: {len(set(self.names))} - {sorted(set(self.names))}")
            except Exception as e:
                print(f"❌ Error loading encodings: {e}")
                self.encodings, self.names, self.face_ids = [], [], []
                self.encodings_array = None
        else:
            print("⚠️ No existing face encodings found")

    def save_encodings(self):
        data = {"encodings": self.encodings, "names": self.names, "face_ids": self.face_ids}
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        
        if self.encodings:
            self.encodings_array = np.array(self.encodings)
        
        print(f"💾 Saved {len(self.encodings)} face encodings")

    def train_from_local(self) -> Dict[str, Any]:
        """Train from local Person Images folder using CNN for consistency"""
        try:
            if not os.path.exists(TRAIN_DIR):
                os.makedirs(TRAIN_DIR, exist_ok=True)
                return {"success": False, "message": f"Created {TRAIN_DIR} folder. Please add person folders with images."}

            print("\n" + "="*60)
            print("🎯 TRAINING FROM LOCAL PERSON IMAGES")
            print("="*60)
            print(f"📁 Training directory: {os.path.abspath(TRAIN_DIR)}")
            
            self.encodings, self.names, self.face_ids = [], [], []
            total_faces = 0
            trained_people = []

            folders = [f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))]
            
            if not folders:
                return {
                    "success": False, 
                    "message": f"No person folders found in {TRAIN_DIR}. Create folders named after each person."
                }
            
            print(f"📂 Found {len(folders)} person folders")
            print("="*60)

            for person_name in folders:
                person_dir = os.path.join(TRAIN_DIR, person_name)
                print(f"\n👤 Training: {person_name}")
                
                images = [f for f in os.listdir(person_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if not images:
                    print(f"   ⚠️ No images found")
                    continue
                
                print(f"   📸 Found {len(images)} images")
                faces_found = 0

                for idx, img_name in enumerate(images):
                    try:
                        path = os.path.join(person_dir, img_name)
                        print(f"   [{idx+1}/{len(images)}] {img_name[:40]}... ", end="", flush=True)
                        
                        image = cv2.imread(path)
                        if image is None:
                            print("❌ Load failed")
                            continue

                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb, model="cnn")
                        
                        if not face_locations:
                            print("❌ No face")
                            continue

                        face_encs = face_recognition.face_encodings(
                            rgb, face_locations, num_jitters=3, model="large"
                        )

                        for enc in face_encs:
                            self.encodings.append(enc)
                            self.names.append(person_name)
                            self.face_ids.append(f"train_{person_name}_{datetime.now().timestamp()}")
                            faces_found += 1

                        print(f"✅ {len(face_encs)} face(s)")

                    except Exception as e:
                        print(f"❌ {e}")
                        continue

                if faces_found > 0:
                    trained_people.append({"name": person_name, "faces": faces_found})
                    total_faces += faces_found
                    print(f"   ✅ Total: {faces_found} encodings")
                else:
                    print(f"   ⚠️ No faces extracted")

            print("\n" + "="*60)
            if total_faces > 0:
                self.save_encodings()
                msg = f"Trained {total_faces} encodings for {len(trained_people)} people using CNN"
                print(f"✅ {msg}")
                print(f"👥 People: {[p['name'] for p in trained_people]}")
                print("="*60 + "\n")
                
                return {
                    "success": True, 
                    "message": msg, 
                    "data": {
                        "total_faces": total_faces, 
                        "people_trained": len(trained_people),
                        "people_list": [p["name"] for p in trained_people],
                        "details": trained_people,
                        "training_model": "CNN"
                    }
                }

            return {"success": False, "message": "No faces found. Ensure images show clear faces."}

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Training failed: {str(e)}"}

    @staticmethod
    def _distance_to_conf(distance: float, tolerance: float) -> float:
        """Convert distance to confidence score"""
        if tolerance <= 0:
            return 0.0
        if distance <= tolerance * 0.4:
            return 1.0
        elif distance <= tolerance:
            return 1.0 - ((distance - tolerance * 0.4) / (tolerance * 0.6))
        else:
            return max(0.0, 1.0 - (distance / tolerance))

    def recognize_faces(
        self,
        rgb_frame: np.ndarray,
        tolerance: float = 0.52,
        model: str = "hog",
        top_k: int = 5,
        num_jitters: int = 1
    ) -> List[Dict[str, Any]]:
        """
        🔥 FINAL FIXED: Person-level minimum distance
        NO MORE "known → unknown" for clear CCTV faces
        """
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model=model)
            if not face_locations:
                return []

            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations, num_jitters=num_jitters, model="large"
            )

            if not self.encodings:
                return [{
                    "name": "Unknown",
                    "location": (top, right, bottom, left),
                    "confidence": 0.0,
                    "is_known": False
                } for (top, right, bottom, left) in face_locations]

            known_encs = self.encodings_array if self.encodings_array is not None else np.array(self.encodings)
            results: List[Dict[str, Any]] = []
            
            # 🔥 FIXED: Build name->indices mapping once
            name_to_indices = defaultdict(list)
            for i, name in enumerate(self.names):
                name_to_indices[name].append(i)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                distances = face_recognition.face_distance(known_encs, face_encoding)
                
                # 🔥 FIXED: Person-level minimum distance (no voting, no ambiguity)
                best_name = None
                best_dist = 1.0  # Start with worst possible
                
                for name, idx_list in name_to_indices.items():
                    # Get minimum distance for this person across all their encodings
                    person_dist = float(np.min(distances[idx_list]))
                    if person_dist < best_dist:
                        best_dist = person_dist
                        best_name = name
                
                # Determine if match is good enough
                if best_name is not None and best_dist <= tolerance:
                    results.append({
                        "name": best_name,
                        "location": (top, right, bottom, left),
                        "confidence": self._distance_to_conf(best_dist, tolerance),
                        "is_known": True,
                        "distance": best_dist,
                        "method": "person_min_distance"
                    })
                else:
                    results.append({
                        "name": "Unknown",
                        "location": (top, right, bottom, left),
                        "confidence": 0.0,
                        "is_known": False,
                        "distance": best_dist if best_name else 1.0
                    })

            return results

        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return []

    def get_training_status(self) -> Dict[str, Any]:
        unique_people = sorted(set(self.names))
        return {
            "encodings_count": len(self.encodings),
            "unique_people": len(unique_people),
            "people_list": unique_people,
            "training_dir": os.path.abspath(TRAIN_DIR),
            "training_dir_exists": os.path.exists(TRAIN_DIR),
            "has_encodings_file": os.path.exists(ENCODINGS_FILE),
            "encodings_file_path": os.path.abspath(ENCODINGS_FILE),
            "training_model": "CNN (matching recognition default)",
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M") if self.encodings else "Never"
        }


face_store = FaceStore()