import pickle
import numpy as np
import cv2
import insightface
import onnxruntime
from datetime import datetime
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict

ENCODINGS_FILE = "face_encodings.pkl"
TRAIN_DIR = "Person Images"


class FaceStore:
    """🔥 ARCFACE VERSION: Higher accuracy for CCTV/IR/low-res"""
    
    def __init__(self):
        # Initialize InsightFace (ArcFace) model
        print("🧠 Initializing InsightFace (ArcFace) model...")
        
        # Load model - buffalo_l is the best balanced model
        self.model = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']  # Use CPU (can change to CUDA if available)
        )
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        
        # Store embeddings
        self.encodings: List[np.ndarray] = []
        self.names: List[str] = []
        self.face_ids: List[str] = []
        
        print("✅ ArcFace model loaded successfully!")
        
        # Load existing embeddings if available
        self.load_encodings()

    def load_encodings(self):
        """Load face embeddings from pickle file"""
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.encodings = data.get("encodings", [])
                    self.names = data.get("names", [])
                    self.face_ids = data.get("face_ids", [])
                
                print(f"✅ Loaded {len(self.encodings)} face encodings")
                print(f"👥 People: {len(set(self.names))} - {sorted(set(self.names))}")
            except Exception as e:
                print(f"❌ Error loading encodings: {e}")
                self.encodings, self.names, self.face_ids = [], [], []
        else:
            print("⚠️ No existing face encodings found")

    def save_encodings(self):
        """Save face embeddings to pickle file"""
        data = {"encodings": self.encodings, "names": self.names, "face_ids": self.face_ids}
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Saved {len(self.encodings)} face encodings")

    def train_from_local(self) -> Dict[str, Any]:
        """Train from local Person Images folder using ArcFace"""
        try:
            if not os.path.exists(TRAIN_DIR):
                os.makedirs(TRAIN_DIR, exist_ok=True)
                return {
                    "success": False, 
                    "message": f"Created {TRAIN_DIR} folder. Please add person folders with images."
                }

            print("\n" + "="*60)
            print("🎯 TRAINING WITH ARCFACE (INSIGHTFACE)")
            print("="*60)
            print(f"📁 Training directory: {os.path.abspath(TRAIN_DIR)}")
            
            # Clear existing encodings
            self.encodings, self.names, self.face_ids = [], [], []
            total_faces = 0
            trained_people = []

            folders = [f for f in os.listdir(TRAIN_DIR) 
                      if os.path.isdir(os.path.join(TRAIN_DIR, f))]
            
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
                        
                        # Load image
                        image = cv2.imread(path)
                        if image is None:
                            print("❌ Load failed")
                            continue

                        # Detect faces using ArcFace
                        faces = self.model.get(image)
                        
                        if not faces:
                            print("❌ No face")
                            continue
                        
                        # 🔒 FIX 6: Use ONLY the largest face (prevents training pollution)
                        faces = sorted(
                            faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                            reverse=True
                        )
                        faces = faces[:1]  # Take only the largest face
                        
                        # Extract embeddings from detected faces
                        for face in faces:
                            if hasattr(face, 'embedding') and face.embedding is not None:
                                # Normalize embedding
                                embedding = face.embedding / np.linalg.norm(face.embedding)
                                self.encodings.append(embedding)
                                self.names.append(person_name)
                                self.face_ids.append(f"train_{person_name}_{datetime.now().timestamp()}")
                                faces_found += 1

                        print(f"✅ {len(faces)} face(s)")

                    except Exception as e:
                        print(f"❌ {e}")
                        continue

                if faces_found > 0:
                    trained_people.append({"name": person_name, "faces": faces_found})
                    total_faces += faces_found
                    print(f"   ✅ Total: {faces_found} embeddings")
                else:
                    print(f"   ⚠️ No faces extracted")

            print("\n" + "="*60)
            if total_faces > 0:
                self.save_encodings()
                msg = f"Trained {total_faces} embeddings for {len(trained_people)} people using ArcFace"
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
                        "training_model": "ArcFace (InsightFace)"
                    }
                }

            return {"success": False, "message": "No faces found. Ensure images show clear faces."}

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Training failed: {str(e)}"}

    def recognize_faces(
        self,
        rgb_frame: np.ndarray,
        tolerance: float = 0.45,
        **kwargs  # Accept extra parameters for compatibility
    ) -> List[Dict[str, Any]]:
        """
        🔥 ARCFACE RECOGNITION: Uses cosine similarity
        
        Args:
            tolerance: Cosine similarity threshold (0.35-0.55)
                      Higher = stricter matching
        """
        try:
            # Convert RGB to BGR (OpenCV format)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Detect faces using ArcFace
            faces = self.model.get(bgr_frame)
            
            if not faces:
                return []
            
            results: List[Dict[str, Any]] = []
            
            for face in faces:
                if not hasattr(face, 'embedding') or face.embedding is None:
                    continue
                    
                # Get face bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Convert to dlib-style format (top, right, bottom, left)
                top, right, bottom, left = y1, x2, y2, x1
                
                # Normalize face embedding
                face_embedding = face.embedding / np.linalg.norm(face.embedding)
                
                # If no trained faces, return as Unknown
                if not self.encodings:
                    results.append({
                        "name": "Unknown",
                        "location": (top, right, bottom, left),
                        "confidence": 0.0,
                        "is_known": False
                    })
                    continue
                
                # Calculate cosine similarities
                similarities = []
                for stored_embedding in self.encodings:
                    # Normalize stored embedding
                    norm_stored = stored_embedding / np.linalg.norm(stored_embedding)
                    # Cosine similarity
                    similarity = np.dot(face_embedding, norm_stored)
                    similarities.append(similarity)
                
                # Find best match
                best_idx = np.argmax(similarities)
                best_similarity = similarities[best_idx]
                
                # Determine if match is good enough
                if best_similarity >= tolerance:
                    # Group matching for same person
                    person_name = self.names[best_idx]
                    results.append({
                        "name": person_name,
                        "location": (top, right, bottom, left),
                        "confidence": float(best_similarity),
                        "is_known": True,
                        "similarity": float(best_similarity),
                        "method": "arcface_cosine"
                    })
                else:
                    results.append({
                        "name": "Unknown",
                        "location": (top, right, bottom, left),
                        "confidence": 0.0,
                        "is_known": False,
                        "similarity": float(best_similarity)
                    })
            
            return results

        except Exception as e:
            print(f"❌ ArcFace recognition error: {e}")
            return []

    def get_training_status(self) -> Dict[str, Any]:
        unique_people = sorted(set(self.names))
        return {
            "model": "ArcFace (InsightFace)",
            "encodings_count": len(self.encodings),
            "unique_people": len(unique_people),
            "people_list": unique_people,
            "training_dir": os.path.abspath(TRAIN_DIR),
            "training_dir_exists": os.path.exists(TRAIN_DIR),
            "has_encodings_file": os.path.exists(ENCODINGS_FILE),
            "encodings_file_path": os.path.abspath(ENCODINGS_FILE),
            "last_trained": datetime.now().strftime("%Y-%m-d %H:%M") if self.encodings else "Never"
        }


# Singleton instance
face_store = FaceStore()