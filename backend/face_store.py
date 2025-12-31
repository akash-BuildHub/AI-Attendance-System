import pickle
import numpy as np
import cv2
import face_recognition
from datetime import datetime
import os
from typing import List, Tuple, Dict, Any

from google_api import google_api

ENCODINGS_FILE = "face_encodings.pkl"
TRAINING_CACHE = "training_cache"

class FaceStore:
    def __init__(self):
        self.encodings = []
        self.names = []
        self.face_ids = []
        self.person_images = {}  # person_name -> list of image data
        
        # Create cache directory
        os.makedirs(TRAINING_CACHE, exist_ok=True)
        
        # Load existing encodings
        self.load_encodings()
        
        # Load training data from Google Drive
        self.load_training_data()
    
    def load_encodings(self):
        """Load known face encodings from disk."""
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.encodings = data.get("encodings", [])
                    self.names = data.get("names", [])
                    self.face_ids = data.get("face_ids", [])
                print(f"✅ Loaded {len(self.encodings)} face encodings from disk")
            except Exception as e:
                print(f"❌ Error loading encodings: {e}")
                self.encodings, self.names, self.face_ids = [], [], []
        else:
            print("⚠️ No existing face encodings found. Starting fresh.")
    
    def save_encodings(self):
        """Save face encodings to disk."""
        data = {
            "encodings": self.encodings,
            "names": self.names,
            "face_ids": self.face_ids
        }
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Saved {len(self.encodings)} face encodings to disk")
    
    def load_training_data(self):
        """Load training data from Google Drive."""
        if not google_api.is_authenticated():
            print("⚠️ Google API not authenticated, skipping training data load")
            return
        
        try:
            print("🔄 Loading training data from Google Drive...")
            
            # Get all person folders from Person Images
            person_folders = google_api.list_person_folders()
            
            for person in person_folders:
                person_name = person['name']
                folder_id = person['id']
                
                # Get images for this person
                images = google_api.list_images_in_folder(folder_id)
                
                if images:
                    self.person_images[person_name] = {
                        'folder_id': folder_id,
                        'images': images,
                        'count': len(images)
                    }
                    print(f"  👤 {person_name}: {len(images)} images")
            
            print(f"✅ Loaded training data for {len(self.person_images)} people")
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
    
    def train_from_google_drive(self) -> Dict[str, Any]:
        """
        Train face recognition system using images from Google Drive.
        Downloads images, extracts faces, and creates encodings.
        """
        if not google_api.is_authenticated():
            return {"success": False, "message": "Google API not authenticated"}
        
        try:
            print("🎯 Starting face training from Google Drive...")
            
            # Reset existing encodings
            self.encodings = []
            self.names = []
            self.face_ids = []
            
            total_faces = 0
            trained_people = []
            
            # Process each person
            for person_name, person_data in self.person_images.items():
                print(f"👤 Training: {person_name}")
                
                images_downloaded = 0
                faces_found = 0
                
                for image_info in person_data['images']:
                    try:
                        # Download image from Google Drive
                        image_bytes = google_api.download_image(image_info['id'])
                        if not image_bytes:
                            continue
                        
                        # Convert to numpy array
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is None:
                            print(f"  ⚠️ Failed to decode image: {image_info['name']}")
                            continue
                        
                        # Convert BGR to RGB for face_recognition
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Detect faces (use CNN if GPU available)
                        model = "hog"  # Default for CPU
                        face_locations = face_recognition.face_locations(rgb_image, model=model)
                        
                        if not face_locations:
                            print(f"  ⚠️ No faces found in: {image_info['name']}")
                            continue
                        
                        # Get face encodings
                        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        
                        # Add each face encoding
                        for face_encoding in face_encodings:
                            self.encodings.append(face_encoding)
                            self.names.append(person_name)
                            self.face_ids.append(f"train_{datetime.now().timestamp()}")
                            faces_found += 1
                        
                        images_downloaded += 1
                        
                        print(f"  ✅ {image_info['name']}: {len(face_encodings)} face(s)")
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {image_info['name']}: {e}")
                        continue
                
                if faces_found > 0:
                    trained_people.append({
                        'name': person_name,
                        'images': images_downloaded,
                        'faces': faces_found
                    })
                    total_faces += faces_found
            
            # Save encodings to disk
            if total_faces > 0:
                self.save_encodings()
                
                result = {
                    "success": True,
                    "message": f"Trained {total_faces} faces for {len(trained_people)} people",
                    "data": {
                        "total_faces": total_faces,
                        "people_trained": len(trained_people),
                        "details": trained_people
                    }
                }
                
                print(f"✅ Training completed: {total_faces} faces for {len(trained_people)} people")
                return result
            else:
                return {
                    "success": False,
                    "message": "No faces found in training images"
                }
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return {
                "success": False,
                "message": f"Training failed: {str(e)}"
            }
    
    def recognize_faces(self, rgb_frame: np.ndarray, tolerance: float = 0.5, model: str = "hog") -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Detect and recognize faces in an RGB frame.
        Returns list of (name, (left, top, right, bottom), confidence)
        model: "hog" for CPU (faster), "cnn" for GPU (more accurate)
        """
        try:
            # 1) Detect face locations first
            face_locations = face_recognition.face_locations(rgb_frame, model=model)
            if not face_locations:
                return []

            # 2) Get encodings for detected faces
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # 3) If no known encodings exist, return all as "Unknown"
            if not self.encodings:
                return [("Unknown", (left, top, right, bottom), 0.0)
                        for (top, right, bottom, left) in face_locations]
            
            results: List[Tuple[str, Tuple[int, int, int, int], float]] = []

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                # Compare with known encodings
                face_distances = face_recognition.face_distance(self.encodings, face_encoding)
                
                if len(face_distances) == 0:
                    results.append(("Unknown", (left, top, right, bottom), 0.0))
                    continue
                
                best_match_index = int(np.argmin(face_distances))
                best_distance = float(face_distances[best_match_index])

                if best_distance <= tolerance:
                    name = self.names[best_match_index]
                    confidence = 1.0 - best_distance
                else:
                    name = "Unknown"
                    confidence = 0.0

                results.append((name, (left, top, right, bottom), confidence))

            return results

        except Exception as e:
            print(f"❌ Error in face recognition: {e}")
            return []
    
    def retrain_after_capture(self):
        """Automatically retrain after new face capture."""
        print("🔄 Retraining after new face capture...")
        self.load_training_data()
        result = self.train_from_google_drive()
        if result.get("success"):
            print("✅ Retraining completed successfully")
        else:
            print("⚠️ Retraining had issues:", result.get("message"))
    
    def get_all_people(self) -> List[str]:
        """Get list of all unique people names."""
        return sorted(set(self.names))
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status and statistics."""
        return {
            "encodings_count": len(self.encodings),
            "unique_people": len(set(self.names)),
            "people_list": self.get_all_people(),
            "training_data": {
                "people_in_drive": len(self.person_images),
                "people_details": list(self.person_images.keys())
            }
        }
    
    def add_person_manual(self, name: str, image_paths: List[str]) -> int:
        """Manually add a person with local images (for testing)."""
        added = 0
        
        for path in image_paths:
            try:
                # Load image
                image = face_recognition.load_image_file(path)
                
                # Detect faces
                face_locations = face_recognition.face_locations(image, model="hog")
                if not face_locations:
                    continue
                
                # Get encodings
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                for face_encoding in face_encodings:
                    self.encodings.append(face_encoding)
                    self.names.append(name)
                    self.face_ids.append(f"manual_{datetime.now().timestamp()}")
                    added += 1
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        if added > 0:
            self.save_encodings()
        
        return added

# Global instance
face_store = FaceStore()