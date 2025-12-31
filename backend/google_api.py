import os
import io
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import gspread

SERVICE_ACCOUNT_FILE = "service_account.json"

# Google Drive Folder Structure
ROOT_FOLDER = "Grow AI"
VIDEO_FOLDER = "AI Attendance Videos"
IMAGE_DATA_FOLDER = "Image Data"
CAPTURED_IMAGES_FOLDER = "Captured Images"
KNOWN_IMAGES_FOLDER = "Known Images"
UNKNOWN_IMAGES_FOLDER = "Unknown Images"
PERSON_IMAGES_FOLDER = "Person Images"

# Google Sheet
SHEET_NAME = "AI Attendance Log"

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

class GoogleAPI:
    def __init__(self):
        self.creds = None
        self.drive_service = None
        self.sheets_service = None
        self.gc = None
        self.worksheet = None
        
        # Folder IDs (cached after setup)
        self.root_id = None
        self.video_folder_id = None
        self.image_data_id = None
        self.captured_images_id = None
        self.known_images_id = None
        self.unknown_images_id = None
        self.person_images_id = None
        
        # Camera video counters
        self.video_counters_file = "video_counters.json"
        self.video_counters = self._load_video_counters()
        
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            self.authenticate()
        else:
            print(f"❌ {SERVICE_ACCOUNT_FILE} not found. Google integration disabled.")
    
    def authenticate(self):
        try:
            self.creds = Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            self.drive_service = build("drive", "v3", credentials=self.creds)
            self.sheets_service = build("sheets", "v4", credentials=self.creds)
            self.gc = gspread.authorize(self.creds)
            
            # Setup Drive structure
            self._setup_drive_structure()
            
            # Setup Google Sheet
            self._setup_sheet()
            
            print("✅ Google authentication and setup successful")
            
        except Exception as e:
            print(f"❌ Google authentication failed: {e}")
    
    def _load_video_counters(self) -> Dict[str, int]:
        """Load video counters for each camera."""
        try:
            if os.path.exists(self.video_counters_file):
                with open(self.video_counters_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_video_counters(self):
        """Save video counters for each camera."""
        try:
            with open(self.video_counters_file, 'w') as f:
                json.dump(self.video_counters, f)
        except:
            pass
    
    def _setup_drive_structure(self):
        """Create or get all required folders in Google Drive."""
        try:
            # 1. Get or create root folder "Grow AI"
            self.root_id = self._get_or_create_folder(None, ROOT_FOLDER)
            
            # 2. Create "AI Attendance Videos" folder
            self.video_folder_id = self._get_or_create_folder(self.root_id, VIDEO_FOLDER)
            
            # 3. Create "Image Data" folder
            self.image_data_id = self._get_or_create_folder(self.root_id, IMAGE_DATA_FOLDER)
            
            # 4. Create "Captured Images" folder
            self.captured_images_id = self._get_or_create_folder(self.image_data_id, CAPTURED_IMAGES_FOLDER)
            
            # 5. Create "Known Images" and "Unknown Images" folders
            self.known_images_id = self._get_or_create_folder(self.captured_images_id, KNOWN_IMAGES_FOLDER)
            self.unknown_images_id = self._get_or_create_folder(self.captured_images_id, UNKNOWN_IMAGES_FOLDER)
            
            # 6. Create "Person Images" folder (for training data)
            self.person_images_id = self._get_or_create_folder(self.image_data_id, PERSON_IMAGES_FOLDER)
            
            print("✅ Google Drive structure verified/created")
            
        except Exception as e:
            print(f"❌ Error setting up Drive structure: {e}")
    
    def _setup_sheet(self):
        """Setup Google Sheet with correct headers."""
        try:
            # Try to open existing sheet
            try:
                spreadsheet = self.gc.open(SHEET_NAME)
            except gspread.SpreadsheetNotFound:
                # Create new spreadsheet
                spreadsheet = self.gc.create(SHEET_NAME)
            
            self.worksheet = spreadsheet.sheet1
            
            # Set headers if not already set
            headers = self.worksheet.row_values(1)
            required_headers = ["Date", "Time", "Camera Name", "Person Name", "Face Image URL"]
            
            if headers != required_headers:
                self.worksheet.clear()
                self.worksheet.append_row(required_headers)
                print("✅ Created headers in Google Sheet")
            
            print(f"✅ Connected to Google Sheet: {SHEET_NAME}")
            
        except Exception as e:
            print(f"❌ Failed to setup worksheet: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if Google API is authenticated and ready."""
        return all([
            self.creds, 
            self.drive_service, 
            self.gc, 
            self.worksheet,
            self.root_id,
            self.video_folder_id
        ])
    
    # ========== FOLDER MANAGEMENT ==========
    
    def _get_or_create_folder(self, parent_id: Optional[str], folder_name: str) -> str:
        """Get existing folder ID or create new one."""
        try:
            # Build query
            query_parts = [
                f"name='{folder_name}'",
                "mimeType='application/vnd.google-apps.folder'",
                "trashed=false"
            ]
            
            if parent_id:
                query_parts.append(f"'{parent_id}' in parents")
            
            query = " and ".join(query_parts)
            
            # Search for existing folder
            results = self.drive_service.files().list(
                q=query, 
                fields="files(id, name)", 
                pageSize=1
            ).execute()
            
            files = results.get("files", [])
            
            if files:
                return files[0]["id"]
            
            # Create new folder
            folder_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            
            if parent_id:
                folder_metadata["parents"] = [parent_id]
            
            folder = self.drive_service.files().create(
                body=folder_metadata, 
                fields="id, name"
            ).execute()
            
            print(f"📁 Created folder: {folder_name}")
            return folder["id"]
            
        except Exception as e:
            print(f"❌ Error getting/creating folder '{folder_name}': {e}")
            return ""
    
    def _get_camera_video_folder(self, camera_name: str) -> str:
        """Get or create camera-specific folder in AI Attendance Videos."""
        return self._get_or_create_folder(self.video_folder_id, camera_name)
    
    # ========== VIDEO UPLOADS ==========
    
    def upload_video_segment(self, camera_name: str, video_path: str) -> str:
        """
        Upload video to: AI Attendance Videos/<camera_name>/Video_N.mp4
        N increments for each upload per camera.
        """
        if not self.is_authenticated() or not os.path.exists(video_path):
            return ""
        
        try:
            # Get camera folder
            camera_folder_id = self._get_camera_video_folder(camera_name)
            if not camera_folder_id:
                return ""
            
            # Get next video number for this camera
            video_num = self.video_counters.get(camera_name, 0) + 1
            self.video_counters[camera_name] = video_num
            self._save_video_counters()
            
            # Create filename
            filename = f"Video_{video_num}.mp4"
            
            # Upload video
            file_metadata = {
                "name": filename,
                "parents": [camera_folder_id],
                "description": f"CCTV Recording - {camera_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            media = MediaIoBaseUpload(
                io.FileIO(video_path, "rb"),
                mimetype="video/mp4",
                resumable=True
            )
            
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, webViewLink"
            ).execute()
            
            # Make it publicly viewable
            permission = {"type": "anyone", "role": "reader"}
            self.drive_service.permissions().create(
                fileId=file["id"],
                body=permission
            ).execute()
            
            print(f"📹 Uploaded video: {camera_name}/{filename}")
            return file["webViewLink"]
            
        except Exception as e:
            print(f"❌ Error uploading video: {e}")
            return ""
    
    # ========== FACE IMAGE UPLOADS ==========
    
    def upload_face_image(self, camera_name: str, person_name: str, image_bytes: bytes) -> str:
        """
        Upload captured face image to correct location:
        - Known person: Image Data/Captured Images/Known Images/<person_name>/
        - Unknown person: Image Data/Captured Images/Unknown Images/
        """
        if not self.is_authenticated():
            return ""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_name}_{person_name}_{timestamp}.jpg"
            
            if person_name and person_name != "Unknown":
                # Known person - upload to their folder
                person_folder_id = self._get_or_create_folder(self.known_images_id, person_name)
                parent_id = person_folder_id
            else:
                # Unknown person - upload to Unknown Images folder
                parent_id = self.unknown_images_id
            
            # Upload image
            media = MediaIoBaseUpload(
                io.BytesIO(image_bytes),
                mimetype="image/jpeg",
                resumable=True
            )
            
            file_metadata = {
                "name": filename,
                "parents": [parent_id]
            }
            
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, webViewLink"
            ).execute()
            
            # Make it publicly viewable
            permission = {"type": "anyone", "role": "reader"}
            self.drive_service.permissions().create(
                fileId=file["id"],
                body=permission
            ).execute()
            
            print(f"📸 Uploaded face image for {person_name or 'Unknown'}")
            return file["webViewLink"]
            
        except Exception as e:
            print(f"❌ Error uploading face image: {e}")
            return ""
    
    # ========== GOOGLE SHEET LOGGING ==========
    
    def log_attendance(self, data: Dict[str, Any]) -> bool:
        """
        Log attendance to Google Sheet with exact columns:
        Date, Time, Camera Name, Person Name, Face Image URL
        """
        if not self.is_authenticated():
            return False
        
        try:
            timestamp = datetime.now()
            
            # Extract required data
            date = timestamp.strftime("%Y-%m-%d")
            time = timestamp.strftime("%H:%M:%S")
            camera_name = data.get("camera", "Unknown Camera")
            person_name = data.get("name", "Unknown")
            face_image_url = data.get("face_image_url", "")
            
            # Create row with exact 5 columns
            row = [date, time, camera_name, person_name, face_image_url]
            
            # Append to sheet
            self.worksheet.append_row(row)
            
            print(f"📝 Logged attendance: {person_name} @ {camera_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error logging attendance: {e}")
            return False
    
    # ========== TRAINING DATA ACCESS ==========
    
    def list_person_folders(self) -> List[Dict[str, str]]:
        """List all person folders in Person Images folder."""
        if not self.is_authenticated():
            return []
        
        try:
            query = f"'{self.person_images_id}' in parents and " \
                    f"mimeType='application/vnd.google-apps.folder' and trashed=false"
            
            results = self.drive_service.files().list(
                q=query, fields="files(id, name)", pageSize=100
            ).execute()
            
            folders = results.get("files", [])
            print(f"📁 Found {len(folders)} person folders in Person Images")
            return folders
            
        except Exception as e:
            print(f"❌ Error listing person folders: {e}")
            return []
    
    def list_images_in_folder(self, folder_id: str) -> List[Dict[str, str]]:
        """List all images in a folder."""
        if not self.is_authenticated():
            return []
        
        try:
            image_mimes = ["image/jpeg", "image/png", "image/jpg"]
            mime_query = " or ".join([f"mimeType='{m}'" for m in image_mimes])
            
            query = f"'{folder_id}' in parents and ({mime_query}) and trashed=false"
            
            results = self.drive_service.files().list(
                q=query, fields="files(id, name, mimeType)", pageSize=100
            ).execute()
            
            return results.get("files", [])
            
        except Exception as e:
            print(f"❌ Error listing images: {e}")
            return []
    
    def download_image(self, file_id: str) -> Optional[bytes]:
        """Download image from Google Drive."""
        if not self.is_authenticated():
            return None
        
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(file_stream, request)
            
            done = False
            while not done:
                _, done = downloader.next_chunk()
            
            file_stream.seek(0)
            return file_stream.read()
            
        except Exception as e:
            print(f"❌ Error downloading image: {e}")
            return None
    
    # ========== SYSTEM INFO ==========
    
    def get_drive_info(self) -> Dict[str, Any]:
        """Get Google Drive folder information."""
        info = {
            "authenticated": self.is_authenticated(),
            "drive_structure": {
                "root": ROOT_FOLDER,
                "videos_folder": VIDEO_FOLDER,
                "image_data_folder": IMAGE_DATA_FOLDER,
                "person_images_folder": PERSON_IMAGES_FOLDER
            }
        }
        
        if self.is_authenticated():
            try:
                # Get person folder count
                person_folders = self.list_person_folders()
                info["person_folders_count"] = len(person_folders)
                info["person_folder_names"] = [p["name"] for p in person_folders]
                
                # Get video folder info
                video_query = f"'{self.video_folder_id}' in parents and trashed=false"
                video_results = self.drive_service.files().list(
                    q=video_query, fields="files(name)", pageSize=20
                ).execute()
                info["video_files_count"] = len(video_results.get("files", []))
                
            except Exception as e:
                info["error"] = str(e)
        
        return info
    
    def get_recent_attendance(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent attendance records from Google Sheet."""
        if not self.is_authenticated():
            return []
        
        try:
            # Get all records (excluding header)
            records = self.worksheet.get_all_records()
            
            # Convert to list of dicts
            result = []
            for record in records[-limit:]:
                result.append({
                    "Date": record.get("Date", ""),
                    "Time": record.get("Time", ""),
                    "Camera Name": record.get("Camera Name", ""),
                    "Person Name": record.get("Person Name", ""),
                    "Face Image URL": record.get("Face Image URL", "")
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting attendance records: {e}")
            return []

# Global instance
google_api = GoogleAPI()