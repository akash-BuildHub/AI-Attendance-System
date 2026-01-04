import os
import io
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import gspread
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# --- Auth options ---
from google.oauth2.service_account import Credentials as SACredentials
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SERVICE_ACCOUNT_FILE = "service_account.json"

# OAuth files (YOU will add credentials.json)
OAUTH_CLIENT_FILE = "credentials.json"
OAUTH_TOKEN_FILE = "token.json"

ROOT_FOLDER = "Grow AI"
VIDEO_FOLDER = "AI Attendance Videos"
CAPTURED_IMAGES_FOLDER = "Captured Images"
PERSON_IMAGES_FOLDER = "Person Images"  # training images folder

SHEET_NAME = "AI Attendance Log"

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

IMAGE_MIMES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]


class GoogleAPI:
    """
    Key change:
    - Prefer OAuth user creds for Drive uploads (has quota)
    - Fall back to service account for read-only if OAuth not available
    """

    def __init__(self):
        self.creds = None                 # active creds (OAuth preferred)
        self.sa_creds = None              # optional service-account creds
        self.drive_service = None
        self.gc = None
        self.worksheet = None

        self.root_id = None
        self.video_folder_id = None
        self.captured_images_id = None
        self.person_images_id = None

        self.video_counters_file = "video_counters.json"
        self.video_counters = self._load_video_counters()

        self.authenticate()

    # ---------------- AUTH ----------------

    def authenticate(self):
        """
        1) Try OAuth (credentials.json + token.json)
        2) Else try Service Account (read-only workable, but uploads to MyDrive likely fail)
        """
        oauth_ok = False

        # OAuth path
        if os.path.exists(OAUTH_CLIENT_FILE):
            try:
                self.creds = self._load_or_create_oauth_creds()
                oauth_ok = True
                print("✅ Google OAuth authentication ready (user account)")
            except Exception as e:
                print(f"⚠️ OAuth auth failed: {e}")
                self.creds = None

        # Service account path (fallback)
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            try:
                self.sa_creds = SACredentials.from_service_account_file(
                    SERVICE_ACCOUNT_FILE, scopes=SCOPES
                )
                if not oauth_ok:
                    self.creds = self.sa_creds
                print("✅ Service Account credentials loaded")
            except Exception as e:
                print(f"⚠️ Service account load failed: {e}")
                self.sa_creds = None

        if not self.creds:
            print("❌ No Google credentials available (OAuth credentials.json or service_account.json missing)")
            return

        # Build clients with active creds
        self.drive_service = build("drive", "v3", credentials=self.creds)
        self.gc = gspread.authorize(self.creds)

        # Setup Drive + Sheet
        self._setup_drive_structure()
        self._setup_sheet()

        # Detect whether uploads are expected to work
        if self._using_service_account_only():
            print("⚠️ Running on Service Account only. Uploads to My Drive may fail with quota error.")
            print("   Fix: add OAuth credentials.json (recommended) or use a Workspace Shared Drive.")

    def _load_or_create_oauth_creds(self) -> UserCredentials:
        creds = None
        if os.path.exists(OAUTH_TOKEN_FILE):
            creds = UserCredentials.from_authorized_user_file(OAUTH_TOKEN_FILE, SCOPES)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CLIENT_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(OAUTH_TOKEN_FILE, "w") as token:
                token.write(creds.to_json())

        return creds

    def _using_service_account_only(self) -> bool:
        # OAuth creds produce UserCredentials; SA creds produce SACredentials
        return isinstance(self.creds, SACredentials) and not os.path.exists(OAUTH_CLIENT_FILE)

    def is_authenticated(self) -> bool:
        return all([self.creds, self.drive_service, self.gc, self.worksheet,
                    self.root_id, self.video_folder_id, self.captured_images_id, self.person_images_id])

    # ---------------- SETUP ----------------

    def _setup_drive_structure(self):
        # Matches your requested structure:
        # Grow AI/
        #   AI Attendance Videos/
        #   Captured Images/
        #   Person Images/
        self.root_id = self._get_or_create_folder(None, ROOT_FOLDER)
        self.video_folder_id = self._get_or_create_folder(self.root_id, VIDEO_FOLDER)
        self.captured_images_id = self._get_or_create_folder(self.root_id, CAPTURED_IMAGES_FOLDER)
        self.person_images_id = self._get_or_create_folder(self.root_id, PERSON_IMAGES_FOLDER)

        print("✅ Drive structure verified")
        print(f"   Root ID:            {self.root_id}")
        print(f"   Videos ID:          {self.video_folder_id}")
        print(f"   Captured Images ID: {self.captured_images_id}")
        print(f"   Person Images ID:   {self.person_images_id}")

    def _setup_sheet(self):
        try:
            try:
                spreadsheet = self.gc.open(SHEET_NAME)
            except gspread.SpreadsheetNotFound:
                spreadsheet = self.gc.create(SHEET_NAME)

            self.worksheet = spreadsheet.sheet1

            required_headers = ["Date", "Time", "Camera Name", "Person Name", "Face Image URL"]
            headers = self.worksheet.row_values(1)
            if headers != required_headers:
                self.worksheet.clear()
                self.worksheet.append_row(required_headers)

            print(f"✅ Connected to Google Sheet: {SHEET_NAME}")
        except Exception as e:
            print(f"❌ Failed to setup sheet: {e}")
            self.worksheet = None

    # ---------------- DRIVE HELPERS ----------------

    def _get_or_create_folder(self, parent_id: Optional[str], folder_name: str) -> str:
        query_parts = [
            f"name='{folder_name}'",
            "mimeType='application/vnd.google-apps.folder'",
            "trashed=false",
        ]
        if parent_id:
            query_parts.append(f"'{parent_id}' in parents")
        query = " and ".join(query_parts)

        results = self.drive_service.files().list(
            q=query, fields="files(id, name)", pageSize=1
        ).execute()
        files = results.get("files", [])
        if files:
            return files[0]["id"]

        folder_metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            folder_metadata["parents"] = [parent_id]

        folder = self.drive_service.files().create(body=folder_metadata, fields="id").execute()
        print(f"📁 Created folder: {folder_name}")
        return folder["id"]

    def _list_children(self, parent_id: str, mime_types: Optional[List[str]] = None) -> List[Dict[str, str]]:
        if not self.is_authenticated():
            return []

        query = [f"'{parent_id}' in parents", "trashed=false"]
        if mime_types:
            mime_query = " or ".join([f"mimeType='{m}'" for m in mime_types])
            query.append(f"({mime_query})")

        q = " and ".join(query)
        results = self.drive_service.files().list(
            q=q, fields="files(id,name,mimeType)", pageSize=1000
        ).execute()
        return results.get("files", [])

    # ---------------- UPLOADS ----------------

    def _load_video_counters(self) -> Dict[str, int]:
        try:
            if os.path.exists(self.video_counters_file):
                with open(self.video_counters_file, "r") as f:
                    return json.load(f)
        except:
            pass
        return {}

    def _save_video_counters(self):
        try:
            with open(self.video_counters_file, "w") as f:
                json.dump(self.video_counters, f)
        except:
            pass

    def _get_camera_video_folder(self, camera_name: str) -> str:
        return self._get_or_create_folder(self.video_folder_id, camera_name)

    def upload_video_segment(self, camera_name: str, video_path: str) -> str:
        if not self.is_authenticated() or not os.path.exists(video_path):
            return ""

        try:
            camera_folder_id = self._get_camera_video_folder(camera_name)
            if not camera_folder_id:
                return ""

            video_num = self.video_counters.get(camera_name, 0) + 1
            self.video_counters[camera_name] = video_num
            self._save_video_counters()

            filename = f"Video_{video_num}.mp4"

            file_metadata = {"name": filename, "parents": [camera_folder_id]}
            media = MediaIoBaseUpload(io.FileIO(video_path, "rb"), mimetype="video/mp4", resumable=True)

            file = self.drive_service.files().create(
                body=file_metadata, media_body=media, fields="id, webViewLink"
            ).execute()

            # public read link
            self.drive_service.permissions().create(
                fileId=file["id"], body={"type": "anyone", "role": "reader"}
            ).execute()

            print(f"📹 Uploaded video: {camera_name}/{filename}")
            return file.get("webViewLink", "")

        except Exception as e:
            print(f"❌ Error uploading video: {e}")
            return ""

    def upload_captured_face(self, camera_name: str, person_name: str, image_bytes: bytes) -> str:
        """
        Your structure:
        Grow AI/Captured Images/<PersonName>/
        Unknown -> Grow AI/Captured Images/Unknown/
        """
        if not self.is_authenticated():
            return ""

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_person = (person_name or "Unknown").strip() or "Unknown"
            safe_person = safe_person.replace("/", "_").replace("\\", "_")

            person_folder_id = self._get_or_create_folder(self.captured_images_id, safe_person)
            filename = f"{camera_name}_{safe_person}_{timestamp}.jpg"

            media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype="image/jpeg", resumable=True)
            file = self.drive_service.files().create(
                body={"name": filename, "parents": [person_folder_id]},
                media_body=media,
                fields="id, webViewLink"
            ).execute()

            self.drive_service.permissions().create(
                fileId=file["id"], body={"type": "anyone", "role": "reader"}
            ).execute()

            print(f"📸 Uploaded {'known' if safe_person != 'Unknown' else 'unknown'} face: {safe_person}")
            return file.get("webViewLink", "")

        except Exception as e:
            print(f"❌ Error uploading captured face: {e}")
            return ""

    # ---------------- SHEETS ----------------

    def log_attendance(self, camera_name: str, person_name: str, face_image_url: str) -> bool:
        if not self.is_authenticated():
            return False
        try:
            ts = datetime.now()
            date = ts.strftime("%d-%m-%Y")
            t = ts.strftime("%H:%M:%S")
            self.worksheet.append_row([date, t, camera_name, person_name, face_image_url])
            print(f"📝 Logged attendance: {person_name} @ {camera_name}")
            return True
        except Exception as e:
            print(f"❌ Error logging attendance: {e}")
            return False

    # ---------------- TRAINING READ ----------------

    def list_training_people(self) -> Dict[str, List[Dict[str, str]]]:
        if not self.is_authenticated():
            return {}

        people: Dict[str, List[Dict[str, str]]] = {}
        children = self._list_children(self.person_images_id, None)

        for item in children:
            mime = item.get("mimeType", "")
            name = item.get("name", "")

            if mime == "application/vnd.google-apps.folder":
                person_name = name
                images = self._list_children(item["id"], IMAGE_MIMES)
                if images:
                    people.setdefault(person_name, []).extend(images)

            elif mime in IMAGE_MIMES:
                stem = Path(name).stem
                people.setdefault(stem, []).append(item)

        return people

    def download_image(self, file_id: str) -> Optional[bytes]:
        if not self.is_authenticated():
            return None
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            fh.seek(0)
            return fh.read()
        except Exception as e:
            print(f"❌ Error downloading image: {e}")
            return None

    def get_drive_info(self) -> Dict[str, Any]:
        return {
            "authenticated": self.is_authenticated(),
            "auth_mode": "oauth" if os.path.exists(OAUTH_CLIENT_FILE) else "service_account",
            "root_id": self.root_id,
            "video_folder_id": self.video_folder_id,
            "captured_images_id": self.captured_images_id,
            "person_images_id": self.person_images_id,
        }

    def get_recent_attendance(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.is_authenticated():
            return []
        try:
            records = self.worksheet.get_all_records()
            result = []
            for record in records[-limit:]:
                result.append({
                    "Date": record.get("Date", ""),
                    "Time": record.get("Time", ""),
                    "Camera Name": record.get("Camera Name", ""),
                    "Person Name": record.get("Person Name", ""),
                    "Face Image URL": record.get("Face Image URL", ""),
                })
            return result
        except Exception as e:
            print(f"❌ Error getting attendance: {e}")
            return []


google_api = GoogleAPI()