import os
import gspread
from datetime import datetime
from typing import List, Dict, Any
from google.oauth2.service_account import Credentials

SERVICE_ACCOUNT_FILE = "service_account.json"
SHEET_NAME = "AI Attendance Log"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


class GoogleAPI:
    def __init__(self):
        self.creds = None
        self.gc = None
        self.worksheet = None
        
        self.authenticate()

    def authenticate(self):
        """Authenticate with Google Sheets only"""
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            print("❌ No service_account.json found. Google Sheets logging disabled.")
            return
            
        try:
            self.creds = Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            self.gc = gspread.authorize(self.creds)
            self._setup_sheet()
            print("✅ Google Sheets authentication ready")
        except Exception as e:
            print(f"❌ Google Sheets auth failed: {e}")

    def _setup_sheet(self):
        try:
            try:
                spreadsheet = self.gc.open(SHEET_NAME)
            except gspread.SpreadsheetNotFound:
                spreadsheet = self.gc.create(SHEET_NAME)

            self.worksheet = spreadsheet.sheet1

            required_headers = ["Date", "Time", "Camera Name", "Person Name", "Face Image Path"]
            headers = self.worksheet.row_values(1)
            if headers != required_headers:
                self.worksheet.clear()
                self.worksheet.append_row(required_headers)

            print(f"✅ Connected to Google Sheet: {SHEET_NAME}")
        except Exception as e:
            print(f"❌ Failed to setup sheet: {e}")
            self.worksheet = None

    def is_authenticated(self) -> bool:
        return all([self.creds, self.gc, self.worksheet])

    def log_attendance(self, camera_name: str, person_name: str, image_path: str) -> bool:
        """Log attendance to Google Sheets"""
        if not self.is_authenticated():
            return False
            
        try:
            now = datetime.now()
            self.worksheet.append_row([
                now.strftime("%d-%m-%Y"),
                now.strftime("%H:%M:%S"),
                camera_name,
                person_name,
                image_path  # This is now a clickable HTTP URL
            ])
            print(f"📝 Logged to Sheet: {person_name} @ {camera_name}")
            return True
        except Exception as e:
            print(f"❌ Error logging to Sheet: {e}")
            return False

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
                    "Face Image Path": record.get("Face Image Path", ""),
                })
            return result
        except Exception as e:
            print(f"❌ Error getting attendance: {e}")
            return []

    def get_drive_info(self) -> Dict[str, Any]:
        return {
            "authenticated": self.is_authenticated(),
            "mode": "sheets_only",
            "sheet_name": SHEET_NAME
        }


google_api = GoogleAPI()