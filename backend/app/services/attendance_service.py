import os
import urllib.parse
from app.core.config import BASE_URL, IMAGES_DIR
from app.integrations.google_api import google_api

def build_media_url(abs_path: str) -> str:
    rel = os.path.relpath(abs_path, IMAGES_DIR).replace(os.sep, "/")
    enc = urllib.parse.quote(rel)
    return f"{BASE_URL}/media/{enc}"

def log_to_sheet(camera_name: str, person_name: str, image_abs_path: str):
    if not google_api.is_authenticated():
        return False
    url = build_media_url(image_abs_path)
    return google_api.log_attendance(camera_name, person_name, url)
