#!/usr/bin/env python3
import sys
import os

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google_api import google_api


def share_folder_with_gmail(email: str):
    """
    One-time helper to give your Gmail access
    to the Service Account's Person Images folder.
    """
    google_api.drive_service.permissions().create(
        fileId=google_api.person_images_id,
        body={
            "type": "user",
            "role": "writer",
            "emailAddress": email,
        }
    ).execute()
    print(f"✅ Folder shared with {email}")


def main():
    print("=" * 60)
    print("🔍 GOOGLE DRIVE PERMISSIONS & TRAINING TEST")
    print("=" * 60)

    if not google_api.is_authenticated():
        print("❌ Google API not authenticated")
        print("   Check service_account.json")
        return

    print("✅ Google API is authenticated!")
    print(f"📁 Person Images Folder ID: {google_api.person_images_id}")
    print(f"🤖 Service Account Email: {google_api.creds.service_account_email}")

    # 🔹 List training people
    people = google_api.list_training_people()
    print(f"\n👥 Training people found: {len(people)}")

    for person, images in people.items():
        print(f"  - {person}: {len(images)} images")

    # 🔹 If no people found, guide user
    if len(people) == 0:
        print("\n📋 ACTION REQUIRED:")
        print(f"1. Open this link in browser:")
        print(f"   https://drive.google.com/drive/folders/{google_api.person_images_id}")
        print("2. If it asks for access → run this script ONCE with sharing enabled")
        print("3. Upload person folders (AKASH, ASWIN K, etc.) there")

    print("\n✅ Test complete")


if __name__ == "__main__":
    # 🔴 CHANGE THIS TO YOUR EMAIL ONLY ONCE
    SHARE_WITH_GMAIL = True
    YOUR_GMAIL = "hashtech113@gmail.com"

    if SHARE_WITH_GMAIL:
        share_folder_with_gmail(YOUR_GMAIL)

    main()
