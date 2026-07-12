"""
extract_signs.py
Extracts the 31 selected GSL signs from the raw dataset
into data/processed/selected_signs/ for further processing.
"""

import os
import shutil

# Define paths
RAW_DATA_PATH = os.path.join("data", "raw", "GSL_openpose_data", "GSL_openpose_data")
OUTPUT_PATH = os.path.join("data", "processed", "selected_signs")

# 31 selected signs
SELECTED_SIGNS = [
    # Greetings & Pleasantries
    "WELCOME",
    "THANK_YOU",
    "PLEASE",
    "SORRY",
    "MORNING",

    # Needs & Actions
    "WANT",
    "EAT",
    "DRINK",
    "WATER",
    "COME",
    "WALK",
    "STOP",
    "WAIT",
    "SEND",
    "LOVE",
    "NAME",

    # People
    "MOTHER",
    "TEACHER",
    "DOCTOR",
    "CHILD",

    # Descriptions
    "BIG_OR_GREAT",
    "SMALL",
    "HOT",
    "COLD",
    "SICK",
    "HUNGRY",

    # Essentials
    "YES",
    "NO",
    "I_OR_ME",
    "YOU",
    "HOME",
]


def extract_signs():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print(f"Extracting {len(SELECTED_SIGNS)} signs from dataset...\n")

    success_count = 0
    failed = []

    for sign in SELECTED_SIGNS:
        source = os.path.join(RAW_DATA_PATH, sign)
        destination = os.path.join(OUTPUT_PATH, sign)

        if not os.path.exists(source):
            print(f"  [NOT FOUND] {sign}")
            failed.append(sign)
            continue

        # Copy the entire sign folder
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

        # Count files in the copied folder
        files = os.listdir(destination)
        mp4_count = len([f for f in files if f.endswith(".mp4")])
        json_count = len([f for f in files if f.endswith(".json")])

        print(f"  [OK] {sign} — {mp4_count} video, {json_count} keypoint files")
        success_count += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"  Successfully copied: {success_count}/{len(SELECTED_SIGNS)} signs")
    print(f"  Output location: {OUTPUT_PATH}")

    if failed:
        print(f"  Failed/missing: {', '.join(failed)}")

    # Calculate total size
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_PATH):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    extract_signs()