"""
build_manifest.py
Step 1 of the WLASL pipeline: build an inventory, do NOT extract anything yet.

Takes the WLASL100 subset (the first 100 glosses in WLASL_v0.3.json), cross-
references each listed video against the videos you actually have on disk
(many WLASL clips fail to download because the source links expire), and
reports how many usable samples survive per sign.

Output:
  - data/processed/wlasl100_manifest.csv   (one row per available video)
  - a printed report of sample counts and any signs left too thin to use

"""

import os
import csv
import json
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG — adjust these two paths only if your files live somewhere else.
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")          # folder that contains WLASL_v0.3.json
JSON_NAME = "WLASL_v0.3.json"
NUM_GLOSSES = 100                   # WLASL100
OUTPUT_CSV = Path("data/processed/wlasl100_manifest.csv")

# How few samples is "too thin to rely on" (just for the warning list)
LOW_SAMPLE_THRESHOLD = 5

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".swf"}


def find_file(root: Path, name: str):
    """Locate a file by name anywhere under root."""
    direct = root / name
    if direct.exists():
        return direct
    for p in root.rglob(name):
        return p
    return None


def find_videos_dir(root: Path):
    """Find a folder named 'videos' under root (case-insensitive)."""
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "videos":
            return p
    return None


def main():
    # --- locate the JSON ---
    json_path = find_file(RAW_DIR, JSON_NAME)
    if json_path is None:
        print(f"ERROR: could not find {JSON_NAME} under {RAW_DIR}/")
        print("Edit RAW_DIR / JSON_NAME at the top of this script to match your layout.")
        return
    print(f"Using metadata: {json_path}")

    # --- locate the videos folder ---
    videos_dir = find_videos_dir(RAW_DIR)
    if videos_dir is None:
        print(f"ERROR: could not find a 'videos' folder under {RAW_DIR}/")
        return
    print(f"Using videos folder: {videos_dir}")

    # --- index every video file actually present, by its stem (the video_id) ---
    available = {}
    for p in videos_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            available[p.stem] = p
    print(f"Videos present on disk: {len(available)}\n")

    # --- load metadata and take the first 100 glosses (WLASL100) ---
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected = data[:NUM_GLOSSES]

    rows = []
    per_sign_available = defaultdict(int)
    per_sign_listed = defaultdict(int)

    for class_index, entry in enumerate(selected):
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            vid = str(inst["video_id"])
            per_sign_listed[gloss] += 1
            if vid in available:
                per_sign_available[gloss] += 1
                bbox = inst.get("bbox", [0, 0, 0, 0])
                rows.append({
                    "gloss": gloss,
                    "class_index": class_index,
                    "video_id": vid,
                    "split": inst.get("split", ""),
                    "frame_start": inst.get("frame_start", 1),
                    "frame_end": inst.get("frame_end", -1),
                    "fps": inst.get("fps", ""),
                    "signer_id": inst.get("signer_id", ""),
                    "bbox_x1": bbox[0], "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2], "bbox_y2": bbox[3],
                    "video_path": str(available[vid]),
                })

    # --- write the manifest ---
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- report ---
    counts = [per_sign_available[g["gloss"]] for g in selected]
    total_available = sum(counts)
    total_listed = sum(per_sign_listed.values())

    print("=" * 60)
    print("WLASL100 INVENTORY")
    print("=" * 60)
    print(f"  Signs: {NUM_GLOSSES}")
    print(f"  Video instances listed in metadata: {total_listed}")
    print(f"  Video instances available on disk:  {total_available}")
    print(f"  Coverage: {total_available / total_listed * 100:.1f}%")
    print(
        f"  Average usable samples per sign: {total_available / NUM_GLOSSES:.1f}")
    print(f"  Most samples: {max(counts)}   Fewest samples: {min(counts)}")
    print(f"  Manifest written to: {OUTPUT_CSV}")

    # signs that came out too thin
    thin = sorted(
        [(g["gloss"], per_sign_available[g["gloss"]]) for g in selected],
        key=lambda x: x[1],
    )
    low = [t for t in thin if t[1] < LOW_SAMPLE_THRESHOLD]
    print(
        f"\n  Signs with fewer than {LOW_SAMPLE_THRESHOLD} usable samples: {len(low)}")
    for gloss, n in low:
        print(f"    {gloss}: {n}")

    # show the 10 weakest and 10 strongest for a feel of the spread
    print("\n  Fewest-sample signs:")
    for gloss, n in thin[:10]:
        print(f"    {gloss}: {n}")
    print("\n  Most-sample signs:")
    for gloss, n in thin[-10:]:
        print(f"    {gloss}: {n}")


if __name__ == "__main__":
    main()