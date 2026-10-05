"""
extract_landmarks.py
Step 2 of the WLASL pipeline: run MediaPipe on every clip in the manifest,
save the landmark arrays, and report hand-detection rates per sign.

This is the check that matters: if detection is high (unlike the 10% we saw
on the GSL overlay videos), the dataset is good and we move on to training.

Reads:
    data/processed/wlasl100_manifest.csv   (from build_manifest.py)
    models/hand_landmarker.task

Writes:
    data/processed/landmarks/<gloss>/<video_id>.npy   (one array per clip)
    data/processed/detection_report.csv               (per-sign detection rates)

Run from the project root with your venv active:
    python extract_landmarks.py
"""

import os
import csv
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
MANIFEST = Path("data/processed/wlasl100_manifest.csv")
MODEL_PATH = Path("models/hand_landmarker.task")
LANDMARKS_DIR = Path("data/processed/landmarks")
REPORT_CSV = Path("data/processed/detection_report.csv")

LANDMARKS_PER_HAND = 21
COORDS = 3                       # x, y, z
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS   # 63
MAX_HANDS = 2
TOTAL_FEATURES = FEATURES_PER_HAND * MAX_HANDS    # 126
ACCEPTANCE = 80.0                # % of frames with a hand, to call a sign "good"
# ---------------------------------------------------------------------------


def create_detector():
    base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def extract_from_video(path, frame_start, frame_end, detector):
    """Return (array of shape (num_frames, 126), detection_rate_percent)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_start is 1-indexed in the metadata; -1 end means last frame
    start = max(0, (frame_start or 1) - 1)
    end = total - 1 if (frame_end is None or frame_end == -1) else min(frame_end - 1, total - 1)

    frames = []
    with_hands = 0
    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx < start:
            continue
        if idx > end:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)

        vec = np.zeros(TOTAL_FEATURES, dtype=np.float32)
        if res.hand_landmarks:
            with_hands += 1
            for h_idx, hand in enumerate(res.hand_landmarks):
                if h_idx >= MAX_HANDS:
                    break
                handed = res.handedness[h_idx][0].category_name
                offset = 0 if handed == "Left" else FEATURES_PER_HAND
                for lm_idx, lm in enumerate(hand):
                    b = offset + lm_idx * COORDS
                    vec[b] = lm.x
                    vec[b + 1] = lm.y
                    vec[b + 2] = lm.z
        frames.append(vec)

    cap.release()
    if not frames:
        return None, 0.0
    arr = np.array(frames, dtype=np.float32)
    rate = with_hands / len(frames) * 100.0
    return arr, rate


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found. Download it first (you did this before).")
        sys.exit(1)
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found. Run build_manifest.py first.")
        sys.exit(1)

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Clips to process: {len(rows)}\n")
    detector = create_detector()

    per_sign_rates = defaultdict(list)
    processed = 0
    failed = 0

    for i, row in enumerate(rows, 1):
        gloss = row["gloss"]
        vid = row["video_id"]
        path = row["video_path"]
        fs = int(row["frame_start"]) if row["frame_start"] else 1
        fe = int(row["frame_end"]) if row["frame_end"] else -1

        arr, rate = extract_from_video(path, fs, fe, detector)
        if arr is None:
            failed += 1
            print(f"  [{i}/{len(rows)}] {gloss}/{vid}  FAILED to read")
            continue

        out_dir = LANDMARKS_DIR / gloss
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{vid}.npy", arr)

        per_sign_rates[gloss].append(rate)
        processed += 1
        if i % 25 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] processed... last: {gloss}/{vid} "
                  f"{arr.shape[0]} frames, {rate:.0f}% with hands")

    detector.close()

    # --- per-sign report ---
    sign_means = {g: float(np.mean(r)) for g, r in per_sign_rates.items()}
    overall = float(np.mean(list(sign_means.values()))) if sign_means else 0.0

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gloss", "mean_detection_rate", "num_clips"])
        for g in sorted(sign_means, key=sign_means.get):
            w.writerow([g, f"{sign_means[g]:.1f}", len(per_sign_rates[g])])

    print("\n" + "=" * 60)
    print("LANDMARK EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Clips processed: {processed}   failed: {failed}")
    print(f"  Landmarks saved to: {LANDMARKS_DIR}")
    print(f"  Per-sign report:   {REPORT_CSV}")
    print(f"  Overall mean detection rate: {overall:.1f}%")

    below = [(g, r) for g, r in sign_means.items() if r < ACCEPTANCE]
    print(f"\n  Signs below {ACCEPTANCE:.0f}% detection: {len(below)} of {len(sign_means)}")
    for g, r in sorted(below, key=lambda x: x[1]):
        print(f"    {g}: {r:.1f}%")

    if overall >= ACCEPTANCE and len(below) <= 10:
        print("\n  VERDICT: dataset looks good. Ready for preprocessing and training.")
    else:
        print("\n  VERDICT: detection is weaker than hoped on some signs. ")


if __name__ == "__main__":
    main()