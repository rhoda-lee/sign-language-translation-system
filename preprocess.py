"""
preprocess.py
Step 3 of the pipeline: turn the raw per-clip landmark arrays into a clean,
model-ready dataset.

For every clip it:
  1. drops frames where no hand was detected (the rest frames at the start/end)
  2. normalises each hand (wrist to origin, then scale by hand size) so that
     position in the frame and distance from the camera stop mattering
  3. pads or samples the clip to a fixed length of 30 frames
  4. labels it and collects it

Then it splits everything 70/15/15 (train/val/test), stratified by sign.

"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
LANDMARKS_DIR = Path("data/processed/landmarks")
OUT_DIR = Path("data/processed/dataset")
SEQ_LEN = 30
FEATURES_PER_HAND = 63          # 21 landmarks x 3
MIN_FRAMES_AFTER_TRIM = 5       # clips with fewer usable frames are skipped
# ---------------------------------------------------------------------------


def frame_is_empty(frame):
    """True if no hand was detected in this frame (all 126 values are zero)."""
    return not np.any(frame)


def trim_rest_frames(clip):
    """Keep only frames that contain at least one detected hand."""
    keep = [f for f in clip if not frame_is_empty(f)]
    return np.array(keep, dtype=np.float32) if keep else np.empty((0, clip.shape[1]), np.float32)


def normalise_hand(block):
    """block: 63 values for one hand. Wrist to origin, then scale by hand size."""
    if not np.any(block):
        return block  # absent hand stays all-zero
    pts = block.reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist                      # translate: wrist at origin
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts = pts / scale                  # scale: largest reach = 1
    return pts.reshape(-1)


def normalise_frame(frame):
    left = normalise_hand(frame[:FEATURES_PER_HAND])
    right = normalise_hand(frame[FEATURES_PER_HAND:])
    return np.concatenate([left, right])


def fix_length(clip, length=SEQ_LEN):
    """Pad short clips (repeat last frame) or uniformly sample long ones."""
    n = len(clip)
    if n == length:
        return clip
    if n < length:
        pad = np.repeat(clip[-1:], length - n, axis=0)
        return np.concatenate([clip, pad], axis=0)
    idx = np.linspace(0, n - 1, length).astype(int)
    return clip[idx]


def main():
    sign_dirs = sorted(d for d in LANDMARKS_DIR.iterdir() if d.is_dir())
    glosses = [d.name for d in sign_dirs]
    label_to_idx = {g: i for i, g in enumerate(glosses)}

    X, y = [], []
    kept_per_sign = defaultdict(int)
    skipped = 0

    for d in sign_dirs:
        for npy in d.glob("*.npy"):
            clip = np.load(npy)
            clip = trim_rest_frames(clip)
            if len(clip) < MIN_FRAMES_AFTER_TRIM:
                skipped += 1
                continue
            clip = np.array([normalise_frame(f) for f in clip], dtype=np.float32)
            clip = fix_length(clip)
            X.append(clip)
            y.append(label_to_idx[d.name])
            kept_per_sign[d.name] += 1

    X = np.array(X, dtype=np.float32)      # (N, 30, 126)
    y = np.array(y, dtype=np.int64)

    # 70/15/15. First split is stratified (every class has >=5 samples, so this
    # is safe). The second split is NOT stratified, because after the first split
    # some classes have only 1 sample left in the temp set, which a stratified
    # split cannot divide. A plain random split avoids that while staying roughly
    # balanced across the ~150 temp samples.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X_train.npy", X_tr)
    np.save(OUT_DIR / "y_train.npy", y_tr)
    np.save(OUT_DIR / "X_val.npy", X_val)
    np.save(OUT_DIR / "y_val.npy", y_val)
    np.save(OUT_DIR / "X_test.npy", X_test)
    np.save(OUT_DIR / "y_test.npy", y_test)
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(glosses, f, indent=2)

    counts = [kept_per_sign[g] for g in glosses]
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Classes: {len(glosses)}")
    print(f"  Usable sequences: {len(X)}   (skipped {skipped} too-short clips)")
    print(f"  Shape of each sequence: {X.shape[1]} frames x {X.shape[2]} features")
    print(f"  Train: {len(X_tr)}   Val: {len(X_val)}   Test: {len(X_test)}")
    print(f"  Samples per sign: min {min(counts)}, max {max(counts)}, "
          f"mean {np.mean(counts):.1f}")
    print(f"  Saved to: {OUT_DIR}")

    thin = sorted([(g, kept_per_sign[g]) for g in glosses], key=lambda x: x[1])[:8]
    print("\n  Fewest-sample signs after preprocessing:")
    for g, n in thin:
        print(f"    {g}: {n}")

    lost = [g for g in glosses if kept_per_sign[g] == 0]
    if lost:
        print(f"\n  WARNING: {len(lost)} signs lost all samples: {lost}")


if __name__ == "__main__":
    main()