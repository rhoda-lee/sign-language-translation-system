"""
pipeline_a.py
Pipeline A: real-time Sign Language -> Speech.

Opens the webcam, extracts hand landmarks with MediaPipe, feeds a rolling
window of 30 frames into the trained LSTM, and speaks the recognised sign.

The frame normalisation here is IDENTICAL to preprocess.py. It must be, or the
model would be fed data that looks different from what it trained on.

"""

import os
import json
import time
import queue
import threading
import numpy as np
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import pyttsx3
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/saved/sign_lstm.keras")
HAND_MODEL = Path("models/hand_landmarker.task")
LABELS = Path("data/processed/dataset/labels.json")

SEQ_LEN = 30
FEATURES_PER_HAND = 63
CONF_THRESHOLD = 0.50      # minimum softmax probability to consider a prediction
STABLE_COUNT = 6           # same top sign this many frames in a row before speaking
COOLDOWN_FRAMES = 25       # frames to wait after speaking before accepting another
NO_HAND_RESET = 12         # consecutive hand-less frames that clear the buffer
CAM_INDEX = 0
# ---------------------------------------------------------------------------


# ---------- text to speech, on its own thread so video never freezes ----------
speak_q = queue.Queue()
sign_speech_latencies = []   # collected recognition -> speech-start times (ms)

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    while True:
        item = speak_q.get()
        if item is None:
            break
        # Accept either a plain word (e.g. from the GUI) or a (word, timestamp)
        # tuple. The tuple records how long from the moment the sign was
        # recognised to the moment the TTS thread picks it up (reported once in
        # the summary; it is effectively zero because the handoff is instant).
        if isinstance(item, tuple):
            word, t_accept = item
            sign_speech_latencies.append((time.perf_counter() - t_accept) * 1000.0)
        else:
            word = item
        engine.say(word)
        engine.runAndWait()

threading.Thread(target=tts_worker, daemon=True).start()


# ---------- normalisation (must match preprocess.py exactly) ----------
def normalise_hand(block):
    if not np.any(block):
        return block
    pts = block.reshape(21, 3)
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts = pts / scale
    return pts.reshape(-1)


def normalise_frame(frame):
    return np.concatenate([
        normalise_hand(frame[:FEATURES_PER_HAND]),
        normalise_hand(frame[FEATURES_PER_HAND:]),
    ])


def landmarks_to_vector(result):
    """Build the 126-value frame vector, left hand first then right."""
    vec = np.zeros(FEATURES_PER_HAND * 2, dtype=np.float32)
    found = False
    for h_idx, hand in enumerate(result.hand_landmarks):
        if h_idx >= 2:
            break
        found = True
        handed = result.handedness[h_idx][0].category_name
        offset = 0 if handed == "Left" else FEATURES_PER_HAND
        for lm_idx, lm in enumerate(hand):
            b = offset + lm_idx * 3
            vec[b] = lm.x
            vec[b + 1] = lm.y
            vec[b + 2] = lm.z
    return vec, found


def draw_landmarks(frame, result):
    h, w = frame.shape[:2]
    for hand in result.hand_landmarks:
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 200, 0), -1)


def print_perf_summary(t_start, n_frames, infer_times):
    """Print the real-time performance figures (called on every exit path)."""
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 54)
    print("PIPELINE A - REAL-TIME PERFORMANCE")
    print("=" * 54)
    print(f"  Frames processed : {n_frames}")
    print(f"  Elapsed time     : {elapsed:.1f} s")
    if elapsed > 0 and n_frames > 0:
        fps = n_frames / elapsed
        print(f"  Sustained FPS    : {fps:.1f}")
    else:
        fps = 0.0
    if infer_times:
        mean_inf = 1000.0 * sum(infer_times) / len(infer_times)
        print(f"  Mean inference   : {mean_inf:.0f} ms per 30-frame window "
              f"({len(infer_times)} windows)")
    mean_lat = (sum(sign_speech_latencies) / len(sign_speech_latencies)
                if sign_speech_latencies else 0.0)
    print(f"  Recognition->speech dispatch : {mean_lat:.1f} ms "
          f"(mean of {len(sign_speech_latencies)}; the TTS handoff is instant)")
    if fps > 0:
        confirm_ms = 1000.0 * STABLE_COUNT / fps
        print(f"  Est. end-to-end (finish signing -> speech): "
              f"{confirm_ms + mean_lat:.0f} ms "
              f"(= {STABLE_COUNT} frames / {fps:.1f} FPS + ~0 dispatch)")
    print("=" * 54)


def main():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    base = python.BaseOptions(model_asset_path=str(HAND_MODEL))
    opts = vision.HandLandmarkerOptions(
        base_options=base, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    detector = vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: could not open the webcam.")
        return

    buffer = deque(maxlen=SEQ_LEN)
    no_hand = 0
    last_sign = None
    stable = 0
    cooldown = 0
    spoken = ""

    # ---------- performance counters ----------
    t_start = time.perf_counter()
    n_frames = 0
    infer_times = []           # seconds per 30-frame inference
    last_t = t_start           # for the live on-screen FPS readout

    print("Pipeline A running. Sign to the camera.")
    print("Quit: click the VIDEO window and press q, or press Ctrl+C here.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1
            frame = cv2.flip(frame, 1)   # mirror, feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            status = "waiting for hands"
            if result.hand_landmarks:
                no_hand = 0
                draw_landmarks(frame, result)
                vec, found = landmarks_to_vector(result)
                if found:
                    buffer.append(normalise_frame(vec))
                    status = f"buffering {len(buffer)}/{SEQ_LEN}"
            else:
                no_hand += 1
                if no_hand >= NO_HAND_RESET:
                    buffer.clear()
                    stable = 0
                    last_sign = None

            if cooldown > 0:
                cooldown -= 1

            # predict once the window is full
            if len(buffer) == SEQ_LEN:
                seq = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
                t0 = time.perf_counter()
                probs = model.predict(seq, verbose=0)[0]
                infer_times.append(time.perf_counter() - t0)
                top = int(np.argmax(probs))
                conf = float(probs[top])

                if conf >= CONF_THRESHOLD:
                    status = f"{labels[top]} ({conf*100:.0f}%)"
                    if top == last_sign:
                        stable += 1
                    else:
                        stable = 1
                        last_sign = top

                    if stable >= STABLE_COUNT and cooldown == 0:
                        spoken = labels[top]
                        # word + time accepted, for the dispatch-latency figure
                        speak_q.put((spoken, time.perf_counter()))
                        cooldown = COOLDOWN_FRAMES
                        stable = 0
                else:
                    status = "thinking..."

            # live FPS readout (instantaneous)
            now = time.perf_counter()
            inst_fps = 1.0 / (now - last_t) if now > last_t else 0.0
            last_t = now

            # on-screen overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (30, 30, 30), -1)
            cv2.putText(frame, f"Status: {status}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Last spoken: {spoken}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 230), 2)
            cv2.putText(frame, f"{inst_fps:4.1f} FPS", (frame.shape[1] - 120, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

            cv2.imshow("Pipeline A - Sign to Speech", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Writing performance summary...")
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        speak_q.put(None)
        print_perf_summary(t_start, n_frames, infer_times)


if __name__ == "__main__":
    main()