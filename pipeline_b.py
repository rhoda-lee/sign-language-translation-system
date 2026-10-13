"""
pipeline_b.py
Pipeline B: Speech -> Sign Language.

Listens to speech, transcribes it, removes filler (stop) words, maps each
remaining content word to a sign in the 100-word vocabulary, and plays the
corresponding sign video. Words with no matching sign are reported, not
silently dropped.

"""

import csv
import json
import time
from pathlib import Path

import cv2
import speech_recognition as sr

# ---------------------------------------------------------------------------
MANIFEST = Path("data/processed/wlasl100_manifest.csv")
LABELS = Path("data/processed/dataset/labels.json")

# common words that carry little content and usually have no distinct sign
STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "so", "do", "does",
    "did", "have", "has", "had", "it", "this", "that", "these", "those",
    "i", "me", "my", "we", "us", "will", "would", "can", "could",
}
# ---------------------------------------------------------------------------


def build_gloss_to_video():
    """One representative video per sign, from the manifest."""
    mapping = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = row["gloss"]
            if g not in mapping:
                mapping[g] = row["video_path"]
    return mapping


def play_sign(word, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / fps)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(frame, f"Sign: {word}", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 230), 2)
        cv2.imshow("Pipeline B - Speech to Sign", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    cap.release()


def main():
    vocab = set(json.loads(LABELS.read_text(encoding="utf-8")))
    gloss_to_video = build_gloss_to_video()

    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Calibrating microphone for background noise, stay quiet a second...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Ready.\n")

    # ---------- performance counters ----------
    transcription_latencies = []    # ms, Google API time
    speech_to_sign_latencies = []   # ms, utterance end -> first playback

    while True:
        cmd = input("Press Enter to speak, or type q then Enter to quit: ").strip()
        if cmd.lower() == "q":
            break

        print("Listening... speak now.")
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
            t_utt_end = time.perf_counter()        # utterance captured
            text = recognizer.recognize_google(audio)
            t_text = time.perf_counter()           # transcription returned
        except sr.WaitTimeoutError:
            print("  No speech detected, try again.\n")
            continue
        except sr.UnknownValueError:
            print("  Could not understand the audio, try again.\n")
            continue
        except sr.RequestError:
            print("  Speech service unreachable. Check your internet connection.\n")
            continue

        print(f"  You said: \"{text}\"")

        # tokenise and drop stop words
        tokens = [t.strip(".,!?;:").lower() for t in text.split()]
        content = [t for t in tokens if t and t not in STOPWORDS]

        to_play = []
        missing = []
        for word in content:
            if word in vocab and word in gloss_to_video:
                to_play.append(word)
            else:
                missing.append(word)

        if to_play:
            print(f"  Signing: {', '.join(to_play)}")
        if missing:
            print(f"  No sign available for: {', '.join(missing)}")

        # report transcription latency regardless of whether we can sign anything
        trans_ms = (t_text - t_utt_end) * 1000.0
        transcription_latencies.append(trans_ms)
        print(f"  [latency] transcription (Google): {trans_ms:.0f} ms")

        if not to_play:
            print("  Nothing to sign from that sentence.\n")
            continue

        # time from end of utterance to the first sign video starting to play
        t_play = time.perf_counter()
        s2s_ms = (t_play - t_utt_end) * 1000.0
        speech_to_sign_latencies.append(s2s_ms)
        print(f"  [latency] utterance end -> first sign playback: {s2s_ms:.0f} ms")

        for word in to_play:
            play_sign(word, gloss_to_video[word])
        cv2.destroyAllWindows()
        print()

    cv2.destroyAllWindows()

    # ---------- performance summary ----------
    print("\n" + "=" * 52)
    print("PIPELINE B - REAL-TIME PERFORMANCE")
    print("=" * 52)
    if transcription_latencies:
        m = sum(transcription_latencies) / len(transcription_latencies)
        print(f"  Mean transcription latency        : {m:.0f} ms "
              f"(n={len(transcription_latencies)})")
    if speech_to_sign_latencies:
        m = sum(speech_to_sign_latencies) / len(speech_to_sign_latencies)
        print(f"  Mean utterance-end -> playback    : {m:.0f} ms "
              f"(n={len(speech_to_sign_latencies)})")
    if not transcription_latencies:
        print("  No utterances were recorded.")
    print("  (Transcription latency depends on your internet connection,")
    print("   not on the system itself; report it with that caveat.)")
    print("=" * 52)


if __name__ == "__main__":
    main()