"""
app_gui.py
Desktop GUI (Tkinter) for the bidirectional sign language translation system.

One window, two modes:
    - Sign to Speech : webcam -> landmarks -> LSTM -> spoken word
    - Speech to Sign : microphone -> text -> sign video playback

The camera / recognition work runs on a background thread and hands finished
frames to the GUI through a queue. Only the main thread touches Tkinter widgets,
which is what keeps the window from freezing.

Reuses pipeline_a.py and pipeline_b.py (no logic is duplicated).

Run from the project root with your venv active:
    python app_gui.py
"""

import sys
import time
import json
import queue
import threading
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk

import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import speech_recognition as sr

import pipeline_a as pa          # constants, normalisation, tts queue (speak_q)
import pipeline_b as pb          # gloss->video mapping, stopwords

# Base (unscaled, at 96 dpi) video size. Everything is multiplied by the real
# display scale factor at runtime so the UI is crisp and correctly sized on
# high-DPI screens.
VIDEO_W, VIDEO_H = 560, 420

# ---------------------------------------------------------------------------
# Colour palette: white surfaces, one modern blue, restrained accents.
# ---------------------------------------------------------------------------
BG        = "#EEF2F9"   # window backdrop (soft grey-blue)
CARD      = "#FFFFFF"   # main content card
INK       = "#0F172A"   # primary text
MUTED     = "#64748B"   # secondary text
BORDER    = "#E2E8F0"   # hairline borders
BLUE      = "#2563EB"   # primary
BLUE_DARK = "#1D4ED8"   # primary hover
BLUE_TINT = "#EFF4FF"   # light blue fill (inactive pill hover)
GREEN     = "#16A34A"   # "listening / active" state
RED       = "#DC2626"   # stop
RED_TINT  = "#FEF2F2"   # stop hover fill
GHOST_HOV = "#F1F5F9"   # neutral hover fill
VIDEO_BG  = "#0B1220"   # video panel background


def enable_dpi_awareness():
    """Tell Windows this process handles its own scaling, so text stays sharp.

    Without this the app is DPI-unaware and Windows bitmap-stretches the whole
    window on scaled displays (125% / 150% / 200%), which blurs every glyph.
    Must be called before the first Tk window is created. No-op off Windows.
    """
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()     # system aware (older)
        except Exception:
            pass


class RoundedButton(tk.Canvas):
    """A flat, rounded, hover-aware button drawn on a Canvas.

    Tkinter's native button cannot do rounded corners or flat hover states,
    so each button is a small canvas with a smooth rounded-rectangle shape and
    a text label. Colours can be re-set at runtime (used for the active-mode
    highlight on the two mode buttons). All sizes passed in are already in
    scaled (physical) pixels.
    """

    def __init__(self, parent, text, command=None, width=200, height=52,
                 radius=16, fill=BLUE, fg="#FFFFFF", hover=BLUE_DARK,
                 outline="", outline_width=0, font=None, parent_bg=CARD):
        super().__init__(parent, width=width, height=height, bg=parent_bg,
                         highlightthickness=0, bd=0, relief="flat")
        self.command = command
        self.fill, self.hover, self.fg = fill, hover, fg
        self.outline, self.outline_width = outline, outline_width
        self._font = font or ("Segoe UI", -16, "bold")
        # NOTE: never assign self._w / self._h here. In Tkinter self._w is the
        # widget's internal Tcl path name; overwriting it breaks every canvas call.

        self._shape = self._round_rect(1, 1, width - 1, height - 1, radius,
                                       fill=fill, outline=outline,
                                       width=outline_width)
        self._label = self.create_text(width // 2, height // 2, text=text,
                                       fill=fg, font=self._font)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.configure(cursor="hand2")

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
               x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
               x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _on_enter(self, _):
        self.itemconfig(self._shape, fill=self.hover)

    def _on_leave(self, _):
        self.itemconfig(self._shape, fill=self.fill)

    def _on_click(self, _):
        if self.command:
            self.command()

    def set_style(self, fill, fg, hover, outline="", outline_width=0):
        self.fill, self.fg, self.hover = fill, fg, hover
        self.outline, self.outline_width = outline, outline_width
        self.itemconfig(self._shape, fill=fill, outline=outline,
                        width=outline_width)
        self.itemconfig(self._label, fill=fg)

    def set_text(self, text):
        self.itemconfig(self._label, text=text)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Sign Language Translation System")
        root.configure(bg=BG)

        # --- work out the real display scale factor (1.0 at 96 dpi) ---
        root.update_idletasks()
        try:
            self.S = max(1.0, root.winfo_fpixels("1i") / 96.0)
        except Exception:
            self.S = 1.0

        self.disp_w = self.px(VIDEO_W)
        self.disp_h = self.px(VIDEO_H)

        # --- load model, labels, hand detector (once) ---
        self.labels = json.loads(pa.LABELS.read_text(encoding="utf-8"))
        self.model = tf.keras.models.load_model(pa.MODEL_PATH, compile=False)
        base = python.BaseOptions(model_asset_path=str(pa.HAND_MODEL))
        opts = vision.HandLandmarkerOptions(
            base_options=base, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.detector = vision.HandLandmarker.create_from_options(opts)
        self.gloss_to_video = pb.build_gloss_to_video()

        # --- shared state between worker thread and GUI ---
        self.frame_q = queue.Queue(maxsize=2)
        self.status_text = "Choose a mode to begin."
        self.running = False
        self.mode = None
        self.worker = None

        self._build_ui()
        self._blank_panel()
        self._set_mode_buttons(active=None)
        self._center_to_content()
        self.root.after(15, self._update_display)

    # ---------------- scaling helpers ----------------
    def px(self, v):
        """Scale a base (96-dpi) pixel value to the current display."""
        return int(round(v * self.S))

    def ft(self, size, weight="normal"):
        """A Segoe UI font tuple in PIXELS (negative size), so it is crisp and
        does not get double-scaled by Tk's own point scaling."""
        return ("Segoe UI", -self.px(size), weight)

    def _center_to_content(self):
        self.root.update_idletasks()
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        self.root.minsize(w, h)
        x = (self.root.winfo_screenwidth() - w) // 2
        y = max(0, (self.root.winfo_screenheight() - h) // 2 - 20)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ---------------- UI ----------------
    def _build_ui(self):
        # The window centres a fixed-width white card. Weighting row/col 0 and
        # placing the card there WITHOUT sticky keeps it centred at any size,
        # including maximised / fullscreen, with the backdrop showing around it.
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        card = tk.Frame(self.root, bg=CARD)
        card.grid(row=0, column=0, padx=self.px(24), pady=self.px(24))
        self.card = card

        pad = tk.Frame(card, bg=CARD)
        pad.pack(padx=self.px(30), pady=self.px(28))

        # ---- header ----
        header = tk.Frame(pad, bg=CARD)
        header.pack(fill="x")

        m = self.px(34)
        mark = tk.Canvas(header, width=m, height=m, bg=CARD,
                         highlightthickness=0, bd=0)
        mark.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, self.px(12)))
        self._round_on(mark, self.px(2), self.px(2), m - self.px(2),
                       m - self.px(2), self.px(10), BLUE)
        mark.create_text(m // 2, m // 2, text="S", fill="#FFFFFF",
                         font=self.ft(15, "bold"))

        tk.Label(header, text="Sign Language Translator", bg=CARD, fg=INK,
                 font=self.ft(20, "bold")).grid(row=0, column=1, sticky="w")
        tk.Label(header, text="Bidirectional  |  Sign to Speech and Speech to Sign",
                 bg=CARD, fg=MUTED, font=self.ft(11)).grid(
                 row=1, column=1, sticky="w")

        # thin divider
        tk.Frame(pad, bg=BORDER, height=1).pack(
            fill="x", pady=(self.px(18), self.px(18)))

        # ---- mode switcher (segmented) ----
        gap = self.px(12)
        bw = (self.disp_w - gap) // 2                 # two buttons span the video
        bh = self.px(54)
        modes = tk.Frame(pad, bg=CARD)
        modes.pack()

        self.btn_a = RoundedButton(
            modes, "Sign  ->  Speech", command=self.start_sign_to_speech,
            width=bw, height=bh, radius=self.px(16), font=self.ft(12, "bold"))
        self.btn_a.grid(row=0, column=0, padx=(0, gap // 2))

        self.btn_b = RoundedButton(
            modes, "Speech  ->  Sign", command=self.start_speech_to_sign,
            width=bw, height=bh, radius=self.px(16), font=self.ft(12, "bold"))
        self.btn_b.grid(row=0, column=1, padx=(gap // 2, 0))

        # ---- video card (fixed PIXEL size so it never blows up the layout) ----
        wrap = tk.Frame(pad, bg=BORDER, width=self.disp_w + 2,
                        height=self.disp_h + 2)
        wrap.pack(pady=(self.px(20), self.px(16)))
        wrap.pack_propagate(False)                    # keep the fixed pixel size
        self.panel = tk.Label(wrap, bg=VIDEO_BG, fg="#94A3B8",
                              font=self.ft(12), justify="center")
        self.panel.pack(fill="both", expand=True, padx=1, pady=1)

        # ---- status row ----
        status = tk.Frame(pad, bg=CARD)
        status.pack(fill="x", pady=(0, self.px(18)))
        dsz = self.px(12)
        self.dot = tk.Canvas(status, width=dsz, height=dsz, bg=CARD,
                             highlightthickness=0, bd=0)
        self.dot.pack(side="left", padx=(self.px(2), self.px(8)))
        self._dot_id = self.dot.create_oval(self.px(2), self.px(2),
                                            dsz - self.px(2), dsz - self.px(2),
                                            fill=MUTED, outline="")
        self.status_var = tk.StringVar(value=self.status_text)
        tk.Label(status, textvariable=self.status_var, bg=CARD, fg=MUTED,
                 font=self.ft(11), anchor="w", justify="left").pack(
                 side="left", fill="x", expand=True)

        # ---- footer controls ----
        footer = tk.Frame(pad, bg=CARD)
        footer.pack(fill="x")
        self.btn_stop = RoundedButton(
            footer, "Stop", command=self.stop_current,
            width=self.px(150), height=self.px(46), radius=self.px(13),
            fill=CARD, fg=RED, hover=RED_TINT, outline=RED, outline_width=1.4,
            font=self.ft(11, "bold"))
        self.btn_stop.pack(side="left")

        self.btn_quit = RoundedButton(
            footer, "Quit", command=self.quit,
            width=self.px(130), height=self.px(46), radius=self.px(13),
            fill=CARD, fg=MUTED, hover=GHOST_HOV, outline=BORDER,
            outline_width=1.4, font=self.ft(11, "bold"))
        self.btn_quit.pack(side="right")

    def _round_on(self, canvas, x1, y1, x2, y2, r, fill):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
               x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
               x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return canvas.create_polygon(pts, smooth=True, fill=fill, outline="")

    def _set_mode_buttons(self, active):
        """active: 'A', 'B', or None. Highlights the running mode in blue."""
        active_style = dict(fill=BLUE, fg="#FFFFFF", hover=BLUE_DARK)
        idle_style = dict(fill=CARD, fg=BLUE, hover=BLUE_TINT,
                          outline=BLUE, outline_width=1.4)
        self.btn_a.set_style(**(active_style if active == "A" else idle_style))
        self.btn_b.set_style(**(active_style if active == "B" else idle_style))

    def _set_dot(self, colour):
        self.dot.itemconfig(self._dot_id, fill=colour)

    def _blank_panel(self):
        self.panel.config(
            image="", text="Camera or sign video appears here.\n"
                           "Pick a mode above to start.",
            fg="#94A3B8", bg=VIDEO_BG)
        self.panel.image = None

    def _show(self, pil_img):
        tkimg = ImageTk.PhotoImage(pil_img)
        self.panel.configure(image=tkimg, text="")
        self.panel.image = tkimg      # keep a reference

    def _update_display(self):
        try:
            frame_rgb = self.frame_q.get_nowait()
            img = Image.fromarray(frame_rgb).resize((self.disp_w, self.disp_h))
            self._show(img)
        except queue.Empty:
            pass
        self.status_var.set(self.status_text)
        self.root.after(15, self._update_display)

    def _push(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            self.frame_q.put_nowait(rgb)
        except queue.Full:
            pass

    # ---------------- mode control ----------------
    def stop_current(self):
        self.running = False
        if self.worker is not None:
            self.worker.join(timeout=2)
        self.worker = None
        self.mode = None
        self.status_text = "Stopped. Choose a mode."
        self._set_mode_buttons(active=None)
        self._set_dot(MUTED)
        self._blank_panel()

    def start_sign_to_speech(self):
        self.stop_current()
        self.running = True
        self.mode = "A"
        self._set_mode_buttons(active="A")
        self._set_dot(BLUE)
        self.worker = threading.Thread(target=self.loop_A, daemon=True)
        self.worker.start()

    def start_speech_to_sign(self):
        self.stop_current()
        self.running = True
        self.mode = "B"
        self._set_mode_buttons(active="B")
        self._set_dot(GREEN)
        self.worker = threading.Thread(target=self.loop_B, daemon=True)
        self.worker.start()

    # ---------------- Pipeline A loop ----------------
    def loop_A(self):
        cap = cv2.VideoCapture(0)
        buffer = deque(maxlen=pa.SEQ_LEN)
        no_hand, last_sign, stable, cooldown, spoken = 0, None, 0, 0, ""

        while self.running and self.mode == "A":
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_img)

            status = "Sign to the camera."
            if result.hand_landmarks:
                no_hand = 0
                pa.draw_landmarks(frame, result)
                vec, found = pa.landmarks_to_vector(result)
                if found:
                    buffer.append(pa.normalise_frame(vec))
                    status = f"Reading... {len(buffer)}/{pa.SEQ_LEN}"
            else:
                no_hand += 1
                if no_hand >= pa.NO_HAND_RESET:
                    buffer.clear(); stable = 0; last_sign = None

            if cooldown > 0:
                cooldown -= 1

            if len(buffer) == pa.SEQ_LEN:
                seq = np.expand_dims(np.array(buffer, dtype=np.float32), 0)
                probs = self.model.predict(seq, verbose=0)[0]
                top = int(np.argmax(probs)); conf = float(probs[top])
                if conf >= pa.CONF_THRESHOLD:
                    status = f"{self.labels[top]} ({conf*100:.0f}%)"
                    if top == last_sign:
                        stable += 1
                    else:
                        stable = 1; last_sign = top
                    if stable >= pa.STABLE_COUNT and cooldown == 0:
                        spoken = self.labels[top]
                        pa.speak_q.put(spoken)
                        cooldown = pa.COOLDOWN_FRAMES; stable = 0

            if spoken:
                cv2.putText(frame, f"Spoken: {spoken}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 230), 2)
            self.status_text = status
            self._push(frame)

        cap.release()

    # ---------------- Pipeline B loop ----------------
    def loop_B(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        self.status_text = "Calibrating microphone..."
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        while self.running and self.mode == "B":
            self.status_text = "Listening... speak now."
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=4, phrase_time_limit=6)
                text = recognizer.recognize_google(audio)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                self.status_text = "Could not understand. Try again."
                time.sleep(1); continue
            except sr.RequestError:
                self.status_text = "Speech service unreachable (needs internet)."
                time.sleep(2); continue

            tokens = [t.strip(".,!?;:").lower() for t in text.split()]
            content = [t for t in tokens if t and t not in pb.STOPWORDS]
            to_play = [w for w in content if w in self.gloss_to_video]
            missing = [w for w in content if w not in self.gloss_to_video]

            note = f'You said: "{text}"'
            if missing:
                note += f"   (no sign for: {', '.join(missing)})"
            self.status_text = note

            for word in to_play:
                if not (self.running and self.mode == "B"):
                    break
                self._play_sign(word)

    def _play_sign(self, word):
        cap = cv2.VideoCapture(self.gloss_to_video[word])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        while self.running and self.mode == "B":
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, f"Sign: {word}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 230), 2)
            self.status_text = f"Signing: {word}"
            self._push(frame)
            time.sleep(1.0 / fps)
        cap.release()

    def quit(self):
        self.running = False
        if self.worker is not None:
            self.worker.join(timeout=2)
        self.root.destroy()


if __name__ == "__main__":
    enable_dpi_awareness()
    root = tk.Tk()
    App(root)
    root.mainloop()