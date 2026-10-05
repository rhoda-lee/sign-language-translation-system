"""
train_model.py
Step 4 of the pipeline: train and evaluate the LSTM sign classifier.

  1. loads the train/val/test splits from preprocess.py
  2. augments the TRAINING data only (rotation, scaling, noise) to expand it
  3. builds the stacked LSTM described in the methodology
  4. trains with early stopping
  5. evaluates on the held-out test set: top-1 and top-5 accuracy,
     precision / recall / F1, and a confusion matrix
  6. saves the model, the training curves, and the confusion matrix

Reads:   data/processed/dataset/{X,y}_{train,val,test}.npy, labels.json
Writes:  models/saved/sign_lstm.keras
         results/training_curves.png
         results/confusion_matrix.png
         results/metrics.txt
         results/most_confused.txt

Run from the project root with your venv active:
    python train_model.py
"""

import os
import json
import numpy as np
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

# ---------------------------------------------------------------------------
DATASET = Path("data/processed/dataset")
MODEL_OUT = Path("models/saved/sign_lstm.keras")
RESULTS = Path("results")
SEQ_LEN = 30
FEATURES_PER_HAND = 63
N_AUG = 5                 # augmented copies generated per training sample
EPOCHS = 100
BATCH = 32
SEED = 42
# ---------------------------------------------------------------------------

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------- augmentation (training data only) ----------
def augment_hand(block, angle, scale, noise):
    """Rotate (xy), scale, and add noise to one present hand block (63 vals)."""
    if not np.any(block):
        return block                       # absent hand stays zero
    pts = block.reshape(21, 3).copy()
    c, s = np.cos(angle), np.sin(angle)
    xy = pts[:, :2]
    rot = np.stack([xy[:, 0] * c - xy[:, 1] * s,
                    xy[:, 0] * s + xy[:, 1] * c], axis=1)
    pts[:, :2] = rot
    pts *= scale
    pts += np.random.normal(0, noise, pts.shape).astype(np.float32)
    return pts.reshape(-1)


def augment_sequence(seq):
    """One augmented copy of a (30,126) sequence: same transform on every frame."""
    angle = np.deg2rad(np.random.uniform(-10, 10))
    scale = np.random.uniform(0.9, 1.1)
    noise = 0.02
    out = np.empty_like(seq)
    for t, frame in enumerate(seq):
        left = augment_hand(frame[:FEATURES_PER_HAND], angle, scale, noise)
        right = augment_hand(frame[FEATURES_PER_HAND:], angle, scale, noise)
        out[t] = np.concatenate([left, right])
    return out


def build_augmented(X, y):
    Xs, ys = [X], [y]
    for _ in range(N_AUG):
        Xs.append(np.array([augment_sequence(s) for s in X], dtype=np.float32))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    X_tr = np.load(DATASET / "X_train.npy")
    y_tr = np.load(DATASET / "y_train.npy")
    X_val = np.load(DATASET / "X_val.npy")
    y_val = np.load(DATASET / "y_val.npy")
    X_test = np.load(DATASET / "X_test.npy")
    y_test = np.load(DATASET / "y_test.npy")
    labels = json.loads((DATASET / "labels.json").read_text(encoding="utf-8"))
    n_classes = len(labels)

    print(f"Loaded: train {len(X_tr)}, val {len(X_val)}, test {len(X_test)}, "
          f"{n_classes} classes")

    print("Augmenting training data...")
    X_tr_aug, y_tr_aug = build_augmented(X_tr, y_tr)
    print(f"Training set after augmentation: {len(X_tr_aug)} sequences")

    # ---------- model ----------
    model = Sequential([
        Input((SEQ_LEN, X_tr.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )
    model.summary()

    # Monitor val ACCURACY, not val loss: on small data the two diverge here
    # (accuracy keeps rising while loss creeps up), and accuracy is what we report.
    stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=15,
                         restore_best_weights=True)

    history = model.fit(
        X_tr_aug, y_tr_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[stop], verbose=2,
    )

    # ---------- evaluate ----------
    test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)

    # ---------- training curves ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=150)

    # ---------- confusion matrix ----------
    cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
    fig2, ax2 = plt.subplots(figsize=(10, 9))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_title("Confusion matrix (test set)")
    ax2.set_xlabel("predicted"); ax2.set_ylabel("true")
    fig2.colorbar(im)
    fig2.tight_layout()
    fig2.savefig(RESULTS / "confusion_matrix.png", dpi=150)

    # most-confused pairs
    confused = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused.append((cm[i, j], labels[i], labels[j]))
    confused.sort(reverse=True)

    # ---------- write metrics ----------
    with open(RESULTS / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("TEST SET RESULTS\n")
        f.write(f"Top-1 accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Top-5 accuracy: {test_top5*100:.2f}%\n")
        f.write(f"Macro precision: {prec*100:.2f}%\n")
        f.write(f"Macro recall:    {rec*100:.2f}%\n")
        f.write(f"Macro F1:        {f1*100:.2f}%\n\n")
        f.write(classification_report(y_test, y_pred,
                labels=list(range(n_classes)),
                target_names=labels, zero_division=0))
    with open(RESULTS / "most_confused.txt", "w", encoding="utf-8") as f:
        for n, a, b in confused[:25]:
            f.write(f"{a}  ->  {b}   ({n})\n")

    # ---------- print summary ----------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Epochs run: {len(history.history['loss'])}")
    print(f"  Final train acc: {history.history['accuracy'][-1]*100:.1f}%  "
          f"val acc: {history.history['val_accuracy'][-1]*100:.1f}%")
    print(f"\n  TEST SET:")
    print(f"    Top-1 accuracy: {test_acc*100:.2f}%")
    print(f"    Top-5 accuracy: {test_top5*100:.2f}%")
    print(f"    Macro precision / recall / F1: "
          f"{prec*100:.1f}% / {rec*100:.1f}% / {f1*100:.1f}%")
    print(f"\n  Model saved: {MODEL_OUT}")
    print(f"  Figures + metrics saved in: {RESULTS}/")
    print(f"\n  Most-confused pairs:")
    for n, a, b in confused[:8]:
        print(f"    {a} -> {b}  ({n})")


if __name__ == "__main__":
    main()