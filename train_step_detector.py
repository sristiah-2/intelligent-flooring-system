#!/usr/bin/env python3
# train_step_detector.py
# CNN-based footstep detector for INA219 waveform
# Produces:
#   step_detector_cnn.h5
#   step_detector_predictions.json

import os
import json
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------- CONFIG ----------------
SR = 25                      # Sampling rate
OUT_DIR = "ina20uf_sessions"
MODEL_PATH = "step_detector_cnn.h5"
PRED_OUT = "step_detector_predictions.json"

WINDOW_SEC = 0.30            # 300ms window
WINDOW_SAMPLES = int(WINDOW_SEC * SR)

STRIDE_SAMPLES = 4           # FINAL FIX: reduced overlap → fewer duplicates

PEAK_MIN_DISTANCE = int(0.25 * SR)
BATCH_SIZE = 64
EPOCHS = 25
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------- LOAD SESSIONS ----------------
def load_sessions():
    sessions = {}
    for fname in sorted(os.listdir(OUT_DIR)):
        if fname.endswith(".npz"):
            try:
                with np.load(os.path.join(OUT_DIR, fname), allow_pickle=True) as d:
                    if "P" in d:
                        sessions[fname] = d["P"].astype(float)
            except:
                pass
    return sessions


# ---------------- PEAK DETECTION ----------------
def detect_peaks(signal):
    peaks, _ = find_peaks(signal, distance=PEAK_MIN_DISTANCE)
    return peaks


# ---------------- BUILD DATASET ----------------
def build_dataset(sessions):
    X, y = [], []

    for sid, sig in sessions.items():

        if len(sig) < WINDOW_SAMPLES:
            continue

        peaks = detect_peaks(sig)

        for start in range(0, len(sig) - WINDOW_SAMPLES, STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            window = sig[start:end]

            # Label window as 1 if it contains any actual peak
            label = 1 if np.any((peaks >= start) & (peaks < end)) else 0

            X.append(window)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Normalize windows
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-8
    X = (X - m) / s

    X = X[..., np.newaxis]
    return X, y


# ---------------- BUILD CNN ----------------
def build_cnn(input_len):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),

        layers.Conv1D(32, 5, padding='same', activation='relu'),
        layers.MaxPool1D(2),

        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.MaxPool1D(2),

        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ---------------- INFERENCE ----------------
def predict_steps(signal, model):

    windows = []
    centers = []

    for start in range(0, len(signal) - WINDOW_SAMPLES, STRIDE_SAMPLES):
        end = start + WINDOW_SAMPLES
        win = signal[start:end]
        win = (win - win.mean()) / (win.std() + 1e-8)

        windows.append(win)
        centers.append(start + WINDOW_SAMPLES // 2)

    W = np.array(windows)[..., np.newaxis]

    probs = model.predict(W, batch_size=256, verbose=0).ravel()

    # FINAL FIX: stricter threshold → fewer false positives
    preds = (probs >= 0.88).astype(int)

    # FINAL FIX: merge peaks within 0.60 seconds
    MAX_MERGE = int(0.60 * SR)

    step_indices = []
    for i in range(len(preds)):
        if preds[i] == 1:
            if len(step_indices) == 0:
                step_indices.append(centers[i])
            elif abs(centers[i] - step_indices[-1]) > MAX_MERGE:
                step_indices.append(centers[i])

    step_times = [round(idx / SR, 3) for idx in step_indices]
    return step_indices, step_times


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\n🔵 Loading sessions...")
    sessions = load_sessions()
    print("Loaded sessions:", len(sessions))

    print("\n🔵 Building dataset...")
    X, y = build_dataset(sessions)
    print("Dataset:", X.shape, "Label counts:", np.bincount(y))

    print("\n🔵 Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n🔵 Building CNN...")
    model = build_cnn(WINDOW_SAMPLES)
    model.summary()

    print("\n🔵 Training...")
    cb = [
        callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=2
    )

    print("\n🔵 Evaluating...")
    probs = model.predict(X_test, batch_size=256).ravel()
    preds = (probs >= 0.88).astype(int)
    print(classification_report(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))

    print("\n🔵 Saving model...")
    model.save(MODEL_PATH)
    print("Saved:", MODEL_PATH)

    # -------- Inference ----------
    print("\n🔵 Running inference on full sessions...")
    all_results = {}

    for sid, sig in sessions.items():
        idxs, times = predict_steps(sig, model)
        all_results[sid] = {
            "pred_step_count": len(idxs),
            "pred_step_sample_indices": idxs,
            "pred_step_times_s": times
        }

    with open(PRED_OUT, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n✅ Saved predictions to:", PRED_OUT)
    print("🎉 Step detection complete!\n")
