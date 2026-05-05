#!/usr/bin/env python3
"""
train_people_counter.py

- Loads single-person session .npz files from OUT_DIR.
- Synthesizes multi-person training examples by summing random subsets of single-session signals.
- Trains a 1D CNN regressor to predict integer people-count from the summed waveform.
- Saves model to people_counter_cnn.h5 and a small scaler to npz.

Assumptions:
- Each .npz contains array 'P' (power waveform, float).
- Sampling rate consistent across sessions (SR).
- Sessions roughly same duration; we trim/pad to MIN_LEN.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- CONFIG ---
OUT_DIR = "ina20uf_sessions"        # folder of single-session .npz files
MODEL_OUT = "people_counter_cnn.h5"
SCALER_OUT = "people_counter_scaler.npz"   # saves mean/std
SR = 25
MAX_PEOPLE = 8            # maximum people to synthesize (you can increase if enough sessions)
EXAMPLES_PER_COUNT = 200  # how many synthetic examples per people-count label
WINDOW_SEC = 10.0         # how long (seconds) each example is (longer windows help)
WINDOW_SAMPLES = int(WINDOW_SEC * SR)
RANDOM_SEED = 42

BATCH_SIZE = 32
EPOCHS = 40

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- 1. load single-person sessions ---
def load_single_sessions(out_dir):
    sessions = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(out_dir, fname)
        try:
            with np.load(path, allow_pickle=True) as data:
                if "P" in data:
                    arr = np.asarray(data["P"], dtype=np.float32)
                    sessions.append(arr)
        except Exception as e:
            print("skip", fname, "->", e)
    return sessions

# --- 2. unify lengths: choose MIN_LEN across sessions or desired window length ---
def build_synth_examples(sessions, max_people=MAX_PEOPLE, examples_per_count=EXAMPLES_PER_COUNT,
                         window_samples=WINDOW_SAMPLES):
    # require sessions longer than window_samples
    good = [s for s in sessions if s.size >= window_samples]
    if len(good) == 0:
        raise SystemExit("No sessions long enough. Increase WINDOW_SEC or provide longer recordings.")
    print(f"Using {len(good)} sessions (>= {window_samples} samples) for synthesis.")

    X = []
    y = []
    # For each people count k=1..max_people, synthesize many examples
    for k in range(1, max_people + 1):
        for ex in range(examples_per_count):
            # pick k sessions randomly (with replacement allowed)
            chosen = [random.choice(good) for _ in range(k)]
            # for each chosen session, pick a random window of length window_samples
            snippets = []
            for s in chosen:
                start = random.randint(0, s.size - window_samples)
                snippets.append(s[start:start + window_samples])
            # align shapes and sum
            summed = np.sum(np.vstack(snippets), axis=0)
            X.append(summed.astype(np.float32))
            y.append(k)
    X = np.array(X)  # shape (N, T)
    y = np.array(y).astype(np.float32)  # regression target
    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y

# --- 3. simple normalization (per-example mean/std) ---
def normalize_windows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True) + 1e-8
    Xn = (X - means) / stds
    return Xn, means.ravel(), stds.ravel()

# --- 4. model ---
def build_regressor(input_len):
    inp = layers.Input(shape=(input_len,1))
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(inp)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)   # regression output

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- MAIN ---
if __name__ == "__main__":
    sessions = load_single_sessions(OUT_DIR)
    if len(sessions) == 0:
        raise SystemExit("No session .npz files found in " + OUT_DIR)

    X, y = build_synth_examples(sessions)
    print("Synth dataset shape:", X.shape, "targets:", np.unique(y)[:10])

    # normalize
    Xn, means, stds = normalize_windows(X)
    # reshape for CNN
    Xn = Xn[..., np.newaxis]   # (N, T, 1)

    # train/test split (regression)
    X_train, X_val, y_train, y_val = train_test_split(Xn, y, test_size=0.15, random_state=RANDOM_SEED)

    model = build_regressor(Xn.shape[1])
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=cb,
                        verbose=2)

    # evaluate (report MAE)
    preds_val = model.predict(X_val).ravel()
    mae = mean_absolute_error(y_val, preds_val)
    print(f"Validation MAE: {mae:.3f} people")

    # Save scaler (mean/std per training example can be used later; we will use per-window normalization at inference)
    np.savez_compressed(SCALER_OUT, means=means, stds=stds)
    print("Saved scaler:", SCALER_OUT)

    # model already saved by checkpoint; ensure final save
    model.save(MODEL_OUT)
    print("Saved model:", MODEL_OUT)

    # Save a small sample of preds/true for quick inspection
    sample_report = {
        "val_mae": float(mae),
        "y_val_sample": y_val[:50].tolist(),
        "preds_val_sample": preds_val[:50].tolist()
    }
    with open("people_counter_report.json", "w") as f:
        json.dump(sample_report, f, indent=2)

    print("Training complete. Report saved to people_counter_report.json")
