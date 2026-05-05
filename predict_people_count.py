#!/usr/bin/env python3
"""
predict_people_count.py

Loads:
- people_counter_cnn.h5   (CNN model)
Reads:
- crowd_dashboard_data.json (base dashboard)
Writes:
- crowd_dashboard_data_people.json (people count prediction)
"""

import numpy as np
import json
import os
from tensorflow.keras.models import load_model

MODEL = "people_counter_cnn.h5"
IN_JSON = "crowd_dashboard_data.json"
OUT_JSON = "crowd_dashboard_data_people.json"

SR = 25
WINDOW_SEC = 10
WINDOW_SAMPLES = int(WINDOW_SEC * SR)
STRIDE = WINDOW_SAMPLES   # non-overlapping windows

# -------------------
# Load Model Safely
# -------------------
if not os.path.exists(MODEL):
    raise SystemExit("ERROR: people_counter_cnn.h5 not found. Run train_people_counter.py first.")

# FIX: load without compiling → avoids keras.metrics.mse deserialization crash
model = load_model(MODEL, compile=False)

# Re-compile manually (safe)
model.compile(optimizer="adam", loss="mse")

print("Model loaded successfully.")

# -------------------
# Load Dashboard JSON
# -------------------
if not os.path.exists(IN_JSON):
    raise SystemExit("ERROR: crowd_dashboard_data.json not found. Run generate_dashboard_data.py first.")

with open(IN_JSON, "r") as f:
    base = json.load(f)

# -------------------
# Extract Crowd Signal
# -------------------
power_mW = np.array(base["power_time_series"]["power_mW"], dtype=np.float32)

# Convert mW → W (because training used W from .npz)
sig = power_mW / 1000.0

if len(sig) < WINDOW_SAMPLES:
    sig = np.pad(sig, (0, WINDOW_SAMPLES - len(sig)), mode="constant")

# -------------------
# Create Windows
# -------------------
windows = []
centers = []

for start in range(0, len(sig) - WINDOW_SAMPLES + 1, STRIDE):
    w = sig[start:start + WINDOW_SAMPLES]

    # Normalize
    mean = w.mean()
    std = w.std() + 1e-8
    w = (w - mean) / std

    windows.append(w)
    centers.append(start + WINDOW_SAMPLES // 2)

if not windows:
    raise SystemExit("ERROR: No windows created from signal.")

W = np.array(windows)[..., None]  # (N, T, 1)

# -------------------
# Predict People Count
# -------------------
preds = model.predict(W).ravel()

est_float = float(np.mean(preds))
est_round = int(round(est_float))

print(f"Predicted people count: {est_float:.3f} → rounded to {est_round}")

# -------------------
# Save People Result JSON
# -------------------
out = {
    "people_count_estimate": {
        "people_est_float": est_float,
        "people_est_rounded": est_round,
        "per_window_preds": preds.tolist(),
        "per_window_centers_s": (np.array(centers)/SR).tolist()
    }
}

with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print(f"Saved people count → {OUT_JSON}")
