#!/usr/bin/env python3
"""
generate_dashboard_data.py

Loads INA219 sessions + metadata, computes metrics, integrates:
- CNN step detector
- CNN people counter output
and writes crowd_dashboard_data.json.

Requires that:
- step_detector_predictions.json exists
- people_count_estimate is written by predict_people_count.py
"""

import numpy as np
import pandas as pd
import os, json

# --- CONFIG ---
SR = 25
n_sessions = 30
Total_Time_s = 60
OUT_DIR = "ina20uf_sessions"
metadata_path = "ina20uf_sessions_metadata.csv"

# --- Load metadata ---
meta_df = pd.read_csv(metadata_path)
peak_P_data = meta_df["peak_P"].astype(float)

# --- Load sessions ---
sessions_power = []
session_names = []

for i in range(n_sessions):
    sid = f"ina20uF_sess_{i+1:03d}"
    fp = os.path.join(OUT_DIR, f"{sid}.npz")
    if not os.path.exists(fp):
        print(f"Missing: {fp}")
        continue

    with np.load(fp, allow_pickle=True) as data:
        if "P" in data:
            arr = np.array(data["P"], dtype=float)
            sessions_power.append(arr)
            session_names.append(sid + ".npz")

if not sessions_power:
    raise SystemExit("No sessions loaded.")

min_len = min(len(p) for p in sessions_power)
sessions_power = [p[:min_len] for p in sessions_power]
P_total = np.sum(np.vstack(sessions_power), axis=0)

# --- Time ---
time_s = np.arange(P_total.size) / SR
dt = 1/SR

# --- Metrics ---
dashboard = {
    "metrics": {
        "total_energy_J": float(np.sum(P_total * dt)),
        "avg_power_mW": float(np.mean(P_total) * 1000),
        "peak_power_W": float(np.max(P_total)),
        "total_footfalls": int(meta_df['n_steps'].sum()),
        "floor_footfall_rate": float(meta_df['n_steps'].sum() / Total_Time_s),
        "avg_steps_per_person_min": float((meta_df['n_steps'].sum()/n_sessions)*1),
        "mean_peak_power_mW": float(peak_P_data.mean()*1000),
    },

    "power_time_series": {
        "time_s": time_s.tolist(),
        "power_mW": (P_total*1000).tolist()
    },

    "sessions": {
        "names": session_names,
        "per_session_peak_mW": (peak_P_data*1000).tolist()
    }
}

# --- Add CNN step detector data ---
if os.path.exists("step_detector_predictions.json"):
    step_json = json.load(open("step_detector_predictions.json"))
    timeline = []
    total_steps = 0
    for sid, item in step_json.items():
        total_steps += item.get("pred_step_count", 0)
        timeline += item.get("pred_step_times_s", [])

    timeline = sorted(timeline)

    dashboard["cnn_step_detection"] = {
        "cnn_total_steps": total_steps,
        "cnn_step_timeline_s": timeline
    }
else:
    dashboard["cnn_step_detection"] = {"error": "No CNN step predictions found"}

# --- Add CNN people counter results ---
if os.path.exists("crowd_dashboard_data_people.json"):
    ppl_json = json.load(open("crowd_dashboard_data_people.json"))
    dashboard["people_count_estimate"] = ppl_json["people_count_estimate"]
else:
    dashboard["people_count_estimate"] = {"error": "People model results not found"}

# --- Save ---
with open("crowd_dashboard_data.json", "w") as f:
    json.dump(dashboard, f, indent=2)

print("DONE → crowd_dashboard_data.json updated")
