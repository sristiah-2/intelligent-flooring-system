#!/usr/bin/env python3
# generate_ina20uF_dataset.py
# Generates synthetic INA219-style dataset:
# -
# sampling rate: 25 Hz
# -
# capacitor: 20 uF
# -
# scenario: single tile, normal walking
# sessions: 30 x 60s
# Output: ./ina20uf_sessions/*.npz, metadata CSV, and a zip file.

import os, json, zipfile, numpy as np, pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(20251106)

OUT_DIR = "ina20uf_sessions"
os.makedirs(OUT_DIR, exist_ok=True)

def simulate_session(session_id, sr=25, length_s=60, cap_uF=20):
    n = int(sr * length_s)
    # Corrected: t = np.arange(n) / sr
    t = np.arange(n) / sr

    # capacitor
    C = cap_uF * 1e-6 

    # uint: delta V per step range for 20uF (tuned from your paper)
    delta_v_range = (0.2, 3.0) 

    # arrays
    V = np.zeros(n, dtype=np.float32) 
    I = np.zeros(n, dtype=np.float32) 

    # choose load condition sometimes
    # Corrected: load_R assignment was split and incorrect.
    load_R = np.random.choice([float('inf'), 20000.0, 10000.0], p=[0.7, 0.2, 0.1]) 

    # normal walking step rate: 0.8 - 1.2 Hz
    step_rate = np.random.uniform(0.8, 1.2) 
    avg_interval = 1.0 / step_rate 

    # generate step times with jitter
    times = [] 
    current_time = np.random.uniform(0, 1.0) 
    while current_time < length_s - 0.05: 
        times.append(current_time) 
        # Corrected: current_time update was a standalone statement.
        current_time += max(0.25, np.random.normal(avg_interval, 0.08)) 

    events = [] 
    for tt in times: 
        idx = int(tt * sr) 
        
        # Corrected: force and delta_v assignment was split and incorrect.
        force = np.random.choice([0.8, 1.0, 1.2], p=[0.25, 0.6, 0.15]) 
        delta_v = np.random.uniform(delta_v_range[0], delta_v_range[1]) * force 

        # voltage rise applied over a few samples (rectifier + cap smoothing)
        # Corrected: rise_len assignment was split.
        rise_len = int(np.clip(np.random.uniform(1, sr*0.08), 1, sr)) 
        for k in range(rise_len):
            if idx+k < n: 
                # Corrected: V[idx+k] calculation was split and incorrect.
                V[idx+k] += delta_v * (1 - np.exp(-5*(k/(rise_len+1)))) 

        # estimate charging current peak: I_peak ~ C * delta_v / rise_time
        # Corrected: rise_time and I_peak assignments were split.
        rise_time = max(1.0/sr, rise_len/sr) 
        I_peak = np.clip(C * delta_v / rise_time, 1e-6, 5e-3) 

        # Corrected: pulse_len assignment was split.
        pulse_len = int(np.clip(np.random.uniform(1, sr*0.12), 1, sr)) 
        for k in range(pulse_len): 
            if idx+k < n: 
                # Corrected: I[idx+k] calculation was split and incorrect.
                I[idx+k] += I_peak * np.random.uniform(0.9, 1.1) * np.exp(-3*k/pulse_len) 

        events.append({"time_s": round(tt, 3), "delta_v": round(float(delta_v), 4), "I_peak_A": float(I_peak)}) 

    # simulate discharge via load (RC)
    dt = 1.0/sr 
    for i in range(1, n): 
        # Corrected: load_R check was split.
        if np.isfinite(load_R): 
            leak = V[i-1] / load_R 
            # Corrected: dV_leak calculation was split.
            dV_leak = (leak/C) * dt 
        else:
            dV_leak = 0.0 
        
        # Apply current (I) and leakage (dV_leak)
        # The discharge model is applied to the previous voltage V[i-1].
        # The V[i] += ... lines from the step charge loop are handled first.
        V[i] += V[i-1] + (I[i] / C) * dt - dV_leak # Standard Euler RC model
        V[i] = max(0.0, V[i]) 
        
        # small noise
        V[i] += np.random.normal(scale=0.002) 

    # small noise in current and occasional small negative backflow events
    I += np.random.normal(scale=5e-5, size=n) 
    # Corrected: loop for backflow events was split.
    for _ in range(np.random.poisson(1.0)): 
        pos = np.random.randint(0, n) 
        width = np.random.randint(1, int(0.15*sr)+1) 
        I[pos:pos+width] += np.random.uniform(-3e-4, -5e-5) 

    # compute power and clip values to realistic bounds
    V = np.clip(V, 0.0, 96.0) 
    I = np.clip(I, -0.01, 0.01) 
    P = V * I 
    
    # summary meta
    # Corrected: meta dictionary construction was split.
    meta = {"session_id": session_id, "sr": sr, "length_s": length_s,
            "cap_uF": cap_uF, "C_F": C, "mode": "normal_walk", 
            "step_rate_hz": round(step_rate, 3), 
            "load_R_ohm": (None if not np.isfinite(load_R) else float(load_R)), 
            "peak_V": float(np.max(V)), "peak_I": float(np.max(I)),
            "peak_P": float(np.max(P)), "n_steps": len(events)} 

    return meta, V.astype(np.float32), I.astype(np.float32), P.astype(np.float32), events 

# create sessions
sessions_meta = [] 
n_sessions = 30 
for i in range(n_sessions): 
    # Corrected: sid string formatting was incorrect.
    sid = f"ina20uF_sess_{i+1:03d}" 
    # Corrected: simulate_session call was split.
    meta, V, I, P, events = simulate_session(sid, sr=25, length_s=60, cap_uF=20) 
    
    # Corrected: fname, np.savez_compressed call, and sessions_meta append were split/incorrect.
    fname = os.path.join(OUT_DIR, f"{sid}.npz") 
    np.savez_compressed(fname, V=V, I=I, P=P, sr=meta["sr"], 
                        start_ts=(datetime.utcnow()+timedelta(seconds=i*75)).isoformat(),
                        meta=json.dumps(meta), events=json.dumps(events)) 
    
    sessions_meta.append({"file_id": sid, "file_path": fname, "peak_V": meta["peak_V"],
                          "peak_I": meta["peak_I"], "peak_P": meta["peak_P"],
                          "n_steps": meta["n_steps"]}) 

# save metadata CSV
meta_df = pd.DataFrame(sessions_meta) 
csv_path = "ina20uf_sessions_metadata.csv" 
meta_df.to_csv(csv_path, index=False) 

# zip everything
zip_path = "synthetic_ina219_20uf_25hz.zip" 
# Corrected: zipfile context manager was split.
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf: 
    for row in sessions_meta: 
        # Corrected: zf.write call for session files was split.
        zf.write(row["file_path"], arcname=os.path.basename(row["file_path"])) 
    zf.write(csv_path, arcname=os.path.basename(csv_path)) 

print("Created dataset:", zip_path) 
print("Sessions directory:", OUT_DIR) 
print("Metadata CSV:", csv_path)