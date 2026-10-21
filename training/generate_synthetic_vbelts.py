"""
Synthetic V-belt run-to-failure data (DEVELOPMENT / DEMO ONLY)
==============================================================
Physics-inspired simulation of classical wrapped, wedge and raw-edge cogged belts so the
pipeline can be demonstrated before real plant data exists. NOT calibrated against
JK Fenner test data - never present models trained on it as real-world accuracy.

Mechanisms:
  - bending stress   ~ (reference pulley dia / actual dia)^2  (cogged belts tolerate small pulleys)
  - flex fatigue     ~ sqrt(bending frequency)
  - load fatigue     ~ (load / 80 %)^3
  - rubber ageing    x2 per +10 °C (Arrhenius rule of thumb)
  - elongation -> tension loss -> slip -> wear and heat; re-tensioning restores tension
  - cracks initiate after a stress-dependent delay, then grow exponentially
A belt fails when any degradation reaches its belt type's limit (same placeholder limits
as the default belt types in the app).

Usage:
  python training/generate_synthetic_vbelts.py --belts 300 --step 250 --seed 42 --output out.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = {
    #                sections (code: reference small-pulley dia mm)        limits: elong %, wear %, crack mm
    "classical":  {"sections": {"A": 75, "B": 125, "C": 200},   "limits": (3.0, 100.0, 2.0), "bend_tolerance": 1.0, "weight": 0.45},
    "wedge":      {"sections": {"SPZ": 63, "SPA": 90, "SPB": 140}, "limits": (2.0, 100.0, 1.5), "bend_tolerance": 1.0, "weight": 0.35},
    "cogged":     {"sections": {"AX": 63, "BX": 100, "CX": 160},  "limits": (2.0, 100.0, 2.5), "bend_tolerance": 1.6, "weight": 0.20},
}
MAX_HOURS = 80000


def simulate_belt(belt_no: int, rng: np.random.Generator, step: int, censor_frac: float = 0.0):
    kind = rng.choice(list(TYPES), p=[t["weight"] for t in TYPES.values()])
    spec = TYPES[kind]
    section = rng.choice(list(spec["sections"]))
    d_ref = spec["sections"][section]

    d_driver = d_ref * rng.uniform(0.85, 2.2)
    ratio = rng.uniform(1.0, 4.0)
    d_driven = d_driver * ratio
    centre = rng.uniform(1.0, 2.5) * (d_driver + d_driven) / 2
    pitch_len = 2 * centre + np.pi * (d_driver + d_driven) / 2 + (d_driven - d_driver) ** 2 / (4 * centre)
    n_pulleys = 2 if rng.random() < 0.85 else 3
    rpm = rng.choice([720, 960, 1440, 2880], p=[0.15, 0.35, 0.4, 0.1])
    speed = np.pi * d_driver * rpm / 60000
    while speed > 30:                                   # classical belts are limited to ~30 m/s
        rpm = rpm / 2
        speed = np.pi * d_driver * rpm / 60000
    bend_freq = speed / (pitch_len / 1000) * n_pulleys
    rated_kw = rng.uniform(2, 75)

    base_load = rng.uniform(40, 110)
    ambient = rng.uniform(22, 45)
    misalign = abs(rng.normal(0, 1))
    t0 = rng.uniform(200, 1500)                         # installation tension per belt, N

    # belt-to-belt manufacturing / installation variability
    k_e, k_w, k_c = (rng.lognormal(0, 0.25) for _ in range(3))
    stress_bend = (d_ref / d_driver) ** 2 / spec["bend_tolerance"]
    crack_init = 5000 * rng.lognormal(0, 0.4)           # stress-hours until a crack initiates

    elong = wear = crack = 0.0
    tension, maint, stress_hours, cracked = t0, 0, 0.0, False
    lim_e, lim_w, lim_c = spec["limits"]
    rows, hour, dt = [], 0, 1.0 * step
    while hour <= MAX_HOURS:
        load = float(np.clip(base_load + rng.normal(0, 5), 5, 140))
        slip = float(np.clip(0.3 + 6 * (1 - tension / t0) + 0.004 * load + rng.normal(0, 0.1), 0, 15))
        temp = ambient + 0.22 * load + 3.0 * slip + rng.normal(0, 1.5)
        heat = 2 ** ((temp - 45) / 10)
        stress = (load / 80) ** 3 * stress_bend * np.sqrt(bend_freq / 10)

        rows.append({
            "belt_id": f"VB-{belt_no:04d}", "belt_type": section, "hour": hour,
            "pitch_length_mm": round(pitch_len, 0), "driver_pulley_dia_mm": round(d_driver, 1),
            "driven_pulley_dia_mm": round(d_driven, 1), "centre_distance_mm": round(centre, 0),
            "num_pulleys": n_pulleys, "driver_rpm": rpm, "rated_power_kw": round(rated_kw, 1),
            "load_pct": round(load, 1), "belt_tension_n": round(tension, 0), "temperature_c": round(temp, 1),
            "slip_pct": round(slip, 2), "vibration_mms": round(max(1.5 + 0.8 * misalign + 0.6 * slip + 1.2 * crack + rng.normal(0, 0.3), 0.2), 2),
            "maintenance_count": maint,
            # inspection measurements carry small reading errors
            "elongation_pct": round(max(elong + rng.normal(0, 0.02), 0), 3),
            "wear_pct": round(max(wear + rng.normal(0, 0.8), 0), 1),
            "crack_depth_mm": round(max(crack + (rng.normal(0, 0.02) if crack > 0 else 0), 0), 3),
        })
        if elong >= lim_e or wear >= lim_w or crack >= lim_c:
            break

        # degradation over the next interval
        elong = 0.35 * (1 - np.exp(-(hour + dt) / 300)) + (elong - 0.35 * (1 - np.exp(-hour / 300))) \
            + 1.1e-4 * k_e * np.sqrt(stress) * heat * dt
        wear += 3.0e-3 * k_w * (load / 80) * (1 + slip) * dt
        stress_hours += stress * heat * dt
        if not cracked and stress_hours >= crack_init:
            cracked, crack = True, 0.05
        if cracked:
            crack += 3.0e-4 * k_c * stress * heat * (0.15 + crack) * dt
        tension = t0 * max(0.3, 1 - 0.18 * elong)
        if tension < 0.75 * t0:                         # re-tension at inspection
            tension, maint = t0 * (1 - 0.05 * elong), maint + 1
        hour += step

    failed = elong >= lim_e or wear >= lim_w or crack >= lim_c
    df = pd.DataFrame(rows)
    # Right-censoring: a fraction of failing belts is "replaced early / still running" - the log just stops
    # at some point of their life and no failure hour is known (blank failure_hour and rul_hours).
    if failed and rng.random() < censor_frac:
        cut = rows[-1]["hour"] * rng.uniform(0.25, 0.9)
        df, failed = df[df["hour"] <= cut].copy(), False
    if failed:
        df["failure_hour"] = rows[-1]["hour"]
        df["rul_hours"] = df["failure_hour"] - df["hour"]
    else:                                               # also covers belts that survived MAX_HOURS
        df["failure_hour"] = np.nan
        df["rul_hours"] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--belts", type=int, default=300)
    ap.add_argument("--step", type=int, default=250, help="Inspection interval in operating hours")
    ap.add_argument("--censor-frac", type=float, default=0.0,
                    help="Share of belts whose log ends before failure (no failure hour), e.g. 0.3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("synthetic_vbelts.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = []
    for i in range(1, args.belts + 1):
        frames.append(simulate_belt(i, rng, args.step, args.censor_frac))
    data = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)

    last = data.groupby("belt_id").last()
    ended = last[last.failure_hour.notna()]
    life = ended["failure_hour"]
    mode = pd.Series({
        "elongation": (ended.elongation_pct >= ended.belt_type.map(_limit(0))).sum(),
        "wear": (ended.wear_pct >= ended.belt_type.map(_limit(1))).sum(),
        "crack": (ended.crack_depth_mm >= ended.belt_type.map(_limit(2))).sum(),
    })
    n_cens = len(last) - len(ended)
    print(f"Wrote {len(data):,} rows / {len(last)} belts to {args.output} "
          f"({n_cens} censored = {n_cens / len(last):.0%}, no failure hour)")
    print(f"Life of failed belts (h): median {life.median():,.0f}, p10 {life.quantile(.1):,.0f}, p90 {life.quantile(.9):,.0f}")
    print("Failure driver counts:", mode.to_dict())


def _limit(i):
    return {sec: t["limits"][i] for t in TYPES.values() for sec in t["sections"]}


if __name__ == "__main__":
    main()
