"""Create physics-informed synthetic RUL (Remaining Useful Life) data for conveyor belts.

This script steps belts through time, applying load, heat, and vibration to 
calculate cumulative wear, elongation, and crack growth until failure.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def generate_lifecycle_data(n_belts: int, max_hours: int, step_hours: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    # Physics Constants — base rates tuned so all 3 failure modes trigger
    # Wear:  threshold=100%  → needs ~6,000–12,000 hrs at normal load  → k_wear ~0.0005
    # Elong: threshold=4%    → needs ~6,000–12,000 hrs                 → k_elong ~0.00025
    # Crack: threshold=50mm  → needs ~4,000–10,000 hrs                 → k_crack ~0.00008
    k_wear_base  = 0.00050
    k_elong_base = 0.00025
    k_crack_base = 0.00008

    for belt_id in range(1, n_belts + 1):
        # 1. Initialize Static Belt Properties
        hardness = rng.normal(65.0, 5.0)            # Shore A
        thickness = rng.normal(15.0, 1.0)           # mm
        tensile_strength = rng.normal(1000.0, 80.0) # kN/m
        belt_width = rng.normal(1000.0, 200.0)      # mm
        belt_length = rng.normal(50000.0, 5000.0)   # mm
        cover_thickness_top = rng.normal(6.0, 1.0)  # mm
        pulley_alignment = rng.normal(0.0, 1.5)     # mm

        # 2. Establish Base Operating Environment for this specific belt
        base_load = rng.uniform(40.0, 95.0)         # %
        base_speed = rng.normal(3.0, 0.4)           # m/s
        base_temp = rng.normal(30.0, 10.0)          # Celsius
        base_vib = rng.normal(4.0, 1.5)             # mm/s
        base_humidity = rng.normal(60.0, 15.0)      # %

        # 3. Per-belt k values — randomised so different belts fail via different modes
        #    Each k is drawn from a range that still produces failure within 2k–15k hrs.
        k_wear  = rng.uniform(0.00030, 0.00090)   # spread: some belts wear fast, some slow
        k_elong = rng.uniform(0.00015, 0.00045)   # spread: some belts stretch fast
        k_crack = rng.uniform(0.00005, 0.00014)   # spread: some belts crack fast

        # Initial Degradation States
        wear_pct = 0.0
        elongation_pct = 0.0
        crack_length_mm = 0.0
        
        belt_history = []
        failed = False
        failure_mode = None

        # 4. Step through time until failure or max_hours
        for hour in range(0, max_hours, step_hours):
            # Introduce operational noise/fluctuations for this time step
            load = np.clip(base_load + rng.normal(0, 5), 10, 110)
            speed = np.clip(base_speed + rng.normal(0, 0.2), 0.5, 6.0)
            temp = np.clip(base_temp + (load - 50) * 0.2 + rng.normal(0, 3), 10, 85)
            vib = np.clip(base_vib + (speed - 3) * 1.2 + rng.normal(0, 0.8), 1.0, 20.0)
            humidity = np.clip(base_humidity + rng.normal(0, 5), 10, 100)
            
            # Tension proxy: heavily dependent on load
            tension = load * 12.0 
            torque = load * 15.0 + rng.normal(0, 5)
            maintenance_count = int(hour / 1000)

            # Temperature acceleration factor
            temp_factor = np.exp((temp - 20.0) / 50.0)

            # --- APPLY PHYSICS FORMULAS ---
            
            # Archard's Wear proxy
            dw = k_wear * (load * speed / hardness) * temp_factor * step_hours
            wear_pct += dw

            # Viscoelastic Creep proxy (Elongation)
            de = k_elong * (tension / tensile_strength) * temp_factor * step_hours
            elongation_pct += de

            # Paris' Law proxy (Crack Growth)
            dc = k_crack * (vib ** 2) * step_hours
            crack_length_mm += dc * (1 + 0.12 * crack_length_mm) # Accelerates as crack grows

            # Record state
            belt_history.append({
                "belt_id": f"BLT-{belt_id:04d}",
                "hour": hour,
                "rubber_hardness_shore_a": round(hardness, 1),
                "belt_thickness_mm": round(thickness, 2),
                "tensile_strength_kn_m": round(tensile_strength, 1),
                "belt_width_mm": round(belt_width, 1),
                "belt_length_mm": round(belt_length, 1),
                "cover_thickness_top_mm": round(cover_thickness_top, 2),
                "pulley_alignment_mm": round(pulley_alignment, 2),
                "load_pct": round(load, 1),
                "belt_speed_mps": round(speed, 2),
                "temperature_c": round(temp, 1),
                "humidity_pct": round(humidity, 1),
                "vibration_mms": round(vib, 2),
                "torque_nm": round(torque, 1),
                "belt_tension_n": round(tension, 1),
                "maintenance_count": maintenance_count,
                "wear_pct": round(wear_pct, 2),
                "elongation_pct": round(elongation_pct, 3),
                "crack_length_mm": round(crack_length_mm, 2),
            })

            # --- CHECK FAILURE THRESHOLDS ---
            if wear_pct >= 100.0:
                failed = True; failure_mode = "Max Wear Reached"
            elif elongation_pct >= 4.0: # 4% stretch is typically critical
                failed = True; failure_mode = "Excessive Elongation"
            elif crack_length_mm >= 50.0:
                failed = True; failure_mode = "Critical Crack Size"

            if failed:
                break

        # 5. Back-calculate the Remaining Useful Life (RUL) for ML targets
        final_hour = belt_history[-1]["hour"]
        for record in belt_history:
            record["failure_hour"] = final_hour
            record["rul_hours"] = final_hour - record["hour"]
            record["eventual_failure_mode"] = failure_mode if failed else "Censored (Survived)"

        records.extend(belt_history)

    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Generate Physics-Informed Belt RUL Data")
    parser.add_argument("--belts", type=int, default=300, help="Number of belts to simulate")
    parser.add_argument("--max-hours", type=int, default=20000, help="Max hours to run")
    parser.add_argument("--step", type=int, default=100, help="Inspection interval (hours)")
    parser.add_argument("--output", type=Path, default=Path("belt_rul_training_data.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_lifecycle_data(args.belts, args.max_hours, args.step, args.seed)
    
    print(f"Generated {len(df):,} inspection records across {df['belt_id'].nunique()} belts.")
    print(f"Average lifetime: {df.groupby('belt_id')['failure_hour'].max().mean():,.0f} hours.")
    
    # Print failure mode distribution
    modes = df.groupby('belt_id')['eventual_failure_mode'].last().value_counts()
    print("\nFailure Modes:")
    for mode, count in modes.items():
        print(f"  - {mode}: {count}")

    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()