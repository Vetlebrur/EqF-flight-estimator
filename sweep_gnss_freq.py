"""Sweep GNSS_UPDATE_FREQ_HZ from 0.2 to 10 Hz and save outputs to outputs/gnss_freq_sweep/."""

import numpy as np
from pathlib import Path

import eqf_filter

FREQS_HZ = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

SWEEP_DIR = Path("outputs/gnss_freq_sweep")
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = {
    "full":    "data/20241011_NIMBUS24_Flight_FC_Data.csv",
    "30s":     "data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",
    "1s_loop": "data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv",
}
csv_in = INPUT_CSV[eqf_filter.DATASET]

results = []

print(f"Sweep: GNSS_UPDATE_FREQ_HZ over {FREQS_HZ}")
print(f"Dataset: {eqf_filter.DATASET}  ({csv_in})")
print(f"Output:  {SWEEP_DIR}/")
print(f"USE_GNSS_UPDATE={eqf_filter.USE_GNSS_UPDATE}  USE_MAG_UPDATE={eqf_filter.USE_MAG_UPDATE}")
print()

for freq in FREQS_HZ:
    tag = f"{freq:.4g}Hz".replace(".", "p")
    csv_out = str(SWEEP_DIR / f"tg_eqf_output_{tag}.csv")

    print(f"[{freq:6.3f} Hz]  ->  {csv_out}")
    rmse = eqf_filter.run(
        csv_in=csv_in,
        csv_out=csv_out,
        gnss_freq_hz=freq,
        silent=True,
    )
    results.append((freq, rmse))
    print(f"           angular RMSE = {rmse.get('angular', float('nan')):.2f} deg")

print()
print("=" * 50)
print(f"{'Freq (Hz)':>10}  {'Angular RMSE (deg)':>20}")
print("-" * 50)
for freq, rmse in results:
    print(f"{freq:>10.3f}  {rmse.get('angular', float('nan')):>20.2f}")
