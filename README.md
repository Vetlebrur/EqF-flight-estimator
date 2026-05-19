# EqF-flight-estimator

TG-EqF and EKF for rocket flight estimation, evaluated on the NIMBUS24 sounding rocket.

## Setup

```bash
pip install -r requirements.txt
# or
uv run python eqf_filter.py
```

## Running

```bash
python eqf_filter.py         # TG-EqF → outputs/tg_eqf_output_full.csv
python ekf_filter.py         # EKF    → outputs/ekf_output_full.csv
python compare_filters.py    # plot EqF vs EKF vs GNSS vs FC
python tests/plot_trajectory.py  # diagnostics: ANIS, ANEES, RMS error vs FC
```

## Data

Put the NIMBUS24 CSV in `data/20241011_NIMBUS24_Flight_FC_Data.csv`.

## Dependencies

Requires `eqf-reference/` to be present (symmetry group implementation).
`pylie` is installed from GitHub — see `requirements.txt`.
