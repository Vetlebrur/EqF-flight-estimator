# EqF-flight-estimator

Using equivariant filter framework, can we create a good estimator for flight?

## NIMBUS24 trajectory + bearing visualisation

This repository now includes a minimal tool to:

- download `20241011_NIMBUS24_Flight_FC_Data.csv` from `icl-rocketry/iclr-data`
- plot the **full flight trajectory**
- plot a **bearing model** (where the rocket points through time)
- replay the whole flight through a simulator scaffold for future live-estimation work

### Quick start

```bash
python -m pip install -r requirements.txt
python flight_analysis.py
```

By default this will:

- download the CSV to `data/20241011_NIMBUS24_Flight_FC_Data.csv` (if missing)
- save plots to:
  - `outputs/nimbus24_trajectory.png`
  - `outputs/nimbus24_bearing.png`

### Simulation scaffold for future live estimation

Use `FlightReplaySimulator` in `flight_analysis.py` and pass an object with:

```python
update(sample: FlightSample) -> object
```

That `update(...)` hook is designed so you can plug in your live estimator logic later while replaying recorded flight data sample-by-sample.
