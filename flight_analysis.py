"""Tools for plotting NIMBUS24 trajectory/bearing and replaying flight data."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol
from urllib.request import urlretrieve

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/icl-rocketry/iclr-data/"
    "main/"
    "20241011_NIMBUS24_Flight/20241011_NIMBUS24_Flight_FC_Data.csv"
)


@dataclass(frozen=True)
class FlightSample:
    """A single flight sample with position and attitude."""

    t: float
    lat_deg: float
    lon_deg: float
    gps_alt_m: float
    north_m: float
    east_m: float
    down_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    @property
    def up_m(self) -> float:
        return -self.down_m

    @property
    def bearing_vector_enu(self) -> tuple[float, float, float]:
        """Approximate body-forward axis in ENU from yaw/pitch."""
        east = math.cos(self.pitch_rad) * math.sin(self.yaw_rad)
        north = math.cos(self.pitch_rad) * math.cos(self.yaw_rad)
        up = math.sin(self.pitch_rad)
        return (east, north, up)


class LiveEstimator(Protocol):
    """Extension point for future live estimators."""

    def update(self, sample: FlightSample) -> object:
        ...


def download_csv(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)
    return destination


def _parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def load_flight_samples(csv_path: Path) -> List[FlightSample]:
    samples: List[FlightSample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                FlightSample(
                    t=_parse_float(row, "time(s)"),
                    lon_deg=_parse_float(row, "gps_long(deg)"),
                    lat_deg=_parse_float(row, "gps_lat(deg)"),
                    gps_alt_m=_parse_float(row, "gps_alt(mm)") / 1000.0,
                    north_m=_parse_float(row, "pn(m)"),
                    east_m=_parse_float(row, "pe(m)"),
                    down_m=_parse_float(row, "pd(m)"),
                    roll_rad=_parse_float(row, "roll(rad)"),
                    pitch_rad=_parse_float(row, "pitch(rad)"),
                    yaw_rad=_parse_float(row, "yaw(rad)"),
                )
            )
    return samples


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return plt


def plot_trajectory(samples: Iterable[FlightSample], output_path: Path) -> None:
    plt = _require_matplotlib()
    sampled = list(samples)
    if not sampled:
        raise ValueError("No samples to plot")

    east = [s.east_m for s in sampled]
    north = [s.north_m for s in sampled]
    up = [s.up_m for s in sampled]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(east, north, up, linewidth=1.2, color="tab:blue")
    ax.set_title("NIMBUS24 Full Flight Trajectory")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_bearing(samples: Iterable[FlightSample], output_path: Path, stride: int = 200) -> None:
    plt = _require_matplotlib()
    sampled = list(samples)
    if not sampled:
        raise ValueError("No samples to plot")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    east = [s.east_m for s in sampled]
    north = [s.north_m for s in sampled]
    up = [s.up_m for s in sampled]
    ax.plot(east, north, up, linewidth=0.9, color="0.65", label="Trajectory")

    arrow_samples = sampled[:: max(stride, 1)]
    u = [s.bearing_vector_enu[0] for s in arrow_samples]
    v = [s.bearing_vector_enu[1] for s in arrow_samples]
    w = [s.bearing_vector_enu[2] for s in arrow_samples]
    ax.quiver(
        [s.east_m for s in arrow_samples],
        [s.north_m for s in arrow_samples],
        [s.up_m for s in arrow_samples],
        u,
        v,
        w,
        length=20.0,
        normalize=True,
        color="tab:red",
        linewidth=0.7,
    )

    ax.set_title("NIMBUS24 Bearing Model (rocket pointing direction)")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


class FlightReplaySimulator:
    """Replay recorded flight samples to emulate a future live data stream."""

    def __init__(self, samples: Iterable[FlightSample]):
        self._samples = list(samples)

    def iter_samples(self) -> Iterable[FlightSample]:
        return iter(self._samples)

    def run(self, estimator: Optional[LiveEstimator] = None) -> List[object]:
        outputs: List[object] = []
        for sample in self._samples:
            if estimator is not None:
                outputs.append(estimator.update(sample))
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/20241011_NIMBUS24_Flight_FC_Data.csv"),
        help="Local path for the FC CSV file.",
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_DATA_URL,
        help="URL used if --csv does not exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--bearing-stride",
        type=int,
        default=200,
        help="Sample step for bearing arrows to keep plots readable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        print(f"Downloading dataset to {args.csv} ...")
        download_csv(args.download_url, args.csv)

    print(f"Loading samples from {args.csv} ...")
    samples = load_flight_samples(args.csv)
    print(f"Loaded {len(samples)} samples")

    trajectory_plot = args.output_dir / "nimbus24_trajectory.png"
    bearing_plot = args.output_dir / "nimbus24_bearing.png"

    plot_trajectory(samples, trajectory_plot)
    plot_bearing(samples, bearing_plot, stride=args.bearing_stride)

    simulator = FlightReplaySimulator(samples)
    sample_count = sum(1 for _ in simulator.iter_samples())

    print(f"Saved trajectory plot: {trajectory_plot}")
    print(f"Saved bearing plot: {bearing_plot}")
    print(
        "Simulation scaffold ready: FlightReplaySimulator can be wired to a live estimator "
        f"(replay length: {sample_count} samples)."
    )


if __name__ == "__main__":
    main()
