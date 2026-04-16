import tempfile
import unittest
from pathlib import Path

from flight_analysis import FlightReplaySimulator, load_flight_samples


CSV_HEADER = (
    "time(s),gps_long(deg),gps_lat(deg),gps_alt(mm),pn(m),pe(m),pd(m),"
    "roll(rad),pitch(rad),yaw(rad)\n"
)
CSV_ROWS = (
    "0.0,-8.2,39.3,200000,0.0,0.0,0.0,0.0,0.1,0.2\n",
    "1.0,-8.2,39.3,201000,10.0,5.0,-2.0,0.0,0.2,0.3\n",
)


class _MockEstimator:
    def __init__(self):
        self.seen = []

    def update(self, sample):
        self.seen.append(sample.t)
        return sample.t


class FlightAnalysisTests(unittest.TestCase):
    def test_load_flight_samples_parses_position_and_attitude(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text(CSV_HEADER + "".join(CSV_ROWS), encoding="utf-8")

            samples = load_flight_samples(csv_path)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[1].north_m, 10.0)
        self.assertEqual(samples[1].east_m, 5.0)
        self.assertEqual(samples[1].up_m, 2.0)

    def test_replay_simulator_runs_estimator_over_full_flight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text(CSV_HEADER + "".join(CSV_ROWS), encoding="utf-8")
            samples = load_flight_samples(csv_path)

        estimator = _MockEstimator()
        outputs = FlightReplaySimulator(samples).run(estimator)

        self.assertEqual(outputs, [0.0, 1.0])
        self.assertEqual(estimator.seen, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
