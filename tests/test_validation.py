import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.validate_data import validate_trial


class ValidationTests(unittest.TestCase):
    def _write_trial(self, path: Path, timestamps: list[int]) -> None:
        pd.DataFrame(
            {
                "time": timestamps,
                "ax": [0.1, 0.2, 0.1, 0.0],
                "ay": [0.0, 0.1, 0.0, -0.1],
                "az": [9.8, 9.9, 9.8, 9.7],
            }
        ).to_csv(path, index=False)

    def test_valid_trial_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.csv"
            self._write_trial(path, [0, 10, 20, 30])
            report = validate_trial(path)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["duration_ms"], 30.0)
        self.assertEqual(report["sample_rate_hz"], 100.0)

    def test_repeated_timestamp_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.csv"
            self._write_trial(path, [0, 10, 10, 30])
            report = validate_trial(path)

        self.assertEqual(report["status"], "FAIL")
        self.assertIn("timestamps are not strictly increasing", str(report["issues"]))


if __name__ == "__main__":
    unittest.main()
