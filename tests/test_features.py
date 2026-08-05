import unittest

import numpy as np
import pandas as pd

from src.features import build_window_feature_table, extract_trial_features


class FeatureTests(unittest.TestCase):
    def test_constant_axes_have_zero_centered_rms(self) -> None:
        frame = pd.DataFrame(
            {
                "time": np.arange(0, 100, 10),
                "ax": np.full(10, 1.0),
                "ay": np.full(10, 2.0),
                "az": np.full(10, 9.8),
            }
        )
        features = extract_trial_features(frame)

        self.assertEqual(features["ax_rms_centered"], 0.0)
        self.assertEqual(features["ay_rms_centered"], 0.0)
        self.assertEqual(features["az_rms_centered"], 0.0)
        self.assertEqual(features["magnitude_rms_centered"], 0.0)

    def test_dominant_frequency_tracks_sine_wave(self) -> None:
        sample_rate_hz = 200.0
        time_seconds = np.arange(0.0, 2.0, 1.0 / sample_rate_hz)
        signal = np.sin(2.0 * np.pi * 10.0 * time_seconds)
        frame = pd.DataFrame(
            {
                "time": time_seconds * 1000.0,
                "ax": signal,
                "ay": np.zeros_like(signal),
                "az": np.full_like(signal, 9.8),
            }
        )
        features = extract_trial_features(frame)

        self.assertLess(abs(features["ax_dominant_frequency_hz"] - 10.0), 0.6)

    def test_window_table_keeps_trial_identity(self) -> None:
        import tempfile
        from pathlib import Path

        sample_rate_hz = 100.0
        time_ms = np.arange(0.0, 10000.0, 1000.0 / sample_rate_hz)
        signal = np.sin(2.0 * np.pi * 5.0 * time_ms / 1000.0)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "0.25g"
            folder.mkdir()
            pd.DataFrame(
                {"time": time_ms, "ax": signal, "ay": signal, "az": signal + 9.8}
            ).to_csv(folder / "trial_01.csv", index=False)
            table = build_window_feature_table(Path(directory))

        self.assertEqual(len(table), 5)
        self.assertEqual(table["trial_id"].nunique(), 1)
        self.assertEqual(table["window_id"].tolist(), [1, 2, 3, 4, 5])
        self.assertTrue((table["mass_g"] == 0.25).all())


if __name__ == "__main__":
    unittest.main()
