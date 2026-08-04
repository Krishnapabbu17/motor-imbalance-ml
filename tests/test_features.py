import unittest

import numpy as np
import pandas as pd

from src.features import extract_trial_features


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


if __name__ == "__main__":
    unittest.main()
