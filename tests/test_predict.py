import unittest

import numpy as np
import pandas as pd

from src.features import extract_window_feature_table
from src.predict import _majority_vote


class PredictionTests(unittest.TestCase):
    def test_new_trial_uses_five_two_second_windows(self) -> None:
        time = np.arange(0.0, 10001.0, 10.0)
        frame = pd.DataFrame(
            {
                "time": time,
                "ax": np.sin(time / 100.0),
                "ay": np.cos(time / 100.0),
                "az": np.sin(time / 50.0),
            }
        )
        windows = extract_window_feature_table(frame, "new_trial", "new.csv")
        self.assertEqual(len(windows), 5)
        self.assertEqual(windows["window_id"].tolist(), [1, 2, 3, 4, 5])

    def test_short_new_trial_is_rejected(self) -> None:
        time = np.arange(0.0, 8500.0, 10.0)
        frame = pd.DataFrame(
            {"time": time, "ax": 0.0, "ay": 0.0, "az": 1.0}
        )
        with self.assertRaisesRegex(ValueError, "Window 5 is incomplete"):
            extract_window_feature_table(frame, "short_trial", "short.csv")

    def test_majority_vote_uses_all_windows(self) -> None:
        self.assertEqual(_majority_vote([0.5, 0.5, 0.75, 0.5, 0.75]), 0.5)


if __name__ == "__main__":
    unittest.main()
