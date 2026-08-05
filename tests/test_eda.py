import unittest

import pandas as pd

from src.eda import eta_squared, make_trial_assignments, screen_features


class EdaTests(unittest.TestCase):
    def test_eta_squared_is_one_for_perfect_class_separation(self) -> None:
        values = pd.Series([0.0, 0.0, 10.0, 10.0])
        labels = pd.Series([0, 0, 1, 1])
        self.assertAlmostEqual(eta_squared(values, labels), 1.0)

    def test_split_reserves_two_complete_trials_per_class(self) -> None:
        rows = []
        for mass_g in (0.0, 0.25):
            for trial in range(1, 11):
                for window in range(5):
                    rows.append(
                        {
                            "trial_id": f"mass_{mass_g:.2f}g_trial_{trial:02d}",
                            "mass_g": mass_g,
                            "window_id": window + 1,
                        }
                    )
        assignments = make_trial_assignments(pd.DataFrame(rows))
        counts = assignments.groupby(["mass_g", "split"]).size().to_dict()
        self.assertEqual(counts[(0.0, "development")], 8)
        self.assertEqual(counts[(0.0, "test")], 2)
        self.assertEqual(counts[(0.25, "development")], 8)
        self.assertEqual(counts[(0.25, "test")], 2)

    def test_screening_ignores_test_only_separation(self) -> None:
        features = pd.DataFrame(
            {
                "trial_id": ["d0", "d1", "t0", "t1"],
                "mass_g": [0.0, 1.0, 0.0, 1.0],
                "development_signal": [0.0, 10.0, 0.0, 0.0],
                "test_only_signal": [0.0, 0.0, 0.0, 10.0],
            }
        )
        assignments = pd.DataFrame(
            {
                "trial_id": ["d0", "d1", "t0", "t1"],
                "split": ["development", "development", "test", "test"],
            }
        )
        screening = screen_features(features, assignments, max_features=1)
        selected = screening.loc[screening["recommended"], "feature"].tolist()
        self.assertEqual(selected, ["development_signal"])


if __name__ == "__main__":
    unittest.main()
