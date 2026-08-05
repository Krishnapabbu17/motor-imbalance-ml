import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluate_models import aggregate_trial_predictions, load_final_test_frame


class FinalEvaluationTests(unittest.TestCase):
    def test_final_loader_uses_only_two_trials_per_class(self) -> None:
        assignments = []
        features = []
        for mass in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for index in range(3):
                trial_id = f"mass_{mass}_{index}"
                split = "test" if index < 2 else "development"
                assignments.append({"trial_id": trial_id, "mass_g": mass, "split": split})
                features.append(
                    {
                        "trial_id": trial_id,
                        "mass_g": mass,
                        "window_id": 1,
                        "feature_value": mass + index,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.csv"
            assignment_path = Path(directory) / "assignments.csv"
            pd.DataFrame(features).to_csv(feature_path, index=False)
            pd.DataFrame(assignments).to_csv(assignment_path, index=False)
            final_test, columns, _ = load_final_test_frame(feature_path, assignment_path)

        self.assertEqual(final_test["trial_id"].nunique(), 10)
        self.assertEqual(set(final_test["split"]), {"test"})
        self.assertEqual(columns, ["feature_value"])

    def test_trial_vote_tie_is_deterministic(self) -> None:
        windows = pd.DataFrame(
            {
                "trial_id": ["trial_1"] * 4,
                "mass_g": [0.5] * 4,
                "predicted_mass_g": [0.5, 0.5, 0.75, 0.75],
            }
        )
        trials = aggregate_trial_predictions(windows)
        self.assertEqual(float(trials.iloc[0]["predicted_mass_g"]), 0.5)


if __name__ == "__main__":
    unittest.main()
