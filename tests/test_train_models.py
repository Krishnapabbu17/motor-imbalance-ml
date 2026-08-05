import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.train_models import (
    EffectSizeCorrelationSelector,
    _development_frame,
    majority_trial_predictions,
)


class ModelComparisonTests(unittest.TestCase):
    def test_selector_removes_near_duplicate_feature(self) -> None:
        features = pd.DataFrame(
            {
                "strong": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
                "duplicate": [0.0, 0.2, 2.0, 2.2, 4.0, 4.2],
                "different": [0.0, 1.0, 0.2, 1.2, 0.4, 1.4],
            }
        )
        labels = pd.Series(["0", "0", "1", "1", "2", "2"])
        selector = EffectSizeCorrelationSelector(max_features=3).fit(features, labels)

        self.assertIn("strong", selector.selected_features_)
        self.assertNotIn("duplicate", selector.selected_features_)
        self.assertIn("different", selector.selected_features_)

    def test_majority_vote_produces_one_row_per_trial(self) -> None:
        predictions = pd.DataFrame(
            {
                "model": ["m"] * 5,
                "trial_id": ["trial_1"] * 5,
                "mass_g": [0.5] * 5,
                "predicted_mass_g": [0.5, 0.5, 0.5, 0.75, 0.75],
                "outer_fold": [1] * 5,
            }
        )
        trials = majority_trial_predictions(predictions)

        self.assertEqual(len(trials), 1)
        self.assertEqual(trials.iloc[0]["predicted_mass_g"], 0.5)

    def test_development_loader_excludes_reserved_test_trial(self) -> None:
        features = pd.DataFrame(
            {
                "trial_id": ["development_trial", "test_trial"],
                "mass_g": [0.0, 0.0],
                "window_id": [1, 1],
                "feature_value": [1.0, 999.0],
            }
        )
        assignment_rows = []
        for index in range(10):
            assignment_rows.append(
                {
                    "trial_id": "test_trial" if index == 0 else f"reserved_{index}",
                    "mass_g": 0.0,
                    "split": "test",
                }
            )
        assignment_rows.append(
            {"trial_id": "development_trial", "mass_g": 0.0, "split": "development"}
        )
        assignments = pd.DataFrame(assignment_rows)

        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.csv"
            assignment_path = Path(directory) / "assignments.csv"
            features.to_csv(feature_path, index=False)
            assignments.to_csv(assignment_path, index=False)
            development, columns, _ = _development_frame(feature_path, assignment_path)

        self.assertEqual(development["trial_id"].tolist(), ["development_trial"])
        self.assertEqual(columns, ["feature_value"])
        self.assertNotIn(999.0, development["feature_value"].tolist())


if __name__ == "__main__":
    unittest.main()
