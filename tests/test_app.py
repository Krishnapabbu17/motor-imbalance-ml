import unittest

from app import format_prediction, predict_from_upload


class AppTests(unittest.TestCase):
    def test_format_prediction_builds_five_window_rows(self) -> None:
        result = {
            "predicted_mass_g": 0.75,
            "window_predictions_g": [0.75, 0.75, 0.5, 0.75, 0.75],
            "mean_model_scores": {"0.00": 0.01, "0.25": 0.04, "0.50": 0.2, "0.75": 0.7, "1.00": 0.05},
        }
        message, windows, scores = format_prediction(result)
        self.assertIn("0.75 g", message)
        self.assertEqual(len(windows), 5)
        self.assertEqual(len(scores), 5)

    def test_missing_upload_has_actionable_message(self) -> None:
        message, windows, scores = predict_from_upload(None)
        self.assertIn("Select a CSV", message)
        self.assertTrue(windows.empty)
        self.assertTrue(scores.empty)


if __name__ == "__main__":
    unittest.main()
