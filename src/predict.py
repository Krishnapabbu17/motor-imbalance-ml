"""Predict motor imbalance from one new approximately 10-second MPU6050 CSV."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import MODEL_METADATA_COLUMNS, extract_window_feature_table
from src.train_models import EffectSizeCorrelationSelector
from src.validate_data import REQUIRED_COLUMNS, validate_trial


CLASS_LABELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load_and_validate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    quality = validate_trial(path)
    if quality["status"] != "PASS":
        raise ValueError(f"Input validation failed: {quality['issues']}")
    frame = pd.read_csv(path)
    return frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise")


def _load_locked_model(model_path: Path, config_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(
            "The local fitted model is missing. Run `python -m src.train_models` first."
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Locked model configuration is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # The locked artifact was created with `python -m src.train_models`, so pickle
    # recorded this custom selector under __main__. Expose the identical class
    # there when loading without changing or reserializing the evaluated model.
    setattr(sys.modules["__main__"], "EffectSizeCorrelationSelector", EffectSizeCorrelationSelector)
    model = joblib.load(model_path)
    selected = list(model.named_steps["selector"].selected_features_)
    if selected != list(config["selected_features"]):
        raise ValueError("Saved model does not match the locked feature configuration.")
    return model, config


def _majority_vote(predictions: list[float]) -> float:
    counts = Counter(float(value) for value in predictions)
    return float(sorted(counts, key=lambda value: (-counts[value], value))[0])


def predict_trial(
    input_path: Path,
    model_path: Path,
    config_path: Path,
) -> dict[str, object]:
    """Validate, featurize, and classify one new recording without fitting."""
    frame = _load_and_validate(input_path)
    windows = extract_window_feature_table(
        frame,
        trial_id=input_path.stem,
        source_file=str(input_path.resolve()),
    )
    model, config = _load_locked_model(model_path, config_path)
    feature_columns = [
        column
        for column in windows.select_dtypes(include="number").columns
        if column not in MODEL_METADATA_COLUMNS
    ]
    expected_columns = list(model.feature_names_in_)
    if feature_columns != expected_columns:
        raise ValueError("New-trial features do not match the model's training columns.")

    raw_predictions = model.predict(windows.loc[:, feature_columns]).astype(float)
    predicted_mass = _majority_vote(raw_predictions.tolist())
    vote_counts = {
        f"{label:.2f}": int(np.sum(raw_predictions == label)) for label in CLASS_LABELS
    }
    result: dict[str, object] = {
        "input_file": str(input_path.resolve()),
        "model": config["model"],
        "predicted_mass_g": predicted_mass,
        "window_count": int(len(windows)),
        "window_predictions_g": [float(value) for value in raw_predictions],
        "vote_counts": vote_counts,
        "selected_features": list(config["selected_features"]),
        "note": "Prediction only; the input was not used to fit or tune the model.",
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(windows.loc[:, feature_columns])
        classes = [float(value) for value in model.classes_]
        mean_probabilities = np.mean(probabilities, axis=0)
        result["mean_model_scores"] = {
            f"{label:.2f}": float(
                mean_probabilities[classes.index(label)] if label in classes else 0.0
            )
            for label in CLASS_LABELS
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV containing time, ax, ay, and az")
    parser.add_argument(
        "--model", type=Path, default=Path("results/models/development_candidate.joblib")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("results/tables/locked_model_config.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise only print the result",
    )
    args = parser.parse_args()
    try:
        result = predict_trial(args.input, args.model, args.config)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"Predicted imbalance: {result['predicted_mass_g']:.2f} g")
    print(f"Window votes: {result['vote_counts']}")
    if args.output:
        print(f"Saved details to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
