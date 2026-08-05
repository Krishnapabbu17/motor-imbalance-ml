"""Validate trials and generate features; model training is intentionally disabled."""

from __future__ import annotations

from pathlib import Path

from src.features import (
    build_feature_dictionary,
    build_trial_feature_table,
    build_window_feature_table,
)
from src.validate_data import validate_dataset


DATA_DIR = Path("data/cleaned")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def main() -> int:
    quality = validate_dataset(DATA_DIR)
    if quality.empty:
        print("No cleaned experimental CSV files were found in data/cleaned.")
        return 0

    RESULTS_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    quality_path = RESULTS_DIR / "tables" / "data_quality.csv"
    quality.to_csv(quality_path, index=False)
    failures = quality[quality["status"] != "PASS"]
    if not failures.empty:
        print(f"Stopped: {len(failures)} trial(s) failed validation.")
        print(f"Review {quality_path}.")
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    trial_features = build_trial_feature_table(DATA_DIR)
    window_features = build_window_feature_table(DATA_DIR)
    trial_path = PROCESSED_DIR / "trial_features.csv"
    window_path = PROCESSED_DIR / "window_features_2s.csv"
    dictionary_path = PROCESSED_DIR / "feature_dictionary.csv"
    trial_features.to_csv(trial_path, index=False)
    window_features.to_csv(window_path, index=False)
    build_feature_dictionary(list(window_features.columns)).to_csv(
        dictionary_path, index=False
    )

    print(f"Validated {len(quality)} experimental trials.")
    print(f"Created {trial_path} with {len(trial_features)} rows.")
    print(f"Created {window_path} with {len(window_features)} balanced windows.")
    print(f"Created {dictionary_path}.")
    print("Stopped before model training, as requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
