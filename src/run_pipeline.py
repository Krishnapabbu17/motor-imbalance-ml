"""Run validation, feature extraction, training, and evaluation in order."""

from __future__ import annotations

from pathlib import Path

from src.features import build_feature_table
from src.validate_data import validate_dataset


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def main() -> int:
    quality = validate_dataset(RAW_DIR)
    if quality.empty:
        print("No experimental CSV files were found in data/raw.")
        print("Add the replacement trials to the five mass folders, then run again.")
        return 0

    RESULTS_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    quality.to_csv(RESULTS_DIR / "tables" / "data_quality.csv", index=False)
    failures = quality[quality["status"] != "PASS"]
    if not failures.empty:
        print(f"Stopped: {len(failures)} trial(s) failed validation.")
        print("Review results/tables/data_quality.csv before modeling.")
        return 1

    feature_table = build_feature_table(RAW_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = PROCESSED_DIR / "trial_features.csv"
    feature_table.to_csv(feature_path, index=False)
    print(f"Created {feature_path} with {len(feature_table)} trial rows.")

    from src.train_models import train_and_compare

    comparison = train_and_compare(feature_table, RESULTS_DIR)
    print("Model comparison complete:")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
