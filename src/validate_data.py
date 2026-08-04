"""Validation helpers for experimental motor vibration trials."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("time", "ax", "ay", "az")


def find_trial_files(raw_dir: Path) -> list[Path]:
    """Return all trial CSV files in a stable order."""
    return sorted(path for path in raw_dir.rglob("*.csv") if path.is_file())


def validate_trial(path: Path) -> dict[str, object]:
    """Validate one trial and return an auditable summary."""
    issues: list[str] = []

    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas provides the useful parser detail
        return {
            "file": str(path),
            "rows": 0,
            "duration_ms": np.nan,
            "sample_rate_hz": np.nan,
            "status": "FAIL",
            "issues": f"could not read CSV: {exc}",
        }

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        issues.append(f"missing columns: {', '.join(missing)}")
        return {
            "file": str(path),
            "rows": len(frame),
            "duration_ms": np.nan,
            "sample_rate_hz": np.nan,
            "status": "FAIL",
            "issues": "; ".join(issues),
        }

    numeric = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_values = int(numeric.isna().sum().sum())
    if invalid_values:
        issues.append(f"{invalid_values} missing or nonnumeric values")

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        issues.append(f"{duplicate_rows} duplicate rows")

    time = numeric["time"].dropna().to_numpy(dtype=float)
    intervals = np.diff(time)
    if len(time) < 2:
        issues.append("fewer than two valid timestamps")
        duration_ms = np.nan
        sample_rate_hz = np.nan
    else:
        duration_ms = float(time[-1] - time[0])
        if np.any(intervals <= 0):
            issues.append("timestamps are not strictly increasing")
        positive_intervals = intervals[intervals > 0]
        sample_rate_hz = (
            float(1000.0 / np.median(positive_intervals))
            if len(positive_intervals)
            else np.nan
        )

    return {
        "file": str(path),
        "rows": len(frame),
        "duration_ms": duration_ms,
        "sample_rate_hz": sample_rate_hz,
        "status": "PASS" if not issues else "FAIL",
        "issues": "; ".join(issues),
    }


def validate_dataset(raw_dir: Path) -> pd.DataFrame:
    """Validate every trial beneath a raw-data directory."""
    return pd.DataFrame(validate_trial(path) for path in find_trial_files(raw_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/data_quality.csv"),
    )
    args = parser.parse_args()

    report = validate_dataset(args.raw_dir)
    if report.empty:
        print(f"No CSV trial files found under {args.raw_dir}.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    failed = int((report["status"] != "PASS").sum())
    print(f"Validated {len(report)} trials; {failed} require attention.")
    print(f"Saved {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
