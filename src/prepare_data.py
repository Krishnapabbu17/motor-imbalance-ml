"""Split the authoritative cleaned workbook into measurement-preserving trial CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_COLUMNS = ("imbalance_g", "trial", "timestamp_ms", "x", "y", "z")
OUTPUT_COLUMNS = ("time", "ax", "ay", "az")


def mass_folder(mass_g: float) -> str:
    return f"{mass_g:.2f}g"


def prepare_workbook(source: Path, output_dir: Path) -> pd.DataFrame:
    frame = pd.read_excel(source)
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing source columns: {', '.join(missing)}")

    numeric = frame.loc[:, SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid = int(numeric.isna().sum().sum())
    if invalid:
        raise ValueError(f"Workbook contains {invalid} missing or nonnumeric values.")

    records: list[dict[str, object]] = []
    for (mass_g, trial), group in numeric.groupby(
        ["imbalance_g", "trial"], sort=True
    ):
        timestamps = group["timestamp_ms"].to_numpy(dtype=float)
        if len(group) < 4 or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Invalid timestamps for mass {mass_g}, trial {trial}.")

        folder = output_dir / mass_folder(float(mass_g))
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"mass_{float(mass_g):.2f}g_trial_{int(trial):02d}.csv"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing trial file: {path}")

        exported = group.loc[:, ["timestamp_ms", "x", "y", "z"]].copy()
        exported.columns = OUTPUT_COLUMNS
        exported.to_csv(path, index=False)
        intervals = np.diff(timestamps)
        records.append(
            {
                "trial_id": path.stem,
                "mass_g": float(mass_g),
                "trial": int(trial),
                "source_rows": len(group),
                "start_timestamp_ms": float(timestamps[0]),
                "end_timestamp_ms": float(timestamps[-1]),
                "duration_ms": float(timestamps[-1] - timestamps[0]),
                "median_sample_rate_hz": float(1000.0 / np.median(intervals)),
                "output_file": path.as_posix(),
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/source/cleaned_experimental_motor_data.xlsx"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/trial_manifest.csv"),
    )
    args = parser.parse_args()

    manifest = prepare_workbook(args.source, args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"Created {len(manifest)} cleaned experimental trial files.")
    print(f"Saved {args.manifest}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
