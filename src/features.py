"""Trial-level feature extraction for motor vibration signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.validate_data import REQUIRED_COLUMNS, find_trial_files


def _signal_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    standard_deviation = float(np.std(values, ddof=0))

    if standard_deviation > 0:
        normalized = centered / standard_deviation
        skewness = float(np.mean(normalized**3))
        excess_kurtosis = float(np.mean(normalized**4) - 3.0)
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": standard_deviation,
        f"{prefix}_rms_centered": float(np.sqrt(np.mean(centered**2))),
        f"{prefix}_peak_abs_centered": float(np.max(np.abs(centered))),
        f"{prefix}_peak_to_peak": float(np.ptp(values)),
        f"{prefix}_energy_centered": float(np.sum(centered**2)),
        f"{prefix}_skewness": skewness,
        f"{prefix}_excess_kurtosis": excess_kurtosis,
    }


def _frequency_features(
    time_ms: np.ndarray, values: np.ndarray, prefix: str
) -> dict[str, float]:
    """Calculate simple spectral features after uniform-time interpolation."""
    time_seconds = np.asarray(time_ms, dtype=float) / 1000.0
    values = np.asarray(values, dtype=float)

    if len(values) < 4 or np.any(np.diff(time_seconds) <= 0):
        return {
            f"{prefix}_dominant_frequency_hz": np.nan,
            f"{prefix}_spectral_centroid_hz": np.nan,
        }

    uniform_time = np.linspace(time_seconds[0], time_seconds[-1], len(time_seconds))
    uniform_values = np.interp(uniform_time, time_seconds, values)
    centered = uniform_values - np.mean(uniform_values)
    dt = float(np.median(np.diff(uniform_time)))
    frequencies = np.fft.rfftfreq(len(centered), d=dt)
    power = np.abs(np.fft.rfft(centered)) ** 2

    nonzero_power = power.copy()
    nonzero_power[0] = 0.0
    if float(np.sum(nonzero_power)) == 0.0:
        dominant_frequency = 0.0
        spectral_centroid = 0.0
    else:
        dominant_frequency = float(frequencies[int(np.argmax(nonzero_power))])
        spectral_centroid = float(
            np.sum(frequencies * nonzero_power) / np.sum(nonzero_power)
        )

    return {
        f"{prefix}_dominant_frequency_hz": dominant_frequency,
        f"{prefix}_spectral_centroid_hz": spectral_centroid,
    }


def extract_trial_features(frame: pd.DataFrame) -> dict[str, float]:
    """Extract time- and frequency-domain features from one trial."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    numeric = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise")
    time_ms = numeric["time"].to_numpy(dtype=float)
    axes = numeric[["ax", "ay", "az"]].to_numpy(dtype=float)
    centered_axes = axes - np.mean(axes, axis=0, keepdims=True)
    magnitude = np.sqrt(np.sum(centered_axes**2, axis=1))

    intervals = np.diff(time_ms)
    if len(time_ms) < 2 or np.any(intervals <= 0):
        raise ValueError("Trial timestamps must be strictly increasing.")

    features: dict[str, float] = {
        "sample_count": float(len(frame)),
        "duration_ms": float(time_ms[-1] - time_ms[0]),
        "median_sample_rate_hz": float(1000.0 / np.median(intervals)),
    }

    signals = {
        "ax": axes[:, 0],
        "ay": axes[:, 1],
        "az": axes[:, 2],
        "magnitude": magnitude,
    }
    for name, signal in signals.items():
        features.update(_signal_features(signal, name))
        features.update(_frequency_features(time_ms, signal, name))

    return features


def mass_from_folder(path: Path) -> float:
    """Convert a class folder such as '0.25g' into a numeric label."""
    name = path.parent.name.lower()
    if not name.endswith("g"):
        raise ValueError(f"Expected a mass folder ending in 'g': {path.parent}")
    return float(name[:-1])


def build_feature_table(raw_dir: Path) -> pd.DataFrame:
    """Create one feature row per complete trial."""
    rows: list[dict[str, object]] = []
    for path in find_trial_files(raw_dir):
        frame = pd.read_csv(path)
        rows.append(
            {
                "trial_id": path.stem,
                "mass_g": mass_from_folder(path),
                "source_file": str(path),
                **extract_trial_features(frame),
            }
        )
    return pd.DataFrame(rows)
