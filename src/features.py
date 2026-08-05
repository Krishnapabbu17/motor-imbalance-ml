"""Feature extraction for cleaned experimental motor-vibration trials."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.validate_data import REQUIRED_COLUMNS, find_trial_files


SIGNAL_NAMES = ("ax", "ay", "az", "magnitude")
MODEL_METADATA_COLUMNS = {
    "trial_id",
    "mass_g",
    "source_file",
    "feature_scope",
    "window_id",
    "window_start_ms",
    "window_end_ms",
    "sample_count",
    "duration_ms",
    "median_sample_rate_hz",
}


def _signal_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    standard_deviation = float(np.std(values, ddof=0))
    rms_centered = float(np.sqrt(np.mean(centered**2)))
    peak_abs_centered = float(np.max(np.abs(centered)))

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
        f"{prefix}_rms_centered": rms_centered,
        f"{prefix}_mean_abs_centered": float(np.mean(np.abs(centered))),
        f"{prefix}_median_abs_deviation": float(
            np.median(np.abs(values - np.median(values)))
        ),
        f"{prefix}_peak_abs_centered": peak_abs_centered,
        f"{prefix}_peak_to_peak": float(np.ptp(values)),
        f"{prefix}_crest_factor": (
            float(peak_abs_centered / rms_centered) if rms_centered > 0 else 0.0
        ),
        f"{prefix}_skewness": skewness,
        f"{prefix}_excess_kurtosis": excess_kurtosis,
    }


def _frequency_features(
    time_ms: np.ndarray, values: np.ndarray, prefix: str
) -> dict[str, float]:
    """Calculate spectral features after interpolation to a uniform time grid."""
    time_seconds = np.asarray(time_ms, dtype=float) / 1000.0
    values = np.asarray(values, dtype=float)
    empty = {
        f"{prefix}_dominant_frequency_hz": np.nan,
        f"{prefix}_spectral_centroid_hz": np.nan,
        f"{prefix}_spectral_entropy": np.nan,
        f"{prefix}_relative_power_0_5_hz": np.nan,
        f"{prefix}_relative_power_5_15_hz": np.nan,
        f"{prefix}_relative_power_15_30_hz": np.nan,
    }
    if len(values) < 4 or np.any(np.diff(time_seconds) <= 0):
        return empty

    uniform_time = np.linspace(time_seconds[0], time_seconds[-1], len(time_seconds))
    uniform_values = np.interp(uniform_time, time_seconds, values)
    centered = uniform_values - np.mean(uniform_values)
    dt = float(np.median(np.diff(uniform_time)))
    frequencies = np.fft.rfftfreq(len(centered), d=dt)
    power = np.abs(np.fft.rfft(centered)) ** 2
    power[0] = 0.0
    total_power = float(np.sum(power))

    if total_power == 0.0:
        return {key: 0.0 for key in empty}

    probabilities = power / total_power
    nonzero = probabilities[probabilities > 0]
    entropy_denominator = np.log(len(nonzero)) if len(nonzero) > 1 else 1.0
    spectral_entropy = float(-np.sum(nonzero * np.log(nonzero)) / entropy_denominator)

    def relative_power(low_hz: float, high_hz: float) -> float:
        mask = (frequencies >= low_hz) & (frequencies < high_hz)
        return float(np.sum(power[mask]) / total_power)

    return {
        f"{prefix}_dominant_frequency_hz": float(
            frequencies[int(np.argmax(power))]
        ),
        f"{prefix}_spectral_centroid_hz": float(
            np.sum(frequencies * power) / total_power
        ),
        f"{prefix}_spectral_entropy": spectral_entropy,
        f"{prefix}_relative_power_0_5_hz": relative_power(0.0, 5.0),
        f"{prefix}_relative_power_5_15_hz": relative_power(5.0, 15.0),
        f"{prefix}_relative_power_15_30_hz": relative_power(15.0, 30.0),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def extract_trial_features(frame: pd.DataFrame) -> dict[str, float]:
    """Extract time- and frequency-domain features from one signal segment."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    numeric = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise")
    time_ms = numeric["time"].to_numpy(dtype=float)
    axes = numeric[["ax", "ay", "az"]].to_numpy(dtype=float)
    intervals = np.diff(time_ms)
    if len(time_ms) < 4 or np.any(intervals <= 0):
        raise ValueError("A segment needs at least four strictly increasing timestamps.")

    dynamic_axes = axes - np.mean(axes, axis=0, keepdims=True)
    dynamic_magnitude = np.sqrt(np.sum(dynamic_axes**2, axis=1))
    features: dict[str, float] = {
        "sample_count": float(len(frame)),
        "duration_ms": float(time_ms[-1] - time_ms[0]),
        "median_sample_rate_hz": float(1000.0 / np.median(intervals)),
        "axis_correlation_xy": _safe_correlation(axes[:, 0], axes[:, 1]),
        "axis_correlation_xz": _safe_correlation(axes[:, 0], axes[:, 2]),
        "axis_correlation_yz": _safe_correlation(axes[:, 1], axes[:, 2]),
    }

    signals = {
        "ax": axes[:, 0],
        "ay": axes[:, 1],
        "az": axes[:, 2],
        "magnitude": dynamic_magnitude,
    }
    for name, signal in signals.items():
        features.update(_signal_features(signal, name))
        features.update(_frequency_features(time_ms, signal, name))
    return features


def mass_from_folder(path: Path) -> float:
    name = path.parent.name.lower()
    if not name.endswith("g"):
        raise ValueError(f"Expected a mass folder ending in 'g': {path.parent}")
    return float(name[:-1])


def build_trial_feature_table(data_dir: Path) -> pd.DataFrame:
    """Create one full-recording feature row per experimental trial."""
    rows: list[dict[str, object]] = []
    for path in find_trial_files(data_dir):
        frame = pd.read_csv(path)
        rows.append(
            {
                "trial_id": path.stem,
                "mass_g": mass_from_folder(path),
                "source_file": path.as_posix(),
                "feature_scope": "full_trial",
                **extract_trial_features(frame),
            }
        )
    return pd.DataFrame(rows)


def build_window_feature_table(
    data_dir: Path,
    window_ms: float = 2000.0,
    analysis_duration_ms: float = 10000.0,
) -> pd.DataFrame:
    """Create balanced, non-overlapping windows while retaining trial identity."""
    rows: list[dict[str, object]] = []
    window_count = int(analysis_duration_ms // window_ms)
    for path in find_trial_files(data_dir):
        frame = pd.read_csv(path)
        numeric_time = pd.to_numeric(frame["time"], errors="raise")
        relative_time = numeric_time - float(numeric_time.iloc[0])
        for window_id in range(window_count):
            start_ms = window_id * window_ms
            end_ms = start_ms + window_ms
            mask = (relative_time >= start_ms) & (relative_time < end_ms)
            segment = frame.loc[mask].copy()
            if len(segment) < 4:
                raise ValueError(f"Too few samples in window {window_id} of {path}")
            segment_duration = float(segment["time"].iloc[-1] - segment["time"].iloc[0])
            if segment_duration < 0.9 * window_ms:
                raise ValueError(f"Incomplete window {window_id} of {path}")
            rows.append(
                {
                    "trial_id": path.stem,
                    "mass_g": mass_from_folder(path),
                    "source_file": path.as_posix(),
                    "feature_scope": "2_second_window",
                    "window_id": window_id + 1,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    **extract_trial_features(segment),
                }
            )
    return pd.DataFrame(rows)


def build_feature_table(data_dir: Path) -> pd.DataFrame:
    """Backward-compatible alias for full-trial features."""
    return build_trial_feature_table(data_dir)


def build_feature_dictionary(columns: list[str]) -> pd.DataFrame:
    """Describe generated columns and identify future model inputs."""
    descriptions = {
        "trial_id": "Unique experimental trial identifier.",
        "mass_g": "Measured imbalance class in grams; future prediction target.",
        "source_file": "Relative path to the cleaned trial CSV.",
        "feature_scope": "Full trial or non-overlapping 2-second window.",
        "window_id": "One-based window number within a trial.",
        "window_start_ms": "Window start relative to the first trial timestamp.",
        "window_end_ms": "Window end relative to the first trial timestamp.",
        "sample_count": "Number of measured samples in the segment.",
        "duration_ms": "Elapsed time covered by the segment.",
        "median_sample_rate_hz": "Sampling rate estimated from median timestamp spacing.",
        "axis_correlation_xy": "Pearson correlation between x and y acceleration.",
        "axis_correlation_xz": "Pearson correlation between x and z acceleration.",
        "axis_correlation_yz": "Pearson correlation between y and z acceleration.",
    }
    rows = []
    for column in columns:
        if column in descriptions:
            description = descriptions[column]
        elif "dominant_frequency_hz" in column:
            description = "Frequency with maximum non-DC spectral power."
        elif "spectral_centroid_hz" in column:
            description = "Power-weighted mean frequency."
        elif "spectral_entropy" in column:
            description = "Normalized spread of spectral power from 0 to 1."
        elif "relative_power" in column:
            description = "Fraction of non-DC power within the named frequency band."
        elif column.endswith("_mean"):
            description = "Arithmetic mean of the signal."
        elif column.endswith("_std"):
            description = "Population standard deviation of the signal."
        elif "rms_centered" in column:
            description = "Root-mean-square after removing the signal mean."
        elif "mean_abs_centered" in column:
            description = "Mean absolute deviation from the signal mean."
        elif "median_abs_deviation" in column:
            description = "Median absolute deviation from the signal median."
        elif "peak_abs_centered" in column:
            description = "Largest absolute deviation from the signal mean."
        elif "peak_to_peak" in column:
            description = "Maximum signal value minus minimum signal value."
        elif "crest_factor" in column:
            description = "Centered absolute peak divided by centered RMS."
        elif "skewness" in column:
            description = "Third standardized moment of the signal."
        elif "excess_kurtosis" in column:
            description = "Fourth standardized moment minus three."
        else:
            description = "Derived feature."
        rows.append(
            {
                "column": column,
                "role": "metadata" if column in MODEL_METADATA_COLUMNS else "feature",
                "use_for_future_model": column not in MODEL_METADATA_COLUMNS,
                "description": description,
            }
        )
    return pd.DataFrame(rows)
