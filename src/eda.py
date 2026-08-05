"""Leakage-safe exploratory analysis and feature screening without model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_SEED = 42
METADATA_COLUMNS = {
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
CLASS_LABELS = {0.0: "0.00 g", 0.25: "0.25 g", 0.5: "0.50 g", 0.75: "0.75 g", 1.0: "1.00 g"}


def eta_squared(values: pd.Series, labels: pd.Series) -> float:
    """Return one-way ANOVA eta-squared as a descriptive effect size."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & labels.notna()
    numeric = numeric.loc[valid]
    classes = labels.loc[valid]
    if numeric.empty:
        return 0.0
    grand_mean = float(numeric.mean())
    total = float(np.sum((numeric - grand_mean) ** 2))
    if total == 0.0:
        return 0.0
    between = 0.0
    for label in sorted(classes.unique()):
        group = numeric.loc[classes == label]
        between += len(group) * (float(group.mean()) - grand_mean) ** 2
    return float(between / total)


def make_trial_assignments(
    features: pd.DataFrame, test_trials_per_class: int = 2
) -> pd.DataFrame:
    """Lock complete physical trials into development or untouched test sets."""
    trials = features.loc[:, ["trial_id", "mass_g"]].drop_duplicates()
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    for mass_g, group in trials.groupby("mass_g", sort=True):
        trial_ids = np.array(sorted(group["trial_id"].astype(str)))
        if len(trial_ids) <= test_trials_per_class:
            raise ValueError(f"Not enough trials for class {mass_g}.")
        test_ids = set(rng.choice(trial_ids, size=test_trials_per_class, replace=False))
        for trial_id in trial_ids:
            rows.append(
                {
                    "trial_id": trial_id,
                    "mass_g": float(mass_g),
                    "split": "test" if trial_id in test_ids else "development",
                    "selection_seed": RANDOM_SEED,
                }
            )
    return pd.DataFrame(rows).sort_values(["mass_g", "trial_id"]).reset_index(drop=True)


def screen_features(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    correlation_threshold: float = 0.95,
    max_features: int = 15,
) -> pd.DataFrame:
    """Rank development-only features and remove near-duplicate measurements."""
    development = features.merge(
        assignments.loc[:, ["trial_id", "split"]], on="trial_id", validate="many_to_one"
    )
    development = development.loc[development["split"] == "development"].copy()
    feature_columns = [
        column
        for column in features.select_dtypes(include="number").columns
        if column not in METADATA_COLUMNS
    ]
    correlation = development[feature_columns].corr().abs().fillna(0.0)
    ranked = sorted(
        feature_columns,
        key=lambda column: eta_squared(development[column], development["mass_g"]),
        reverse=True,
    )

    selected: list[str] = []
    records: list[dict[str, object]] = []
    for column in ranked:
        series = development[column]
        redundant_with = next(
            (
                kept
                for kept in selected
                if float(correlation.loc[column, kept]) >= correlation_threshold
            ),
            "",
        )
        near_constant = int(series.nunique(dropna=True)) <= 1 or float(series.std()) == 0.0
        choose = not near_constant and not redundant_with and len(selected) < max_features
        if choose:
            selected.append(column)
        max_other_corr = float(
            correlation.loc[column].drop(labels=[column], errors="ignore").max()
        )
        records.append(
            {
                "feature": column,
                "eta_squared_development": eta_squared(series, development["mass_g"]),
                "missing_values": int(series.isna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "standard_deviation": float(series.std()),
                "max_absolute_correlation": max_other_corr,
                "redundant_with_selected": redundant_with,
                "recommended": choose,
                "reason": (
                    "selected"
                    if choose
                    else "near_constant"
                    if near_constant
                    else f"redundant_with_{redundant_with}"
                    if redundant_with
                    else "below_feature_limit"
                ),
            }
        )
    return pd.DataFrame(records)


def class_summary(
    development: pd.DataFrame, selected_features: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mass_g, group in development.groupby("mass_g", sort=True):
        for feature in selected_features:
            values = group[feature]
            rows.append(
                {
                    "mass_g": float(mass_g),
                    "feature": feature,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                    "window_count": len(group),
                    "trial_count": int(group["trial_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _save_representative_signals(
    data_dir: Path, assignments: pd.DataFrame, output: Path
) -> None:
    figure, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True, sharey=True)
    for axis, mass_g in zip(axes, sorted(CLASS_LABELS), strict=True):
        candidates = assignments.loc[
            (assignments["mass_g"] == mass_g)
            & (assignments["split"] == "development"),
            "trial_id",
        ]
        trial_id = sorted(candidates.astype(str))[0]
        path = data_dir / f"{mass_g:.2f}g" / f"{trial_id}.csv"
        frame = pd.read_csv(path)
        relative_time = frame["time"] - float(frame["time"].iloc[0])
        shown = frame.loc[relative_time < 2000.0].copy()
        shown_time = relative_time.loc[shown.index] / 1000.0
        dynamic = shown[["ax", "ay", "az"]] - shown[["ax", "ay", "az"]].mean()
        magnitude = np.sqrt(np.sum(dynamic.to_numpy() ** 2, axis=1))
        axis.plot(shown_time, magnitude, linewidth=0.9)
        axis.set_ylabel(CLASS_LABELS[mass_g])
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time from trial start (s)")
    figure.supylabel("Dynamic acceleration magnitude (m/s²)")
    figure.suptitle("Representative measured vibration signals (development trials)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_boxplot_grid(
    development: pd.DataFrame,
    features: list[str],
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.flat, features, strict=True):
        sns.boxplot(data=development, x="mass_g", y=feature, ax=axis, color="#6FA8DC")
        sns.stripplot(
            data=development,
            x="mass_g",
            y=feature,
            ax=axis,
            color="#1F2937",
            size=2,
            alpha=0.35,
        )
        axis.set_xlabel("Imbalance mass (g)")
        axis.set_ylabel(feature.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_feature_ranking(screening: pd.DataFrame, output: Path) -> None:
    top = screening.head(15).sort_values("eta_squared_development")
    figure, axis = plt.subplots(figsize=(10, 7))
    colors = ["#2E75B6" if value else "#A5A5A5" for value in top["recommended"]]
    axis.barh(top["feature"], top["eta_squared_development"], color=colors)
    axis.set_xlabel("Development-only eta² (higher = stronger class separation)")
    axis.set_title("Top exploratory feature effect sizes")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_selected_correlation(
    development: pd.DataFrame, selected_features: list[str], output: Path
) -> None:
    correlation = development[selected_features].corr()
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        square=True,
        linewidths=0.2,
        ax=axis,
    )
    axis.set_title("Correlation among recommended development features")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output: Path,
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    screening: pd.DataFrame,
) -> None:
    recommended = screening.loc[screening["recommended"], "feature"].tolist()
    top = screening.head(8)
    split_counts = assignments.groupby(["split", "mass_g"]).size()
    development = features.merge(
        assignments.loc[:, ["trial_id", "split"]], on="trial_id", validate="many_to_one"
    )
    development = development.loc[development["split"] == "development"]
    class_means = development.groupby("mass_g").mean(numeric_only=True)
    healthy_centroid = float(class_means.loc[0.0, "magnitude_spectral_centroid_hz"])
    faulted_centroid = float(
        class_means.loc[class_means.index > 0, "magnitude_spectral_centroid_hz"].mean()
    )
    healthy_entropy = float(class_means.loc[0.0, "magnitude_spectral_entropy"])
    faulted_entropy = float(
        class_means.loc[class_means.index > 0, "magnitude_spectral_entropy"].mean()
    )
    lines = [
        "# Exploratory data analysis and feature screening",
        "",
        "No machine-learning model was trained during this stage.",
        "",
        "## Dataset used",
        "",
        f"- {features['trial_id'].nunique()} physical trials.",
        f"- {len(features)} non-overlapping two-second windows.",
        f"- {len(sorted(features['mass_g'].unique()))} balanced imbalance classes.",
        "- No test-trial labels or values were used for feature ranking.",
        "",
        "## Locked evaluation split",
        "",
        "Two complete trials per class are reserved as an untouched final test set.",
        "The remaining eight trials per class form the development set.",
        "All future cross-validation must group rows by `trial_id`.",
        "",
        "| Split | Trials per class | Total trials |",
        "|---|---:|---:|",
        f"| Development | {int(split_counts['development'].iloc[0])} | {int((assignments['split'] == 'development').sum())} |",
        f"| Test | {int(split_counts['test'].iloc[0])} | {int((assignments['split'] == 'test').sum())} |",
        "",
        "## Key development-data findings",
        "",
        f"- Mean dynamic magnitude rose from {class_means.loc[0.0, 'magnitude_mean']:.3f} m/s^2 at 0.00 g "
        f"to {class_means.loc[1.0, 'magnitude_mean']:.3f} m/s^2 at 1.00 g.",
        f"- Y-axis standard deviation rose from {class_means.loc[0.0, 'ay_std']:.3f} m/s^2 at 0.00 g "
        f"to {class_means.loc[1.0, 'ay_std']:.3f} m/s^2 at 1.00 g.",
        f"- Mean magnitude spectral centroid was {healthy_centroid:.2f} Hz for 0.00 g versus "
        f"{faulted_centroid:.2f} Hz across the four faulted classes.",
        f"- Mean magnitude spectral entropy was {healthy_entropy:.3f} for 0.00 g versus "
        f"{faulted_entropy:.3f} across the faulted classes.",
        f"- Screening retained {len(recommended)} of {len(screening)} candidate features; "
        f"{int(screening['reason'].str.startswith('redundant').sum())} were removed as near-duplicates.",
        "",
        "## Strongest development-only class effects",
        "",
        "| Feature | Eta-squared | Recommended |",
        "|---|---:|:---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.eta_squared_development:.3f} | "
            f"{'yes' if row.recommended else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Recommended exploratory feature set",
            "",
            *[f"- `{feature}`" for feature in recommended],
            "",
            "The shortlist was built only from development trials. Features were ranked by",
            "descriptive class effect size, then measurements with absolute correlation of",
            "0.95 or greater to an already selected feature were removed. The list is capped",
            "at 15 features to reduce overfitting risk.",
            "",
            "## Interpretation limits",
            "",
            "Feature effect sizes are exploratory and are not model accuracy. Final feature",
            "selection and scaling must occur inside grouped cross-validation. The reserved",
            "test trials must be evaluated only after the model family and tuning plan are",
            "locked.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eda(
    feature_path: Path,
    data_dir: Path,
    split_path: Path,
    figures_dir: Path,
    tables_dir: Path,
    report_path: Path,
) -> None:
    features = pd.read_csv(feature_path)
    if features.empty or features.isna().any().any():
        raise ValueError("Window feature table is empty or contains missing values.")

    assignments = make_trial_assignments(features)
    assignments["split"].value_counts()
    split_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(split_path, index=False)

    screening = screen_features(features, assignments)
    screening.to_csv(tables_dir / "feature_screening.csv", index=False)
    selected = screening.loc[screening["recommended"], "feature"].tolist()
    screening.loc[screening["recommended"]].to_csv(
        tables_dir / "recommended_features.csv", index=False
    )

    development = features.merge(
        assignments.loc[:, ["trial_id", "split"]], on="trial_id", validate="many_to_one"
    )
    development = development.loc[development["split"] == "development"].copy()
    class_summary(development, selected).to_csv(
        tables_dir / "development_class_summary.csv", index=False
    )

    sns.set_theme(style="whitegrid", context="notebook")
    _save_representative_signals(
        data_dir, assignments, figures_dir / "representative_signals.png"
    )
    _save_boxplot_grid(
        development,
        ["ax_std", "ay_std", "az_std", "magnitude_mean"],
        figures_dir / "vibration_amplitude_by_class.png",
        "Vibration amplitude by imbalance class (development trials)",
    )
    _save_boxplot_grid(
        development,
        [
            "ax_dominant_frequency_hz",
            "ay_dominant_frequency_hz",
            "az_dominant_frequency_hz",
            "magnitude_spectral_entropy",
        ],
        figures_dir / "frequency_features_by_class.png",
        "Frequency-domain features by imbalance class (development trials)",
    )
    _save_feature_ranking(screening, figures_dir / "feature_effect_ranking.png")
    _save_selected_correlation(
        development, selected, figures_dir / "recommended_feature_correlation.png"
    )
    write_report(report_path, features, assignments, screening)
    print(f"Locked {len(assignments)} physical trials into development/test splits.")
    print(f"Recommended {len(selected)} exploratory features from development trials.")
    print(f"Saved EDA report to {report_path}.")
    print("No model was trained.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("data/processed/window_features_2s.csv")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument(
        "--split-path", type=Path, default=Path("data/splits/trial_assignments.csv")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument(
        "--report-path", type=Path, default=Path("results/eda_report.md")
    )
    args = parser.parse_args()
    run_eda(
        args.features,
        args.data_dir,
        args.split_path,
        args.figures_dir,
        args.tables_dir,
        args.report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
