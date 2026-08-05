"""Run the single locked-model evaluation on the reserved final-test trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.train_models import (
    CLASS_LABELS,
    METADATA_COLUMNS,
    EffectSizeCorrelationSelector,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_final_test_frame(
    feature_path: Path, assignment_path: Path
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load only the ten prespecified test trials and validate the split."""
    features = pd.read_csv(feature_path)
    assignments = pd.read_csv(assignment_path)
    if assignments["trial_id"].duplicated().any():
        raise ValueError("Trial split file contains duplicate trial IDs.")
    if set(assignments["split"]) != {"development", "test"}:
        raise ValueError("Trial split file must contain development and test only.")

    test_assignments = assignments.loc[assignments["split"] == "test"].copy()
    class_counts = test_assignments.groupby("mass_g")["trial_id"].nunique()
    expected_counts = pd.Series(2, index=CLASS_LABELS, dtype="int64")
    class_counts = class_counts.reindex(CLASS_LABELS, fill_value=0).astype("int64")
    if len(test_assignments) != 10 or not class_counts.equals(expected_counts):
        raise ValueError("Final test set must contain exactly two trials per class.")

    merged = features.merge(
        assignments.loc[:, ["trial_id", "mass_g", "split"]],
        on="trial_id",
        suffixes=("", "_assignment"),
        validate="many_to_one",
    )
    if not np.allclose(merged["mass_g"], merged["mass_g_assignment"]):
        raise ValueError("Class labels disagree between features and split assignments.")
    final_test = merged.loc[merged["split"] == "test"].copy()
    if final_test["trial_id"].nunique() != 10:
        raise ValueError("Not all ten reserved trials have feature rows.")

    feature_columns = [
        column
        for column in features.select_dtypes(include="number").columns
        if column not in METADATA_COLUMNS
    ]
    if final_test[feature_columns].isna().any().any():
        raise ValueError("Final-test feature table contains missing values.")
    return final_test, feature_columns, assignments


def aggregate_trial_predictions(window_predictions: pd.DataFrame) -> pd.DataFrame:
    """Use deterministic majority vote to produce one result per physical trial."""
    rows: list[dict[str, object]] = []
    for trial_id, group in window_predictions.groupby("trial_id", sort=True):
        counts = Counter(group["predicted_mass_g"].astype(float))
        predicted = sorted(counts, key=lambda value: (-counts[value], value))[0]
        true_values = group["mass_g"].astype(float).unique()
        if len(true_values) != 1:
            raise ValueError(f"Trial {trial_id} has inconsistent true labels.")
        rows.append(
            {
                "trial_id": trial_id,
                "mass_g": float(true_values[0]),
                "predicted_mass_g": float(predicted),
                "correct": bool(float(predicted) == float(true_values[0])),
                "window_votes": json.dumps(
                    {f"{label:.2f}": int(counts.get(label, 0)) for label in CLASS_LABELS},
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_summary(true: pd.Series, predicted: pd.Series) -> dict[str, float]:
    labels = [f"{label:.2f}" for label in CLASS_LABELS]
    true_strings = true.astype(float).map(lambda value: f"{value:.2f}")
    predicted_strings = predicted.astype(float).map(lambda value: f"{value:.2f}")
    return {
        "accuracy": float(accuracy_score(true_strings, predicted_strings)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_strings, predicted_strings)
        ),
        "macro_precision": float(
            precision_score(
                true_strings, predicted_strings, labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                true_strings, predicted_strings, labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                true_strings, predicted_strings, labels=labels, average="macro", zero_division=0
            )
        ),
    }


def _classification_table(true: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    labels = [f"{label:.2f}" for label in CLASS_LABELS]
    true_strings = true.astype(float).map(lambda value: f"{value:.2f}")
    predicted_strings = predicted.astype(float).map(lambda value: f"{value:.2f}")
    report = classification_report(
        true_strings,
        predicted_strings,
        labels=labels,
        target_names=[f"{label} g" for label in labels],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index(names="class_or_average")


def _save_confusion_matrix(trials: pd.DataFrame, output: Path) -> None:
    labels = [f"{label:.2f}" for label in CLASS_LABELS]
    true_strings = trials["mass_g"].map(lambda value: f"{float(value):.2f}")
    predicted_strings = trials["predicted_mass_g"].map(
        lambda value: f"{float(value):.2f}"
    )
    matrix = confusion_matrix(true_strings, predicted_strings, labels=labels)
    sns.set_theme(style="white", context="notebook")
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[f"{label:.2f} g" for label in CLASS_LABELS],
        yticklabels=[f"{label:.2f} g" for label in CLASS_LABELS],
        ax=axis,
    )
    axis.set_xlabel("Predicted imbalance")
    axis.set_ylabel("True imbalance")
    axis.set_title("Locked random-forest final-test results (10 trials)")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    output: Path,
    metrics: dict[str, object],
    trials: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    trial_metrics = metrics["trial_level"]
    window_metrics = metrics["window_level"]
    lines = [
        "# Final locked-test evaluation",
        "",
        "The development-selected random-forest pipeline was evaluated once on the",
        "10 prespecified, previously untouched trials (two trials per imbalance level).",
        "No feature selection, parameter tuning, or model fitting used these test trials.",
        "",
        "## Final results",
        "",
        f"- Trial accuracy: **{trial_metrics['accuracy']:.3f}** ({int(trials['correct'].sum())}/10 trials)",
        f"- Trial macro F1: **{trial_metrics['macro_f1']:.3f}**",
        f"- Trial balanced accuracy: **{trial_metrics['balanced_accuracy']:.3f}**",
        f"- Window accuracy: **{window_metrics['accuracy']:.3f}** ({int(metrics['correct_windows'])}/{int(metrics['window_count'])} windows)",
        f"- Window macro F1: **{window_metrics['macro_f1']:.3f}**",
        "",
        "Because the final set contains only two physical trials per class, each trial",
        "changes a class recall by 0.50. These results should be reported with the sample",
        "count and should not be treated as a precise population-performance estimate.",
        "",
        "## Trial predictions",
        "",
        "| Trial | True | Predicted | Correct |",
        "|---|---:|---:|:---:|",
    ]
    for row in trials.itertuples():
        lines.append(
            f"| {row.trial_id} | {row.mass_g:.2f} g | {row.predicted_mass_g:.2f} g | "
            f"{'yes' if row.correct else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Per-class trial results",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in classification.iloc[: len(CLASS_LABELS)].iterrows():
        lines.append(
            f"| {row['class_or_average']} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['f1-score']:.3f} | {int(row['support'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "This is the final held-out result for this fixed dataset split. It must not be",
            "used to retune the model and then reported again as an independent test result.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_evaluation(
    feature_path: Path,
    assignment_path: Path,
    model_path: Path,
    config_path: Path,
    results_dir: Path,
) -> dict[str, object]:
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    summary_path = tables_dir / "final_test_metrics.json"
    if summary_path.exists():
        raise RuntimeError(
            "Final-test metrics already exist. Refusing to repeat the one-time evaluation."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            "Locked development model is missing. Run `python -m src.train_models` first."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("final_test_evaluated") is not False:
        raise ValueError("Locked config is not in the expected pre-evaluation state.")
    model = joblib.load(model_path)
    selected = list(model.named_steps["selector"].selected_features_)
    if selected != list(config["selected_features"]):
        raise ValueError("Saved model feature selection does not match the locked config.")

    final_test, feature_columns, assignments = load_final_test_frame(
        feature_path, assignment_path
    )
    expected_columns = list(model.feature_names_in_)
    if feature_columns != expected_columns:
        raise ValueError("Feature columns do not match those used to fit the locked model.")

    predicted = model.predict(final_test.loc[:, feature_columns]).astype(float)
    windows = final_test.loc[:, ["trial_id", "window_id", "mass_g"]].copy()
    windows["predicted_mass_g"] = predicted
    windows["correct"] = windows["mass_g"].astype(float) == windows["predicted_mass_g"]
    trials = aggregate_trial_predictions(windows)
    if len(trials) != 10:
        raise RuntimeError("Final evaluation did not produce exactly 10 trial predictions.")

    window_metrics = metric_summary(windows["mass_g"], windows["predicted_mass_g"])
    trial_metrics = metric_summary(trials["mass_g"], trials["predicted_mass_g"])
    classification = _classification_table(
        trials["mass_g"], trials["predicted_mass_g"]
    )
    metrics: dict[str, object] = {
        "evaluation_completed_utc": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "development_trial_count": int((assignments["split"] == "development").sum()),
        "final_test_trial_count": int(len(trials)),
        "window_count": int(len(windows)),
        "correct_trials": int(trials["correct"].sum()),
        "correct_windows": int(windows["correct"].sum()),
        "trial_level": trial_metrics,
        "window_level": window_metrics,
        "audit": {
            "feature_table_sha256": _sha256(feature_path),
            "assignment_file_sha256": _sha256(assignment_path),
            "locked_config_sha256": _sha256(config_path),
            "model_file_sha256": _sha256(model_path),
            "retuning_performed": False,
        },
    }

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(tables_dir / "final_test_window_predictions.csv", index=False)
    trials.to_csv(tables_dir / "final_test_trial_predictions.csv", index=False)
    classification.to_csv(tables_dir / "final_test_classification_report.csv", index=False)
    summary_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _save_confusion_matrix(trials, figures_dir / "final_test_confusion_matrix.png")
    _write_report(results_dir / "final_evaluation_report.md", metrics, trials, classification)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("data/processed/window_features_2s.csv")
    )
    parser.add_argument(
        "--assignments", type=Path, default=Path("data/splits/trial_assignments.csv")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("results/models/development_candidate.joblib")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("results/tables/locked_model_config.json")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics = run_final_evaluation(
        args.features, args.assignments, args.model, args.config, args.results_dir
    )
    print("Completed the one-time evaluation on 10 locked final-test trials.")
    print(json.dumps(metrics["trial_level"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
