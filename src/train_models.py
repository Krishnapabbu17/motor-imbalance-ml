"""Nested, trial-grouped model comparison using development trials only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_SEED = 42
CLASS_LABELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
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


def _eta_squared(values: pd.Series, labels: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & labels.notna()
    numeric = numeric.loc[valid]
    labels = labels.loc[valid]
    if numeric.empty:
        return 0.0
    grand_mean = float(numeric.mean())
    total = float(np.sum((numeric - grand_mean) ** 2))
    if total == 0.0:
        return 0.0
    between = sum(
        len(group) * (float(group.mean()) - grand_mean) ** 2
        for _, group in numeric.groupby(labels)
    )
    return float(between / total)


class EffectSizeCorrelationSelector(BaseEstimator, TransformerMixin):
    """Select strong, nonredundant features using training-fold data only."""

    def __init__(self, max_features: int = 15, correlation_threshold: float = 0.95):
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold

    def fit(self, features: pd.DataFrame, labels: pd.Series):
        frame = pd.DataFrame(features).copy()
        target = pd.Series(np.asarray(labels), index=frame.index)
        correlation = frame.corr().abs().fillna(0.0)
        ranked = sorted(
            frame.columns,
            key=lambda column: _eta_squared(frame[column], target),
            reverse=True,
        )
        selected: list[str] = []
        for column in ranked:
            if frame[column].nunique(dropna=True) <= 1:
                continue
            if any(
                float(correlation.loc[column, kept]) >= self.correlation_threshold
                for kept in selected
            ):
                continue
            selected.append(str(column))
            if len(selected) >= self.max_features:
                break
        if not selected:
            raise ValueError("Feature selection retained no columns.")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.selected_features_ = selected
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(features, columns=getattr(self, "feature_names_in_", None))
        return frame.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.selected_features_, dtype=object)


def candidate_models() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    selector = EffectSizeCorrelationSelector()
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("selector", selector),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
                    ),
                ]
            ),
            {
                "selector__max_features": [8, 12, 15],
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
        "svm_rbf": (
            Pipeline(
                [
                    ("selector", EffectSizeCorrelationSelector()),
                    ("scale", StandardScaler()),
                    ("model", SVC(kernel="rbf")),
                ]
            ),
            {
                "selector__max_features": [8, 12, 15],
                "model__C": [0.5, 2.0, 8.0],
            },
        ),
        "random_forest": (
            Pipeline(
                [
                    ("selector", EffectSizeCorrelationSelector()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=150,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "selector__max_features": [8, 12, 15],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "gradient_boosting": (
            Pipeline(
                [
                    ("selector", EffectSizeCorrelationSelector()),
                    ("model", GradientBoostingClassifier(random_state=RANDOM_SEED)),
                ]
            ),
            {
                "selector__max_features": [8, 12, 15],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2],
            },
        ),
    }


def majority_trial_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate five window predictions into one prediction per physical trial."""
    rows = []
    for (model, trial_id), group in predictions.groupby(["model", "trial_id"]):
        counts = Counter(group["predicted_mass_g"].astype(float))
        predicted = sorted(counts, key=lambda value: (-counts[value], value))[0]
        rows.append(
            {
                "model": model,
                "trial_id": trial_id,
                "mass_g": float(group["mass_g"].iloc[0]),
                "predicted_mass_g": float(predicted),
                "outer_fold": int(group["outer_fold"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _metric_record(
    true: np.ndarray, predicted: np.ndarray, prefix: str
) -> dict[str, float]:
    true_labels = np.asarray([f"{float(value):.2f}" for value in true])
    predicted_labels = np.asarray([f"{float(value):.2f}" for value in predicted])
    class_strings = [f"{label:.2f}" for label in CLASS_LABELS]
    recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=class_strings,
        average=None,
        zero_division=0,
    )
    record = {
        f"{prefix}_macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro")
        ),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        f"{prefix}_accuracy": float(accuracy_score(true_labels, predicted_labels)),
    }
    for label, recall in zip(CLASS_LABELS, recalls, strict=True):
        record[f"{prefix}_recall_{label:.2f}g"] = float(recall)
    return record


def _development_frame(
    feature_path: Path, assignment_path: Path
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    features = pd.read_csv(feature_path)
    assignments = pd.read_csv(assignment_path)
    if assignments["trial_id"].duplicated().any():
        raise ValueError("Trial split file contains duplicate trial IDs.")
    merged = features.merge(
        assignments.loc[:, ["trial_id", "mass_g", "split"]],
        on="trial_id",
        suffixes=("", "_assignment"),
        validate="many_to_one",
    )
    if not np.allclose(merged["mass_g"], merged["mass_g_assignment"]):
        raise ValueError("Class labels disagree between features and split assignments.")
    test_trials = assignments.loc[assignments["split"] == "test", "trial_id"]
    if len(test_trials) != 10:
        raise ValueError("Exactly 10 physical trials must remain reserved for final testing.")
    development = merged.loc[merged["split"] == "development"].copy()
    feature_columns = [
        column
        for column in features.select_dtypes(include="number").columns
        if column not in METADATA_COLUMNS
    ]
    if development[feature_columns].isna().any().any():
        raise ValueError("Development feature table contains missing values.")
    return development, feature_columns, assignments


def _save_comparison_plot(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("trial_macro_f1_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(
        ordered["model"],
        ordered["trial_macro_f1_mean"],
        xerr=ordered["trial_macro_f1_std"],
        color="#2E75B6",
        alpha=0.9,
        capsize=4,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Trial-level macro F1 (outer grouped CV mean ± SD)")
    axis.set_title("Leakage-safe development model comparison")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_confusion_plot(trials: pd.DataFrame, model_name: str, output: Path) -> None:
    chosen = trials.loc[trials["model"] == model_name]
    true_labels = chosen["mass_g"].map(lambda value: f"{float(value):.2f}")
    predicted_labels = chosen["predicted_mass_g"].map(
        lambda value: f"{float(value):.2f}"
    )
    class_strings = [f"{label:.2f}" for label in CLASS_LABELS]
    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=class_strings
    )
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
    axis.set_title(f"Development out-of-fold trial predictions: {model_name}")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    output: Path,
    summary: pd.DataFrame,
    winner: str,
    locked_config: dict[str, object],
    assignments: pd.DataFrame,
) -> None:
    lines = [
        "# Development model comparison",
        "",
        "Four classical classifiers were compared with nested, trial-grouped",
        "cross-validation. No final-test trial was used for fitting, feature selection,",
        "hyperparameter tuning, model selection, or reported metrics.",
        "",
        "## Evaluation design",
        "",
        f"- Development trials: {int((assignments['split'] == 'development').sum())}",
        f"- Reserved final-test trials: {int((assignments['split'] == 'test').sum())}",
        "- Outer evaluation: five-fold StratifiedGroupKFold by trial ID.",
        "- Inner tuning: four-fold StratifiedGroupKFold by trial ID.",
        "- Feature screening was refitted inside every training fold.",
        "- Primary selection metric: trial-level macro F1.",
        "",
        "## Cross-validation results",
        "",
        "| Model | Trial macro F1 | Trial balanced accuracy | Window macro F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.sort_values("trial_macro_f1_mean", ascending=False).itertuples():
        lines.append(
            f"| {row.model} | {row.trial_macro_f1_mean:.3f} +/- {row.trial_macro_f1_std:.3f} "
            f"| {row.trial_balanced_accuracy_mean:.3f} +/- {row.trial_balanced_accuracy_std:.3f} "
            f"| {row.window_macro_f1_mean:.3f} +/- {row.window_macro_f1_std:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Locked development winner",
            "",
            f"The selected model family is **{winner}**.",
            "",
            f"Locked parameters: `{json.dumps(locked_config['best_params'], sort_keys=True)}`",
            "",
            "This is a development result, not final project performance. The fitted",
            "development candidate has been saved without evaluating the 10 reserved test",
            "trials. The next stage is a single final evaluation using those trials.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model_comparison(
    feature_path: Path,
    assignment_path: Path,
    results_dir: Path,
) -> pd.DataFrame:
    development, feature_columns, assignments = _development_frame(
        feature_path, assignment_path
    )
    features = development.loc[:, feature_columns]
    labels = development["mass_g"].map(lambda value: f"{float(value):.2f}")
    groups = development["trial_id"].astype(str)
    models = candidate_models()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    cv_assignment_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for outer_fold, (train_index, validation_index) in enumerate(
        outer.split(features, labels, groups), start=1
    ):
        train_groups = set(groups.iloc[train_index])
        validation_groups = set(groups.iloc[validation_index])
        if train_groups & validation_groups:
            raise RuntimeError("A physical trial crossed an outer fold boundary.")
        for trial_id in sorted(validation_groups):
            cv_assignment_rows.append({"trial_id": trial_id, "outer_fold": outer_fold})

        inner = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=RANDOM_SEED + outer_fold
        )
        x_train = features.iloc[train_index]
        y_train = labels.iloc[train_index]
        g_train = groups.iloc[train_index]
        x_validation = features.iloc[validation_index]
        y_validation = labels.iloc[validation_index]

        for model_name, (pipeline, grid) in models.items():
            search = GridSearchCV(
                pipeline,
                grid,
                scoring="f1_macro",
                cv=inner,
                refit=True,
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(x_train, y_train, groups=g_train)
            predicted = search.predict(x_validation).astype(float)
            selected = search.best_estimator_.named_steps["selector"].selected_features_
            for feature in selected:
                selected_rows.append(
                    {
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "feature": feature,
                    }
                )
            fold_prediction_rows = []
            for position, predicted_mass in zip(validation_index, predicted, strict=True):
                record = {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "trial_id": str(development.iloc[position]["trial_id"]),
                    "window_id": int(development.iloc[position]["window_id"]),
                    "mass_g": float(development.iloc[position]["mass_g"]),
                    "predicted_mass_g": float(predicted_mass),
                }
                prediction_rows.append(record)
                fold_prediction_rows.append(record)
            fold_predictions = pd.DataFrame(fold_prediction_rows)
            trial_predictions = majority_trial_predictions(fold_predictions)
            fold_rows.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "inner_best_macro_f1": float(search.best_score_),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    "selected_feature_count": len(selected),
                    **_metric_record(
                        y_validation.astype(float).to_numpy(), predicted, prefix="window"
                    ),
                    **_metric_record(
                        trial_predictions["mass_g"].to_numpy(),
                        trial_predictions["predicted_mass_g"].to_numpy(),
                        prefix="trial",
                    ),
                }
            )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    trial_predictions = majority_trial_predictions(predictions)
    metric_columns = [
        column
        for column in folds.columns
        if column.startswith("window_") or column.startswith("trial_")
    ]
    summary_rows = []
    for model_name, group in folds.groupby("model", sort=True):
        record: dict[str, object] = {"model": model_name}
        for column in metric_columns:
            record[f"{column}_mean"] = float(group[column].mean())
            record[f"{column}_std"] = float(group[column].std(ddof=0))
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["trial_macro_f1_mean", "window_macro_f1_mean", "trial_macro_f1_std"],
        ascending=[False, False, True],
    )
    winner = str(summary.iloc[0]["model"])

    final_pipeline, final_grid = models[winner]
    final_search = GridSearchCV(
        final_pipeline,
        final_grid,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_SEED
        ),
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    final_search.fit(features, labels, groups=groups)
    locked_config = {
        "model": winner,
        "best_params": final_search.best_params_,
        "development_inner_macro_f1": float(final_search.best_score_),
        "selected_features": final_search.best_estimator_.named_steps[
            "selector"
        ].selected_features_,
        "development_trial_count": int(groups.nunique()),
        "reserved_test_trial_count": int((assignments["split"] == "test").sum()),
        "final_test_evaluated": False,
        "random_seed": RANDOM_SEED,
    }

    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    models_dir = results_dir / "models"
    for directory in (tables_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    folds.to_csv(tables_dir / "model_cv_fold_metrics.csv", index=False)
    summary.to_csv(tables_dir / "model_cv_summary.csv", index=False)
    predictions.to_csv(tables_dir / "development_oof_window_predictions.csv", index=False)
    trial_predictions.to_csv(
        tables_dir / "development_oof_trial_predictions.csv", index=False
    )
    pd.DataFrame(cv_assignment_rows).drop_duplicates().sort_values("trial_id").to_csv(
        tables_dir / "development_cv_assignments.csv", index=False
    )
    selected_frame = pd.DataFrame(selected_rows)
    selected_frequency = (
        selected_frame.groupby(["model", "feature"]).size().rename("folds_selected").reset_index()
    )
    selected_frequency.to_csv(
        tables_dir / "model_feature_selection_frequency.csv", index=False
    )
    (tables_dir / "locked_model_config.json").write_text(
        json.dumps(locked_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    joblib.dump(final_search.best_estimator_, models_dir / "development_candidate.joblib")

    sns.set_theme(style="whitegrid", context="notebook")
    _save_comparison_plot(summary, figures_dir / "development_model_comparison.png")
    _save_confusion_plot(
        trial_predictions,
        winner,
        figures_dir / "development_winner_trial_confusion_matrix.png",
    )
    _write_report(
        results_dir / "model_selection_report.md",
        summary,
        winner,
        locked_config,
        assignments,
    )
    print(f"Compared {len(models)} models using nested trial-grouped CV.")
    print(f"Locked development winner: {winner}.")
    print("The 10 reserved final-test trials were not evaluated.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("data/processed/window_features_2s.csv")
    )
    parser.add_argument(
        "--assignments", type=Path, default=Path("data/splits/trial_assignments.csv")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    summary = run_model_comparison(args.features, args.assignments, args.results_dir)
    print(summary.loc[:, ["model", "trial_macro_f1_mean", "window_macro_f1_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
