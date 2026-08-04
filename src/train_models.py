"""Leakage-safe comparison of classical machine-learning models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate_models import evaluate_predictions, save_confusion_matrix


RANDOM_SEED = 42


def candidate_models() -> dict[str, object]:
    """Return models suited to a small engineered-feature dataset."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
            ]
        ),
        "knn": Pipeline(
            [("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=3))]
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED
        ),
        "svm": Pipeline(
            [("scale", StandardScaler()), ("model", SVC(kernel="rbf"))]
        ),
    }


def train_and_compare(feature_table: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Compare models with cross-validation and evaluate the best on a holdout."""
    if feature_table.empty:
        raise ValueError("Feature table is empty.")

    labels = feature_table["mass_g"]
    feature_columns = feature_table.select_dtypes(include="number").columns.difference(
        ["mass_g"]
    )
    features = feature_table.loc[:, feature_columns].replace(
        [float("inf"), float("-inf")], pd.NA
    )
    if features.isna().any().any():
        raise ValueError("Feature table contains missing or infinite numeric values.")

    folds = min(5, int(labels.value_counts().min()))
    if folds < 2:
        raise ValueError("At least two trials per class are required.")

    validation = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_SEED
    )
    rows: list[dict[str, float | str]] = []
    for name, model in candidate_models().items():
        scores = cross_val_score(
            model, features, labels, cv=validation, scoring="f1_macro"
        )
        rows.append(
            {
                "model": name,
                "cv_macro_f1_mean": float(scores.mean()),
                "cv_macro_f1_std": float(scores.std()),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        "cv_macro_f1_mean", ascending=False
    )
    best_name = str(comparison.iloc[0]["model"])
    best_model = candidate_models()[best_name]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    best_model.fit(x_train, y_train)
    predictions = best_model.predict(x_test)
    metrics, report = evaluate_predictions(y_test, predictions)
    for key, value in metrics.items():
        comparison.loc[comparison["model"] == best_name, f"holdout_{key}"] = value

    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    models_dir = results_dir / "models"
    for directory in (tables_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    report.to_csv(tables_dir / "best_model_classification_report.csv")
    save_confusion_matrix(
        y_test, predictions, figures_dir / "best_model_confusion_matrix.png"
    )
    joblib.dump(best_model, models_dir / "best_model.joblib")
    return comparison
