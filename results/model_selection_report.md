# Development model comparison

Four classical classifiers were compared with nested, trial-grouped
cross-validation. No final-test trial was used for fitting, feature selection,
hyperparameter tuning, model selection, or reported metrics.

## Evaluation design

- Development trials: 40
- Reserved final-test trials: 10
- Outer evaluation: five-fold StratifiedGroupKFold by trial ID.
- Inner tuning: four-fold StratifiedGroupKFold by trial ID.
- Feature screening was refitted inside every training fold.
- Primary selection metric: trial-level macro F1.

## Cross-validation results

| Model | Trial macro F1 | Trial balanced accuracy | Window macro F1 |
|---|---:|---:|---:|
| random_forest | 0.619 +/- 0.157 | 0.700 +/- 0.126 | 0.575 +/- 0.122 |
| svm_rbf | 0.522 +/- 0.128 | 0.600 +/- 0.089 | 0.481 +/- 0.132 |
| logistic_regression | 0.494 +/- 0.181 | 0.560 +/- 0.196 | 0.509 +/- 0.167 |
| gradient_boosting | 0.472 +/- 0.073 | 0.580 +/- 0.075 | 0.508 +/- 0.106 |

## Locked development winner

The selected model family is **random_forest**.

Locked parameters: `{"model__max_depth": 8, "model__min_samples_leaf": 3, "selector__max_features": 12}`

This is a development result, not final project performance. The fitted
development candidate has been saved without evaluating the 10 reserved test
trials. The next stage is a single final evaluation using those trials.
