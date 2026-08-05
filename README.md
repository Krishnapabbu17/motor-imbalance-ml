# Motor Imbalance Detection with Machine Learning

This project studies MPU6050 vibration measurements from a small motor under
`0.00 g`, `0.25 g`, `0.50 g`, `0.75 g`, and `1.00 g` imbalance conditions.

The repository covers data organization, validation, feature engineering,
leakage-safe model comparison, and a locked final-test evaluation.

## Dataset and structure

The authoritative workbook contains cleaned, real experimental sensor data: 50
trials, with 10 trials at each imbalance level. Organization does not replace or
extend the measured acceleration readings.

```text
data/source/       Authoritative cleaned experimental workbook
data/cleaned/      One measurement-preserving CSV per trial and class
data/processed/    Trial manifest and generated feature tables
src/               Preparation, validation, and feature code
tests/             Automated calculation checks
results/tables/    Validation reports
```

Each trial CSV contains `time,ax,ay,az`. `time` is the original timestamp in
milliseconds; the acceleration columns retain the cleaned MPU6050 measurements.

## Feature tables

- `trial_features.csv`: one row per full experimental trial.
- `window_features_2s.csv`: five balanced, non-overlapping two-second windows per
  trial (250 total), recommended for future modeling.
- `feature_dictionary.csv`: meaning and intended role of every column.

Features cover vibration amplitude, distribution shape, cross-axis correlation,
dominant frequency, spectral centroid, spectral entropy, and relative spectral
power in 0–5 Hz, 5–15 Hz, and 15–30 Hz bands.

## Scientific safeguards

1. The authoritative cleaned workbook is preserved unchanged.
2. Future train/test splits must be grouped by `trial_id`.
3. `mass_g` is the future prediction target, not an input feature.
4. File names, sample counts, durations, and sampling rate are metadata.
5. Scaling and feature selection must be fitted only on training trials.

## Reproduce preparation

```powershell
python -m pip install -r requirements.txt
python -m src.prepare_data
python -m src.run_pipeline
python -m unittest discover -s tests -v
```

`src.run_pipeline` stops after feature generation and does not train a model.

## Exploratory analysis

Run the leakage-safe EDA after feature generation:

```powershell
python -m src.eda
```

This command locks two complete trials per class as the untouched final test set,
uses only development trials for feature screening, and saves:

- `data/splits/trial_assignments.csv`
- `results/eda_report.md`
- `results/figures/` signal, amplitude, frequency, ranking, and correlation plots
- `results/tables/feature_screening.csv`
- `results/tables/recommended_features.csv`
- `results/tables/development_class_summary.csv`

It does not train or evaluate a machine-learning model.

## Development model comparison

After the EDA split is locked, run:

```powershell
python -m src.train_models
```

This compares logistic regression, RBF SVM, random forest, and gradient boosting
with nested trial-grouped cross-validation. Feature selection and hyperparameter
tuning are repeated inside the training folds. Results are saved to:

- `results/model_selection_report.md`
- `results/tables/model_cv_summary.csv`
- `results/tables/model_cv_fold_metrics.csv`
- `results/tables/development_oof_trial_predictions.csv`
- `results/tables/development_oof_window_predictions.csv`
- `results/tables/locked_model_config.json`
- `results/figures/development_model_comparison.png`
- `results/figures/development_winner_trial_confusion_matrix.png`

The command fits a development candidate under `results/models/`, which remains
local and ignored by Git. It does not evaluate the 10 reserved final-test trials.

## One-time final evaluation

After the model family, parameters, and selected features are locked, run once:

```powershell
python -m src.evaluate_models
```

This loads the saved development model and predicts only the 10 prespecified
test trials. It performs no fitting, feature selection, or parameter tuning. It
saves the final metrics, per-trial predictions, classification report, and
confusion matrix under `results/`. The command refuses to run again when a final
metrics file already exists, protecting the held-out result from repeated use.

## Predict a new recording

Prepare an approximately 10-second CSV with these columns:

```text
time,ax,ay,az
```

Then run:

```powershell
python -m src.predict path\to\new_motor_trial.csv
```

Add `--output results\predictions\new_motor_trial.json` to save the full five
window votes and model scores. The predictor validates the recording, uses only
the first five non-overlapping two-second windows, applies the exact training
feature calculations, and never uses the file name as a model input.

## Local web interface

Install the project requirements, then start the interface from the repository
root:

```powershell
python -m pip install -r requirements.txt
python app.py
```

The app opens in the default browser. Choose an approximately 10-second CSV,
press **Predict imbalance**, and review the overall result and five window votes.
It binds only to `127.0.0.1` with public sharing disabled, so the interface is
available only on the computer running it.

On Windows, you can instead double-click `start_app.bat`. The first launch sets
up the local Python environment and installs the requirements; later launches
open the interface directly.
