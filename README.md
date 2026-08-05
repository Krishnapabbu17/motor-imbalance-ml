# Motor Imbalance Detection with Machine Learning

This project studies MPU6050 vibration measurements from a small motor under
`0.00 g`, `0.25 g`, `0.50 g`, `0.75 g`, and `1.00 g` imbalance conditions.

The repository currently covers data organization, validation, and feature
engineering. Model training is intentionally not run yet.

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
