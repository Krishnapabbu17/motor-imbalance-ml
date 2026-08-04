# Motor Imbalance Detection with Machine Learning

This project studies whether vibration measurements from a low-cost MPU6050 accelerometer can classify controlled imbalance in a small DC motor.

The planned experimental classes are `0.00 g`, `0.25 g`, `0.50 g`, `0.75 g`, and `1.00 g` of added mass at a fixed location on one propeller blade. Each trial is stored as a separate CSV file with these columns:

```text
time,ax,ay,az
```

- `time`: milliseconds
- `ax`, `ay`, `az`: acceleration in m/s^2

## Project structure

```text
data/raw/          Original trial CSV files, grouped by imbalance class
data/processed/    Generated feature tables
src/               Validation, feature extraction, modeling, and evaluation code
tests/             Automated checks for important calculations
results/figures/   Generated research figures
results/tables/    Generated quality and model tables
results/models/    Generated trained models
```

## Scientific rules

1. Raw trial files are never modified.
2. Each CSV represents one independent experimental trial.
3. Individual sensor rows from the same trial are never split between training and testing.
4. Preprocessing and feature scaling are fitted using training data only.
5. Generated and experimental data must never be mixed.

## Data layout

Place the final experimental CSV files here:

```text
data/raw/0.00g/
data/raw/0.25g/
data/raw/0.50g/
data/raw/0.75g/
data/raw/1.00g/
```

Use descriptive names such as `mass_0.25g_trial_03.csv`.

## Setup

Create and activate a Python virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the pipeline

```powershell
python -m src.run_pipeline
```

The pipeline validates every trial, creates a trial-level feature table, compares several classical machine-learning models, and saves results under `results/`.

## Run the tests

```powershell
python -m unittest discover -s tests -v
```

The repository currently contains the software foundation only. Final model results should not be reported until the replacement experimental trials have been collected and validated.
