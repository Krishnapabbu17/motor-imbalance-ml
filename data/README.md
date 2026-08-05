# Experimental data record

This project uses cleaned, real experimental acceleration measurements collected
from an MPU6050 attached to the motor setup. The measurements are not replacement
or artificially extended values.

## Folders

- `source/cleaned_experimental_motor_data.xlsx`: authoritative cleaned workbook.
- `cleaned/<mass>/`: one CSV for each experimental trial.
- `processed/trial_manifest.csv`: trial timing and sample-count record.
- `processed/trial_features.csv`: one feature row per complete trial.
- `processed/window_features_2s.csv`: five non-overlapping two-second windows per
  trial, preserving `trial_id` for leakage-safe future splitting.
- `processed/feature_dictionary.csv`: meaning and intended role of every column.

The per-trial conversion retains every cleaned timestamp and acceleration value.
It only separates trials and renames `timestamp_ms,x,y,z` to `time,ax,ay,az` for
the analysis code. The source workbook is never modified.
