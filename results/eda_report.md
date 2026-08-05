# Exploratory data analysis and feature screening

No machine-learning model was trained during this stage.

## Dataset used

- 50 physical trials.
- 250 non-overlapping two-second windows.
- 5 balanced imbalance classes.
- No test-trial labels or values were used for feature ranking.

## Locked evaluation split

Two complete trials per class are reserved as an untouched final test set.
The remaining eight trials per class form the development set.
All future cross-validation must group rows by `trial_id`.

| Split | Trials per class | Total trials |
|---|---:|---:|
| Development | 8 | 40 |
| Test | 2 | 10 |

## Key development-data findings

- Mean dynamic magnitude rose from 0.955 m/s^2 at 0.00 g to 1.921 m/s^2 at 1.00 g.
- Y-axis standard deviation rose from 0.153 m/s^2 at 0.00 g to 0.482 m/s^2 at 1.00 g.
- Mean magnitude spectral centroid was 23.20 Hz for 0.00 g versus 16.31 Hz across the four faulted classes.
- Mean magnitude spectral entropy was 0.647 for 0.00 g versus 0.823 across the faulted classes.
- Screening retained 15 of 67 candidate features; 15 were removed as near-duplicates.

## Strongest development-only class effects

| Feature | Eta-squared | Recommended |
|---|---:|:---:|
| `magnitude_mean` | 0.814 | yes |
| `ay_std` | 0.781 | yes |
| `ay_rms_centered` | 0.781 | no |
| `ay_mean_abs_centered` | 0.773 | no |
| `az_mean_abs_centered` | 0.765 | no |
| `az_std` | 0.762 | no |
| `az_rms_centered` | 0.762 | no |
| `az_peak_abs_centered` | 0.693 | yes |

## Recommended exploratory feature set

- `magnitude_mean`
- `ay_std`
- `az_peak_abs_centered`
- `ax_std`
- `ay_median_abs_deviation`
- `magnitude_std`
- `magnitude_spectral_entropy`
- `magnitude_spectral_centroid_hz`
- `az_median_abs_deviation`
- `ax_peak_to_peak`
- `ay_peak_to_peak`
- `axis_correlation_xz`
- `axis_correlation_yz`
- `magnitude_peak_abs_centered`
- `az_relative_power_15_30_hz`

The shortlist was built only from development trials. Features were ranked by
descriptive class effect size, then measurements with absolute correlation of
0.95 or greater to an already selected feature were removed. The list is capped
at 15 features to reduce overfitting risk.

## Interpretation limits

Feature effect sizes are exploratory and are not model accuracy. Final feature
selection and scaling must occur inside grouped cross-validation. The reserved
test trials must be evaluated only after the model family and tuning plan are
locked.
