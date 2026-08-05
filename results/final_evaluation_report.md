# Final locked-test evaluation

The development-selected random-forest pipeline was evaluated once on the
10 prespecified, previously untouched trials (two trials per imbalance level).
No feature selection, parameter tuning, or model fitting used these test trials.

## Final results

- Trial accuracy: **0.700** (7/10 trials)
- Trial macro F1: **0.707**
- Trial balanced accuracy: **0.700**
- Window accuracy: **0.620** (31/50 windows)
- Window macro F1: **0.626**

Because the final set contains only two physical trials per class, each trial
changes a class recall by 0.50. These results should be reported with the sample
count and should not be treated as a precise population-performance estimate.

## Trial predictions

| Trial | True | Predicted | Correct |
|---|---:|---:|:---:|
| mass_0.00g_trial_01 | 0.00 g | 0.50 g | no |
| mass_0.00g_trial_08 | 0.00 g | 0.00 g | yes |
| mass_0.25g_trial_04 | 0.25 g | 0.50 g | no |
| mass_0.25g_trial_05 | 0.25 g | 0.25 g | yes |
| mass_0.50g_trial_01 | 0.50 g | 0.75 g | no |
| mass_0.50g_trial_07 | 0.50 g | 0.50 g | yes |
| mass_0.75g_trial_01 | 0.75 g | 0.75 g | yes |
| mass_0.75g_trial_06 | 0.75 g | 0.75 g | yes |
| mass_1.00g_trial_07 | 1.00 g | 1.00 g | yes |
| mass_1.00g_trial_08 | 1.00 g | 1.00 g | yes |

## Per-class trial results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0.00 g | 1.000 | 0.500 | 0.667 | 2 |
| 0.25 g | 1.000 | 0.500 | 0.667 | 2 |
| 0.50 g | 0.333 | 0.500 | 0.400 | 2 |
| 0.75 g | 0.667 | 1.000 | 0.800 | 2 |
| 1.00 g | 1.000 | 1.000 | 1.000 | 2 |

## Interpretation rule

This is the final held-out result for this fixed dataset split. It must not be
used to retune the model and then reported again as an independent test result.
