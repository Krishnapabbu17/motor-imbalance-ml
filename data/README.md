# Experimental data

Store each original trial as a separate CSV under the matching imbalance folder in `data/raw/`.

Required columns:

```text
time,ax,ay,az
```

The raw files are the scientific record and must not be edited after collection. If a trial is invalid, preserve it and document the rejection reason instead of silently changing its measurements.

Files created by the analysis pipeline belong in `data/processed/`.
