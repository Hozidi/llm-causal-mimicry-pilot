# Results (per model)

The notebook runs **one model per runtime** and writes a date-stamped folder per model. The canonical
output location is Google Drive:

```
MyDrive/causal_mimicry/<MODEL_KEY>/<MODEL_KEY>_<DATA_VERSION>_<RUN_DATE>_<artifact>
```

e.g. `mistral-small-24b-base/mistral-small-24b-base_v4_2026-06-13_stage5_direction_vs_order.png`.

## What to commit here

Mirror each model into `results/<MODEL_KEY>/`:

```
results/
├── mistral-small-24b-base/
│   ├── *_behavioural_direction.csv          # Stage 1
│   ├── *_dose_graded.csv / *_dose_ordinal.csv
│   ├── *_hidden_probes.csv                   # Stage 4
│   ├── *_direction_vs_order.csv              # Stage 5
│   ├── *_alignment.csv                       # Stage 6
│   ├── *_dose_geometry.csv                   # Stage 8
│   └── *.png                                 # the figures for that model
├── mistral-nemo-12b-base/
└── ...
```

**Commit the CSVs** — they are small and they *are* the results (every figure can be regenerated from
them). Commit a **curated** set of figures (the ones you'd actually show). Keep exhaustive figure dumps in
Drive or attach a zip to a **GitHub Release** rather than bloating git history.

`master_*.csv` (the Stage 13 cross-model accumulator) is the one file that spans models — keep it at
`results/master_cross_model.csv` and update it after each run.

## One-liner: sync a Drive folder into the repo

After a run, from a machine with the Drive folder synced locally:

```bash
bash scripts/sync_results_from_drive.sh "/path/to/MyDrive/causal_mimicry" mistral-small-24b-base
```

This copies the model's CSVs (always) and PNGs (optional) into `results/<MODEL_KEY>/`.
