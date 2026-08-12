# ch4rerun

Chapter 4 rerun. Controlled comparison of 2D convolution, temporal shift and 3D
convolution over identical motion proposals, on the static subset of
Drone-vs-Bird. Binary task, drone against background.

This project reuses nothing from `DroneDetection_2D_3D`, which stays frozen as
Stage 0 evidence.

## Status

- Stage 1a (measurement, parameter-free, ground truth only, train and val):
  scaffold in place, layout verification not yet run.
- Stage 1b-M (measurement at a declared provisional operating point):
  authorised, runs after Stage 1a reports.
- Stage 1b-E (extraction at ruled parameters): HELD.
- Stage 2 (classifier matrix): HELD.

Item scope per stage is in `docs/STAGE_SCOPE.md`, which is the repository's
record rather than a chat's.

## Order of operations

Nothing about the dataset layout is assumed. Run these in order and stop at the
first failure.

```bash
conda create -n ch4rerun --clone sussexdrone
conda activate ch4rerun
conda env export > environment.yml
git add environment.yml && git commit -m "Pin environment cloned from sussexdrone"

# 1. Layout report. Answers: what is actually at the data path.
python tools/inventory.py --config configs/ch4.yaml --stage layout

# 2. Per-video inventory. Frame rate read from the file, not from any manifest.
python tools/inventory.py --config configs/ch4.yaml --stage videos

# 2b. gopro_000-008 continuity. Reads container metadata for gopro_004, which
#     is in test, so the flag is required and the access is logged.
python tools/inventory.py --config configs/ch4.yaml --stage continuity \
    --inspect-test-metadata

# 3. Split integrity. Writes dvb_splits_v2.0-static.csv only if every
#    assertion holds. Exits non-zero with a diff on any mismatch.
python tools/build_split.py --config configs/ch4.yaml

# Separate, and independent of the above. Owed to the manager.
python tools/epoch_timing.py --config configs/ch4.yaml
```

Steps 2 and 3 depend on step 1, because the parser and the file pairing are
chosen from what step 1 reports. The measurement scripts for Stage 1a items B
to H are not written yet, for the same reason.

## Rules held in code

- Data path lives in `configs/ch4.yaml`. It never appears in a source file, and
  the dataset is never copied into this project.
- `src/splits.py` refuses to return the test partition unless
  `allow_test=True` is passed explicitly. Stage 1a never passes it.
- `src/seeding.py` seeds torch, numpy, random and DataLoader workers, and is
  called from every entry point.
- No method parameter is set anywhere in this repo. `configs/ch4.yaml` carries
  no value for min_area, max_area, T, coverage thresholds, morphology or
  padding. Those come from Stage 1a measurements and a later ruling.

## Committed artefacts

Configs, manifests, logs, results, CSVs and reports are committed. Large
binaries are ignored.

Strip images are not committed (ruled 11 August 2026). Committed instead are
the sampling manifests under `data/strip_manifests/`, which carry video,
tubelet identifier, frame indices, T, stride and seed, and the completed
scoring sheets under `reports/scoring_sheets/`, which carry a manual judgement
that nothing can regenerate. Any strip promoted to a chapter figure is
committed individually under `figures/`, so that no thesis figure depends on a
regeneration step years later.

`data/dvb_camera_motion.csv` is a committed input, not an output. It defines
the static subset as `camera == static`, and it is the only record of a
single-rater manual review of all 77 videos carried out on 10 August 2026. The
label is a judgement, not a measurement, and the method section must say so.
Without this file the 37-video subset is not reproducible.

## What build_split.py asserts

Each of these stops the run with a non-zero exit and a diff. None is
reconciled, and the split is never adjusted to fit.

- 77 videos labelled, 40 dynamic and 37 static, every name present in the v1.0
  manifest and every manifest video labelled
- static subset 37 videos and 30,788 boxes
- train 24 / 18,624, val 6 / 5,259, test 7 / 6,905, counted twice, once from
  the manifest's boxes column and once from the label files themselves
- zero session overlap between any pair of partitions
- all five surviving scenes present in train
