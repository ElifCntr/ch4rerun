# ch4rerun

Chapter 4 rerun. Controlled comparison of 2D convolution, temporal shift and 3D convolution over identical motion proposals, on the static subset of Drone-vs-Bird. Binary task, drone against background.

This project reuses nothing from DroneDetection_2D_3D, which stays frozen as Stage 0 evidence.

## Status

Last updated 17 August 2026.

- **Stage 1a** (measurement, parameter-free, ground truth only, train and val): **CLOSED**. Items A, B first half, C, D, E, F, G1 and H are measured, plus the epoch-timing check and the stride-2 confirmation pass. Every ruling made during it rests on a committed measurement rather than an inherited value.
- **Stage 1b-M** (measurement at declared provisional operating points): **IN PROGRESS**, step 3 of 4. Steps 1 and 2 are done, the OpenCV 5.0.0 smoke test and the ViBe implementation. Step 3 is the parameter work, step 4 is the item F repeat against real unmatched proposals.
- **Stage 1b-E** (extraction at ruled parameters): **HELD**, pending the step 3 operating point.
- **Stage 2** (classifier matrix, four arms by three seeds): **HELD**.

Item scope per stage is in `docs/STAGE_SCOPE.md`, which is the repository's record rather than a chat's. Anything in this README that disagrees with a committed report is wrong, and the report wins.

## What is ruled and what is not

Ruled and carried in `configs/ch4.yaml`:

- `T = 8`, stride 1, padding fraction `p = 0.3`. Decided 12 August on item E, item H and the stride pass, not inherited.
- Crop geometry: square-pad in the original frame using real image context, then resize to 112x112. Aspect ratio preserved.
- Batch 162, set by the largest that fits `r3d_18` at T=8 on 16 GB and then applied to every arm.

Still unruled, and therefore absent from the config rather than defaulted:

- `min_area`, `max_area`, morphology kernel, MOG2 `history`, coverage thresholds, negative capping ratio, epoch count.

A parameter that is unruled has no entry at all. Nothing in this repository supplies a silent default for one.

## Order of operations

Nothing about the dataset layout is assumed. Run these in order and stop at the first failure.

```bash
conda create -n ch4rerun --clone sussexdrone
conda activate ch4rerun
conda env export > environment.yml
git add environment.yml && git commit -m "Pin environment cloned from sussexdrone"
```

### Stage 1a

```bash
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

# 4. Items B, C, D, G1 and H. Ground truth only, train and val.
python tools/measure_geometry.py --config configs/ch4.yaml

# 5. Items E and F. Strip generation, then manual scoring, then analysis.
python tools/make_strips.py   --config configs/ch4.yaml
python tools/make_strips_f.py --config configs/ch4.yaml
python tools/make_scorer.py   --config configs/ch4.yaml
python tools/make_stride_sheet.py --config configs/ch4.yaml
python tools/analyse_strips.py    --config configs/ch4.yaml
python tools/analyse_strips_f.py  --config configs/ch4.yaml
python tools/analyse_stride2.py   --config configs/ch4.yaml

# Separate, and independent of the above. Owed to the manager.
python tools/epoch_timing.py --config configs/ch4.yaml
```

Steps 2 and 3 depend on step 1, because the parser and the file pairing are chosen from what step 1 reports.

### Stage 1b-M

```bash
# Background subtractor validation. Runs before anything that uses them.
python tools/validate_vibe.py
python tools/check_mog2_learning_rate.py

# Item D stationarity statistic and the item B per-resolution split.
python tools/measure_stationarity.py
python tools/analyse_size_split.py
```

`analyse_size_split.py` exits non-zero if any figure disagrees with the recorded item B values. That is intended. A disagreement means the inputs differ from what was measured, and it is reported rather than reconciled.

## Rules held in code

- Data path lives in `configs/ch4.yaml`. It never appears in a source file, and the dataset is never copied into this project.
- `src/splits.py` refuses to return the test partition unless `allow_test=True` is passed explicitly. Stage 1a never passes it, and every use is logged to `reports/test_access_log.txt`.
- `src/seeding.py` seeds torch, numpy, random and DataLoader workers, and is called from every entry point.
- `src/vibe.py` seeds its own RNG and records the seed. ViBe is non-deterministic by design and the original Chapter 4 code failed exactly here.
- `build_split.py` asserts rather than checks. See the assertion list below.
- No unruled method parameter is set anywhere in this repository, and none is defaulted silently.

## Environment

`environment.yml` pins the stack cloned from `sussexdrone`, so Chapters 4, 5 and 6 share one verified torch build. OpenCV is 5.0.0.93 and numpy is 2.4.6.

**pandas is deliberately not installed.** Analysis tools use the standard library `csv` module plus numpy, so that no analysis convenience perturbs a verified torch stack. Anything added here needs a reason.

## Committed artefacts

Configs, manifests, logs, results, CSVs and reports are committed. Large binaries are ignored.

Strip images are not committed (ruled 11 August 2026). Committed instead are the sampling manifests under `data/strip_manifests/`, which carry video, tubelet identifier, frame indices, T, stride and seed, and the completed scoring sheets under `reports/scoring_sheets/`, which carry a manual judgement that nothing can regenerate. Any strip promoted to a chapter figure is committed individually under `figures/`, so that no thesis figure depends on a regeneration step years later.

`data/dvb_camera_motion.csv` is a committed input, not an output. It defines the static subset as `camera == static`, and it is the only record of a single-rater manual review of all 77 videos carried out on 10 August 2026. The label is a judgement, not a measurement, and the method section must say so. Without this file the 37-video subset is not reproducible.

## What build_split.py asserts

Each of these stops the run with a non-zero exit and a diff. None is reconciled, and the split is never adjusted to fit.

- 77 videos labelled, 40 dynamic and 37 static, every name present in the v1.0 manifest and every manifest video labelled
- static subset 37 videos and 30,788 boxes
- train 24 / 18,624, val 6 / 5,259, test 7 / 6,905, counted twice, once from the manifest's boxes column and once from the label files themselves
- zero session overlap between any pair of partitions
- all five surviving scenes present in train
