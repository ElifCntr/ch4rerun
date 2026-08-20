# ch4rerun

Chapter 4 rerun. Controlled comparison of 2D convolution, temporal shift and 3D convolution over identical motion proposals, on the static subset of Drone-vs-Bird. Binary task, drone against background.

This project reuses nothing from DroneDetection_2D_3D, which stays frozen as Stage 0 evidence.

## Status

Last updated 19 August 2026.

- **Stage 1a** (measurement, parameter-free, ground truth only, train and val): **CLOSED**. Items A, B first half, C, D, E, F, G1 and H measured, plus the epoch-timing check and the stride-2 confirmation pass.
- **Stage 1b-M** (measurement at declared operating points): **CLOSED**. The OpenCV smoke test, the ViBe implementation, the parameter work and the three-method comparison. The item F repeat was cut on 18 August, since the three-method pass supplies the negative-difficulty evidence it was meant to probe.
- **Stage 1b-E** (extraction at ruled parameters): **DONE**. Proposal coordinates extracted for all three partitions, negatives drawn for seeds 1, 2 and 3, and 89,727 training tubelets materialised.
- **Stage 2** (classifier matrix, four arms by three seeds): **HELD**, pending the training harness, which is not yet written.

Item scope per stage is in `docs/STAGE_SCOPE.md`, which is the repository's record rather than a chat's. Anything in this README that disagrees with a committed report is wrong, and the report wins.

## What is ruled and what is not

Every ruled parameter lives in the `ruled` block of `configs/ch4.yaml` and is read through `src/ruled.py`, which has **no defaults** and raises on a missing key. Until 19 August the extraction constants were hardcoded in `tools/extract_proposals.py` instead, and the config's header comment still described T and the padding fraction as absent by design when they had been ruled on 12 August. Both are fixed.

Ruled, with the measurement or rationale behind each:

| key | value | basis |
|---|---|---|
| `window.T` | 8 | item E, item H and the stride pass, 12 Aug |
| `window.stride` | 1 | stride 2 dominated on every axis |
| `window.padding_fraction` | 0.3 | item E found no benefit at 0.75 |
| `window.batch` | 162 | largest fitting `r3d_18` at T=8 on 16 GB |
| `bgs.method` | mog2 | beats ViBe and KNN in every scene and both resolutions |
| `bgs.history` | 6960 | non-binding on every clip, so the method is alpha = 1/(2·nframes), a growing window with no forgetting |
| `bgs.learning_rate` | null | never passed; passing any rate makes `history` inert |
| `bgs.morphology` | null | a 3×3 opening cost 43 points of recall |
| `bgs.warmup_frames` | 0 | every candidate lost once discarded frames were charged as misses |
| `proposals.min_area_frac` | 7.23e-06 | measured 1080p minimum annotated box area fraction |
| `proposals.max_area_px` | per resolution | measured maximum annotated box area |
| `proposals.positive_coverage` | 0.5 | positives are the scarce resource |
| `negatives.ratio` | 2 | preserves the measured budget; negatives are effectively unlimited |
| `training.*` | see config | SGD 0.9 / 1e-4, base LR 1e-3, ÷10 at 10 and 20, ceiling 30 epochs, patience 5 |
| `augmentation.*` | see config | flip p=0.5, brightness and contrast in [0.9, 1.1], one draw per tubelet |

Not ruled, and therefore absent from the config rather than defaulted: the results framing, and whether the negative draw should stratify by scene as well as by size.

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
python tools/measure_geometry.py  --config configs/ch4.yaml
python tools/measure_instances.py --config configs/ch4.yaml

# 5. Items E and F. Strip generation, then manual scoring, then analysis.
python tools/make_strips.py   --config configs/ch4.yaml
python tools/make_strips_f.py --config configs/ch4.yaml
python tools/make_scorer.py   --config configs/ch4.yaml
python tools/make_stride_sheet.py --config configs/ch4.yaml
python tools/analyse_strips.py    --config configs/ch4.yaml
python tools/analyse_strips_f.py  --config configs/ch4.yaml
python tools/analyse_stride2.py   --config configs/ch4.yaml

# Separate, and independent of the above.
python tools/epoch_timing.py --config configs/ch4.yaml
```

Steps 2 and 3 depend on step 1, because the parser and the file pairing are chosen from what step 1 reports.

### Stage 1b-M

```bash
# Background subtractor validation. Runs before anything that uses them.
python tools/validate_vibe.py
python tools/validate_mog2.py

# Item D stationarity statistic and the item B per-resolution split.
python tools/measure_stationarity.py
python tools/analyse_size_split.py

# One pass per method, then the offline analysis. Nothing is re-run to change
# a threshold; every threshold is post-hoc arithmetic on the dumps.
python tools/run_bgs_pass.py --method mog2 --history 6960 --data-root <path>
python tools/run_bgs_pass.py --method knn  --history 6960 --data-root <path>
python tools/run_bgs_pass.py --method vibe --seed 20260818 --data-root <path>
python tools/analyse_bgs_pass.py --dump reports/bgs_pass_mog2
python tools/compare_bgs_methods.py

# Cost measurements that set the operating point and the schedule.
python tools/analyse_warmup.py
python tools/size_storage.py    --data-root <path>
python tools/check_decode_4k.py --data-root <path>
python tools/inference_timing.py
```

`analyse_size_split.py` exits non-zero if any figure disagrees with the recorded item B values. That is intended. A disagreement means the inputs differ from what was measured, and it is reported rather than reconciled.

### Stage 1b-E

```bash
python tools/build_val_subset.py
python tools/extract_proposals.py --include-test
python tools/draw_negatives.py
python tools/materialise_tubelets.py --data-root <path>
```

`extract_proposals.py` refuses to start on a config that contradicts a ruling: a non-null learning rate, a non-null morphology, or an assignment rule it does not implement.

## Rules held in code

- Data path lives in `configs/ch4.yaml`. It never appears in a source file, and the dataset is never copied into this project.
- **All data access goes through `src/splits.py`, benchmarks included.** It refuses the test partition unless `allow_test=True` is passed explicitly, and every use is logged to `reports/test_access_log.txt`. This is stricter than it was: a decode benchmark once read the split CSV directly and touched test without the flag, which is why reading the CSV is no longer acceptable anywhere.
- **All ruled parameters go through `src/ruled.py`**, which has no defaults, raises on a missing key, and reports every missing key at once rather than the first.
- `src/seeding.py` seeds torch, numpy, random and DataLoader workers, and is called from every entry point.
- `src/vibe.py` seeds its own RNG and records the seed. ViBe is non-deterministic by design and the original Chapter 4 code failed exactly here.
- `build_split.py` asserts rather than checks. See the assertion list below.

## Environment

`environment.yml` pins the stack cloned from `sussexdrone`, so Chapters 4, 5 and 6 share one verified torch build. OpenCV is 5.0.0.93 and numpy is 2.4.6.

It is a **raw conda export** and carries the artefacts of one: a machine-specific `prefix` line, a mix of cu12 and cu13 wheels, and packages inherited from the clone that this project never uses, `ultralytics` among them. It is committed as a record of what the runs actually executed under, not as a minimal specification.

**No dataframe dependency was added for analysis.** The analysis tools use the standard library `csv` module plus numpy. `polars` is present in the environment, inherited from the clone rather than added here, and nothing in `tools/` imports it. The claim is that the verified torch stack was never perturbed to make an analysis script more convenient, not that no dataframe library exists.

## Committed artefacts

Configs, manifests, logs, results, CSVs and reports are committed. Large binaries and regenerable bulk outputs are ignored.

**Strip images are not committed** (ruled 11 August 2026). Committed instead are the sampling manifests under `data/strip_manifests/`, which carry video, tubelet identifier, frame indices, T, stride and seed, and the completed scoring sheets under `reports/scoring_sheets/`, which carry a manual judgement that nothing can regenerate. Any strip promoted to a chapter figure is committed individually under `figures/`.

**`data/proposals/proposals.csv` (715 MB) and `data/tubelets/` (25.2 GB) are not committed**, by the same reasoning. Both regenerate from committed scripts, committed data and the declared parameters, and MOG2 is deterministic with OpenCL pinned off, evidenced by the 18 August re-run reproducing the 17 August pass to the last digit. `extraction_meta.json`, `extraction_stats.csv`, `tubelets_meta.json` and the tubelet manifest carry everything needed to verify a regeneration. The negative draws themselves ARE committed, since they are small and a re-draw would not reproduce without them.

`data/dvb_camera_motion.csv` is a committed input, not an output. It defines the static subset as `camera == static`, and it is the only record of a single-rater manual review of all 77 videos carried out on 10 August 2026. The label is a judgement, not a measurement, and the method section must say so. Without this file the 37-video subset is not reproducible.

## What build_split.py asserts

Each of these stops the run with a non-zero exit and a diff. None is reconciled, and the split is never adjusted to fit.

- 77 videos labelled, 40 dynamic and 37 static, every name present in the v1.0 manifest and every manifest video labelled
- static subset 37 videos and 30,788 boxes
- **partition totals** train 24 / 18,624, val 6 / 5,259, test 7 / 6,905, counted twice, once from the manifest's boxes column and once from the label files themselves
- zero session overlap between any pair of partitions
- all five surviving scenes present in train

**The assertion is at partition level.** Per-video file-versus-manifest mismatches are reported as notes rather than failures, so in principle compensating per-video errors would pass the partition check. Promoting the per-video check to fatal is an open option.

## Known record corrections

`cv2_frame_count` **overcounts by two frames** on four `seaside_cuts` clips: `00_01_52_to_00_01_58`, `00_06_10_to_00_06_27`, `00_09_30_to_00_10_09` and `00_10_09_to_00_10_40`. Decode is the truth, and extraction probes each video rather than trusting the inventory. Any figure quoting 31,654 total frames is high by eight; the decoded total is 31,646. Box-count assertions are unaffected.
