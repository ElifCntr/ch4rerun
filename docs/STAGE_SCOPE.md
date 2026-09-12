# Stage boundaries

Ruled 11 August 2026, revised 17, 19 and 22 August 2026. This file is the
repository's record of what may run at each stage and under what constraint. It
exists because the boundary moved once already, and a boundary that lives only
in a chat is not a boundary.

Revision note. The 11 August version of this file recorded the item F boundary
correctly and the boundary was still wrong, for reasons given under item F
below. A boundary being written down does not make it right, only auditable.

## Stage 1a, measurement

CLOSED 12 August 2026. Strictly parameter-free and ground-truth-only. No
background subtractor ran. No number produced here came from a chosen
parameter value.

| Item | What it measures |
| --- | --- |
| A | Per-video inventory. Frame rate read from the file, not from a manifest |
| B | Size distribution of annotated instances only (first half) |
| C | Track length, consecutive annotated frames per instance |
| D | Per-frame centre drift, in pixels and as a fraction of box size |
| E | Crop-retention strips, T in {3, 5, 8, 16}, stride 1, two padding conditions |
| F | Motion separability, drone tubelets against random background crops |
| G1 | Kernel-versus-target-size geometry, arithmetic on B's measured sizes |
| H | Padding sweep against drift |

Train and val only.

**Item E, amended 12 August.** Stride 2 was dropped from the first pass and its
budget spent on a second padding condition, because stride changes span, which
is already understood arithmetically, whereas padding changes what the rater
actually sees. Stride 2 later ran as a narrowed confirmation pass at T=8 only,
30 strips. Both passes are complete.

**Item F, amended 12 August, and this correction matters.** The 11 August
version of this file claimed item F "still answers whether drone motion is
separable from clutter". It does not. Item F's negatives are random crops from
drone-free frames, whereas Stage 2's negatives are unmatched motion proposals,
which by construction moved enough to trigger the subtractor. Random sky is
trivially not a drone; motion-selected clutter is not. Appearance alone scored
23 of 23, leaving no headroom for motion to demonstrate anything, so item F as
run cannot answer the question in either direction.

The cause is the 11 August rescope recorded in this file. Making item F
runnable inside a parameter-free Stage 1a changed what it measures. The
original run is kept and reported as a floor, and as validation that crops at
T=8 and p=0.3 are legible enough to support the task at all. It is never
reported as evidence about Claim A's premise.

**The item F repeat was CUT on 18 August**, reversing the 12 August ruling. The
bound set then already established that cue counts cannot confirm or falsify
Claim A in either direction, and the three-method pass supplies the
negative-difficulty evidence the repeat was meant to probe.

**Item G1** answers whether a kernel of a given size erases a target of a given
size. That is geometry, not a sweep. Its honest claim is that a kernel of 7 or
above costs about an eighth of the targets. **It does not establish that k=3 is
safe**, because it measures ground-truth boxes rather than real masks, and a
real mask is smaller and holier than its bounding box. That caution was
vindicated on 18 August: a 3x3 opening cost 43 points of proposal recall and
erased the drone entirely for one box in seven.

**Item D, corrected 21 August.** The ceiling comparison was degenerate for
stationary runs beginning before frame 3, which can never be a T=8 window
centre and whose computed ceiling falls below one frame. Excluding them gives
**30 of 44 tracks fitting and 14 not, 31.8 per cent**, against the original 30
of 46 and 16, 34.8 per cent. Only two tracks were affected, so the finding
survives essentially intact. Both figures are reported side by side in
`reports/item_d_ceiling_corrected.txt`, and the write-up quotes the corrected
one.

## Stage 1b-M, measurement with declared operating points

CLOSED 18 August 2026. Every output is labelled provisional and no number from
it is quoted as a chapter result. Its purpose was to produce the inputs from
which the real parameters were ruled, and it did.

What it carried: item B second half, the three-method proposal comparison, the
coverage curves, and the cost measurements that set the operating point.

Operating point, all of it declared rather than chosen:

- `learningRate` **left at the OpenCV default and never passed explicitly**.
  Measured 15 August by `tools/validate_mog2.py`: `apply()` uses any
  non-negative rate you pass and only derives alpha from `history` otherwise,
  so passing a rate makes `history` inert. The original Chapter 4 config passed
  0.001 and its `history = 200` therefore had no effect on a single frame.
- `history` **not derived from an absorption target**. That derivation was
  abandoned 17 August as unsound on this data, for the reason recorded below.
- `varThreshold` at the OpenCV default, reported as a default.
- `complexityReductionThreshold` at OpenCV's default of 0.05, with the
  divergence from Zivkovic 2004's suggested cT = 0.01 **named explicitly**
  rather than passed over as "OpenCV defaults".
- `detectShadows = false`. OpenCL **pinned off**, not merely recorded.
- **No morphology.** Ruled out 18 August on measurement.

Why the absorption-target derivation was dropped. MOG2's learning rate ramps as
`1/min(2*nframes, history)`, so alpha is set by elapsed frames rather than by
`history` until half a history has passed. Absorption is therefore capped at a
fraction of the time elapsed when a target stops moving, whatever `history` is
set to. A single absorption target in seconds is not uniformly achievable, so
the derivation is replaced and the analysis is written up as a methodological
finding rather than a gap.

**The three-method result.** MOG2, ViBe and KNN on identical frames. At
coverage 0.5, recall is 0.7455, 0.3654 and 0.5770, and the union of all three
is 0.7572. The union beats the best single method by 1.17 points and ViBe finds
nothing the others miss, so the roughly 24 per cent missed by all three is a
property of motion proposals at this target scale rather than of MOG2. MOG2 is
the ruled subtractor: it leads in every scene, both resolutions, and at matched
and at lower cost.

## Stage 1b-E, extraction

DONE. Re-run 19 August after the upper area limit was re-ruled; **the figures
below supersede every earlier recall number in this repository.**

Proposal recall: **train 0.7209, val 0.8363, test 0.8899**. Positives 13,347,
4,384 and 6,120. **76,682 tubelets materialised, 21.5 GB, zero dropped.**

**The upper limit was re-ruled on 19 August, and the earlier anchor was wrong.**
`max_area` was set at the largest annotated BOX per resolution. A covering blob
that merges a drone with adjacent motion exceeds any box, and 213 of 23,883
boxes had their best cover rejected on size alone, every one covering its drone
at 0.7 or better, concentrated at 4K (3.86 per cent) and in industrial (3.40).

It is now a **single frame-area fraction, 3.1587e-02**, the geometric midpoint
of two measured anchors: the largest observed best-covering blob at coverage
0.5 or better (1.6629e-02) and the fraction at which blobs fall to one per
video (6.0e-02), which is the frame-0 initialisation blob and the only thing
the limit demonstrably needs to reject. Finding those anchors required
excluding frames below 3, since on frame 0 everything is foreground and the two
anchors collapse into the same object.

Note the single value is **tighter at 1080p** (3.16e-02 against 4.10e-02) and
looser at 4K. No 1080p cover is lost, since the largest is 1.6629e-02.

**Negatives are stratified by size band AND scene**, ruled 19 August. Band-only
stratification put 56 per cent of negatives in two videos; it is now 33.8 per
cent across 24. Shortfall 2,538 of 26,694, recorded and not backfilled, giving
an **effective ratio of 1.81** rather than 2. It concentrates in the seaside
and meadow top bands, where the scenes with the largest drones have the fewest
large false positives, which is a property of the data rather than of the draw.

**Edge rule: replicate-pad, never clamp.** The crop window stays centred
everywhere and edge pixels are replicated where it leaves the frame. Clamping
would make the target's position within the crop a function of its position in
the frame, and edge proposals cluster where the subtractor sees jitter, so the
shift would be a scene-correlated confound. Condition attached: every tubelet
records its synthetic-pixel fraction. Measured, **9.89 per cent carry any,
median zero**.

## Stage 2, classifier matrix

IN PROGRESS. The training harness exists: `src/tubelets.py`, `src/metrics.py`,
`src/streaming_eval.py`, `src/arms.py` and `tools/train_arm.py`.

Four arms, one init, three seeds, twelve runs. `T = 8`, stride 1, padding
`p = 0.3`, batch 162, all ruled 12 August on measurement rather than inherited.
Training values ruled 18 August: SGD with momentum 0.9 and weight decay 1e-4,
base LR 1e-3 divided by 10 at epochs 10 and 20, a 30-epoch ceiling, early
stopping on subset val AP with patience 5, seeds 1, 2 and 3, and augmentation
of horizontal flip at p=0.5 plus brightness and contrast in [0.9, 1.1] drawn
once per tubelet.

**The schedule is identical for all four arms.** That guarantees an identical
protocol, which is Chen et al.'s own prescription, and NOT per-arm optimality.
The chapter states this, names the under-tuned-3D objection, and scopes any
result to the matched protocol.

**The schedule is largely inert and this is reported rather than fixed.** Runs
stop well before epoch 20, so the second learning-rate step never fires; the 3D
arm converges by epoch 2 and never reaches the first. The appendix run table
carries best epoch and stopping epoch for every run so this is visible.

**Weights fault, 22 August, and the guard that now prevents it.** The first
nine runs trained **from scratch**. `tools/epoch_timing.py` builds every arm
with `weights=None`, which is correct for a timing stub, and `train_arm.py`
imported `build_arm` from it to keep model definitions identical to the ones
timed. The fault surfaced through a BatchNorm diagnostic: running statistics
were at initialisation defaults, exactly 0 and 1.

`src/arms.py` now loads `ResNet18_Weights.IMAGENET1K_V1` and
`R3D_18_Weights.KINETICS400_V1`, and **refuses to return a model whose weights
are at initialisation defaults**, checking BatchNorm running statistics against
init and the first convolution against an independently built reference. It
raises rather than warns. Every run prints its preflight line.

`epoch_timing.py` is unchanged. It is committed evidence for the 0.170 s per
tubelet per epoch coefficient, which was therefore measured on randomly
initialised models. That affects nothing, since weights do not change
per-iteration cost, and it is stated rather than left to be discovered.

**The from-scratch runs are kept as an appendix baseline**, under one
condition: the table is reported only when the `r3d_18` cell is resolved,
either filled or documented as a convergence failure, which would itself be a
measured instance of the small-data finding. A three-arm table with the 3D cell
absent is not reported. Scratch runs fill idle card time only and never
displace the pretrained matrix. They are preserved in `runs_scratch/`.

**Threshold selection, ruled 21 August.** The operating threshold is the point
of maximum F1 on each run's uncapped val precision-recall curve, selected per
run at its selected checkpoint. The procedure is identical across arms while
its outputs are per-run: threshold selection joins best epoch and stopping
epoch as a protocol output. A shared threshold would be the deviation, since it
would import one arm's score calibration into another's operating point.

## Test partition

Unreachable outside an explicit `allow_test=True` call through
`src/splits.py`. **All data access goes through that module, benchmarks
included**, after a decode benchmark read the split CSV directly on 18 August
and touched test without the flag.

Val and test carry **the same three scenes**, industrial, meadow and fisheye,
because val was carved to mirror test's scene mix. `build_split.py` asserts
this and fails on a mismatch. **Lake and seaside are train-only**, being the
sole sessions of their scenes, and are reported as a note. The scope claim
follows: no lake or seaside evaluation exists anywhere in this chapter, every
evaluated claim is scoped to the three evaluated scenes, and the two train-only
scenes contribute training diversity but no evaluated numbers.

**Exception one, taken.** The gopro continuity check in `tools/inventory.py
--stage continuity` reads container metadata for `gopro_004`. No frames
decoded, no annotation read. Ruled 12 August that no deeper continuity test
runs, since it cannot change the split.

**Exception two, self-reported fault.** `tools/check_decode_4k.py` decoded 400
frames from three test clips without the flag. Throughput depends on codec and
resolution rather than frame content, so no output is tainted. Logged with its
cause and its fix.

**Exception three, taken twice.** `tools/extract_proposals.py --include-test`
decoded the test partition and read its annotations, authorised as part of
1b-E. The first run of 19 August is **VOID**: it read ground truth from an item
B output covering train and val only, so every test proposal was labelled
negative. The second run, after the upper limit was re-ruled, is the valid one.

**Exception four, NOT YET TAKEN.** `tools/evaluate_final.py --split test`
scores all twelve checkpoints in a single decode pass. It is developed and
rehearsed entirely on val, then frozen and executed **once**. It refuses to run
on test without an explicit confirmation flag. If a defect surfaces after test
has been seen, the fix and the re-run are documented rather than silent.

Any other test access needs its own ruling and its own log line.

## Declined, logged as future work

Ruled 27 August, after the crop equivalence report attributed 67 per cent of
the val pass wall clock to the twelve model forwards, which are intrinsic to a
twelve-model pass. Each was declined for this week: minutes to a couple of
hours would be trimmed from a pass that no longer races a fixed deadline, and
the scoring path would be changed to get them. The pass runs as committed.

- Clips could be transferred to the card as uint8 and converted and normalised
  there, removing the host-side float conversion and quartering the transfer.
- The host-to-device copy could be hoisted out of the per-model loop, since the
  same clips are at present transferred once per resident checkpoint.
- The six videos could be evaluated as separate processes, which would still be
  serialised on a single card.

**The test proposal count, ruled 30 August.** The committed
`data/proposals/proposals.csv` governs: 2,563,843 test proposals across seven
videos. A figure of 2,553,294 was carried in a contemporaneous report and is
superseded. The projection is unchanged.

**The flip regime, ruled 30 August.** The horizontal-flip defect fixed in
`008b7bf` divides the runs into two regimes. The twelve pretrained matrix runs
of 24 and 25 August and the nine standing scratch runs of 23 August were
trained under the constant flip; only the three `r3d_18` scratch runs of 27 and
28 August were trained under the honoured coin toss. The scratch-versus-
pretrained comparisons for the three 2D arms are same-regime and stand. The
`r3d_18` cell crosses regimes and is reported with that named: no evidence is
found that Kinetics pretraining helps the 3D arm, 0.7929 +/- 0.0098 coin-toss
against 0.7909 +/- 0.0151 constant-flip, heavily overlapping at three seeds.

Judgement, one seed, stated as such: the only measurement of the fix's effect,
`runs_flipfix/2d_single_seed2` at full val AP 0.7522 against 0.7708 for the
same arm and seed under the constant flip, moved AP down by 0.0186, so a
matched-regime comparison would more plausibly widen than close the scratch
cell's position.

Closed, so that nothing re-holds. No standing run is retrained under any flip
reasoning. The nine standing scratch runs are not re-run to chase regime
matching. Reintroducing the defect to match regimes is prohibited. The 26
August probe was pre-registered to settle whether the constant-flip matrix
stands and passed inside its declared band, and that gate binds in both
directions.

**The val rehearsal record, ruled 30 August.** A contemporaneous report of a
rehearsal killed mid-video on 26 August is not supported by any on-disk trace,
and the completed pass of 26 August 22:22:36 is taken as the record.

**GPU utilisation during the 26 August val rehearsal, unverifiable.** A 1 per
cent utilisation figure was reported for that evaluation, but no utilisation
log exists for the run; `reports/gpu_thermal.csv` samples 24 August training
and not this pass. Measured stage timings attribute the wall clock to the
twelve model forwards. The discrepancy is recorded, not explained.

## Closed, 12 September 2026

- **The from-scratch `r3d_18` cell. FILLED.** Three runs completed on 27 and 28
  August 2026 and stopped on patience, at subset AP 0.7978, 0.7881 and 0.7853
  and full uncapped val AP 0.8040, 0.7895 and 0.7853. Records in
  `runs_scratch/r3d_18_seed{1,2,3}/`.

  How the always-pretrained guard was handled. `src/arms.py` remains
  always-pretrained by default and the preflight guard is not weakened. A
  `--weights scratch` path was added at `fdaa841`, which passes
  `pretrained=False` and `verify=False` into `build_arm`, so the backbone is
  built with `weights=None` and the guard is skipped rather than defeated. The
  guard is skipped BECAUSE initialisation defaults are the correct state for a
  from-scratch run and would otherwise raise. Any run without that flag still
  meets the full check. No preflight line is printed for a scratch run, which
  is why the scratch logs carry none.

  The three earlier attempts at this cell were terminated by CUDA out of memory
  after one epoch each and are not convergence outcomes. Preserved with their
  README in `runs_terminated/`, and the correction to the figures that were
  quoted from them is at `reports/r3d_bn_diagnostic_correction.md`.

- **Thermal logging. CLOSED AS A DOCUMENTED FAILURE.** Sustained capture was
  attempted and failed silently: the sampling loop's output file was recreated
  shortly after launch, so nineteen days of samples were written to an unlinked
  inode. Six lines of `reports/gpu_thermal.csv`, five samples from 24 August
  2026 02:38 to 02:42, are the only thermal record and no re-collection is
  possible, evaluation being closed. The cost material makes no thermal claim
  beyond those samples and states timing conditions without them. Full account
  at `reports/gpu_thermal_note.md`.

## Open, not yet ruled

- **The from-scratch appendix table mixes flip regimes.** The nine non-r3d
  scratch runs predate `008b7bf` and carry the constant flip; the three
  `r3d_18` runs postdate it and carry the coin toss. A twelve-cell table drawn
  from both is not uniform on augmentation. Not ruled.
- **The flip-fix matrix re-run is one of twelve.** `runs_flipfix/` holds
  `2d_single_seed2` alone. Whether the remaining eleven run, given that
  evaluation is closed, is not ruled.
