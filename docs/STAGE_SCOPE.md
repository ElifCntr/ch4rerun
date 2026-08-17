# Stage boundaries

Ruled 11 August 2026, revised 17 August 2026. This file is the repository's
record of what may run at each stage and under what constraint. It exists
because the boundary moved once already, and a boundary that lives only in a
chat is not a boundary.

Revision note. The 11 August version of this file recorded the item F
boundary correctly and the boundary was still wrong, for reasons given under
item F below. A boundary being written down does not make it right, only
auditable.

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

**Item E, amended 12 August.** Stride 2 was dropped from the first pass and
its budget spent on a second padding condition, because stride changes span,
which is already understood arithmetically, whereas padding changes what the
rater actually sees. Stride 2 later ran as a narrowed confirmation pass at
T=8 only, 30 strips. Both passes are complete.

**Item F, amended 12 August, and this correction matters.** The 11 August
version of this file claimed item F "still answers whether drone motion is
separable from clutter". It does not. Item F's negatives are random crops
from drone-free frames, whereas Stage 2's negatives are unmatched motion
proposals, which by construction moved enough to trigger the subtractor.
Random sky is trivially not a drone; motion-selected clutter is not. Appearance
alone scored 23 of 23, leaving no headroom for motion to demonstrate anything,
so item F as run cannot answer the question in either direction.

The cause is the 11 August rescope recorded in this file. Making item F
runnable inside a parameter-free Stage 1a changed what it measures. The
original run is kept and reported as a floor, and as validation that crops at
T=8 and p=0.3 are legible enough to support the task at all. It is never
reported as evidence about Claim A's premise. The repeat against real
unmatched proposals sits in Stage 1b-M.

**Item G1** answers whether a kernel of a given size erases a target of a given
size. That is geometry, not a sweep. Its honest claim is that a kernel of 7 or
above costs about an eighth of the targets. **It does not establish that k=3 is
safe**, because it measures ground-truth boxes rather than real masks, and a
real mask is smaller and holier than its bounding box.

## Stage 1b-M, measurement with declared provisional operating points

IN PROGRESS. Every output is labelled provisional, and no number from it may be
quoted as a chapter result. Its purpose is to produce the inputs from which the
real parameters are ruled.

Operating point, all of it declared rather than chosen:

- `learningRate` **left at the OpenCV default of -1 and never passed
  explicitly**. Measured 15 August: `apply()` uses any non-negative rate you
  pass and only derives alpha from `history` otherwise, so passing a rate makes
  `history` inert. The original Chapter 4 config passed 0.001 and its
  `history = 200` therefore had no effect on a single frame.
- `history` **not derived from an absorption target**. That derivation was
  abandoned 17 August as unsound on this data, for measured reasons recorded
  below.
- `varThreshold` at the OpenCV default, reported as a default.
- `complexityReductionThreshold` at OpenCV's default of 0.05, with the
  divergence from Zivkovic 2004's suggested cT = 0.01 **named explicitly**
  rather than passed over as "OpenCV defaults".
- `min_area` and `max_area` from Stage 1a item B. `max_area` is set **per
  resolution**, not as a single frame-area-normalised value, because the
  maximum annotated area fraction differs 15.5x between 1080p and 4K.
  `min_area` normalises cleanly at a 3.3x spread and needs no such split.
- `detectShadows = false`.
- `cv2.ocl` state recorded with every MOG2 run. MOG2 has a separate OpenCL code
  path and all results so far came from it.

Why the absorption-target derivation was dropped. MOG2's learning rate ramps as
`1/min(2*nframes, history)`, so alpha is set by elapsed frames rather than by
`history` until half a history has passed. Absorption is therefore capped at a
fraction of the time elapsed when a target stops moving, whatever `history`
is set to. Item D measured 46 tracks: 30 have their longest stationary run
inside that cap and 16 do not. A single absorption target in seconds is not
uniformly achievable, so the derivation is replaced and the analysis is written
up as a methodological finding rather than a gap.

Carries item B second half (proposal size distribution), item G2 (the
proposal-recall morphology sweep), the coverage curves, and the item F repeat
against real unmatched proposals.

## Stage 1b-E, extraction

HELD. Runs once, at the ruled parameters. If a ruling moves the upstream
operating point materially, 1b-M's sweeps are re-run first.

## Stage 2, classifier matrix

HELD. Four arms, one init, three seeds, twelve runs. `T = 8`, stride 1,
padding `p = 0.3`, batch 162, all ruled 12 August on measurement rather than
inherited.

## Test partition

Unreachable outside an explicit `allow_test=True` call.

**Declared exception, taken.** The gopro continuity check in
`tools/inventory.py --stage continuity` reads container metadata for
`gopro_004` and appends to `reports/test_access_log.txt`. No frames are decoded
and no annotation file is read. Ruled 12 August that no deeper continuity test
runs, since it cannot change the split.

**Declared exception, deferred not refused.** The size-distribution split on the
test partition is authorised only **after** the 1b-E operating point is frozen
from train and val. Knowing that test is weighted toward the tiny
industrial-at-1080p regime could bias the operating point toward a small
`min_area`, and the figure is not needed until the results write-up, where
per-size-band reporting requires it anyway.

Any other test access needs its own ruling and its own log line.

## Under review, not yet ruled

Recorded here so that the boundary and the argument about the boundary are not
confused with one another.

- Whether the multi-axis sweep runs at all, or every parameter is declared with
  a stated rationale. Ruled 17 August that `history` becomes a swept axis; a
  scope review the same day proposes cutting the sweep entirely.
- Whether ViBe and KNN stay on the critical path or move to an appendix
  proposal-coverage comparison.
- Whether the item F repeat runs.
- Whether a parameter may be declared with a rationale rather than measured.
  The no-invented-numbers instruction of 7 August came from Elif; the scoping
  of it came from the manager. Elif's confirmation is outstanding.
