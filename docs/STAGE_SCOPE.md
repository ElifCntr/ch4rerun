# Stage boundaries

Ruled 11 August 2026. This file is the repository's record of what may run at
each stage and under what constraint. It exists because the boundary moved once
already, and a boundary that lives only in a chat is not a boundary.

## Stage 1a, measurement

Strictly parameter-free and ground-truth-only. No background subtractor runs.
No number produced here comes from a chosen parameter value.

| Item | What it measures |
| --- | --- |
| A | Per-video inventory. Frame rate read from the file, not from a manifest |
| B | Size distribution of annotated instances only |
| C | Track length, consecutive annotated frames per instance |
| D | Per-frame centre drift, in pixels and as a fraction of box size |
| E | Crop-retention strips, T in {3, 5, 8, 16}, stride 1 and 2 |
| F | Motion separability, drone tubelets against random background crops |
| G1 | Kernel-versus-target-size geometry, arithmetic on B's measured sizes |
| H | Padding sweep against drift |

Item F takes random crops from non-drone regions of the same frames,
size-matched to the drone distribution. It uses no proposals and no BGS, and
it still answers whether drone motion is separable from clutter.

Item G1 answers whether a kernel of a given size erases a target of a given
size. That is geometry, not a sweep.

Train and val only.

## Stage 1b-M, measurement with a declared provisional operating point

Authorised, runs only after Stage 1a reports. Every output is labelled
provisional, and no number from it may be quoted as a chapter result. Its
purpose is to produce the inputs from which the real parameters are ruled.

Operating point, all of it declared rather than chosen:

- MOG2 `history` from the Zivkovic and van der Heijden 2006 absorption
  formula against a stated absorption target
- `varThreshold` and `learningRate` at OpenCV defaults, reported as defaults
- `min_area` and `max_area` provisionally from Stage 1a item B
- `detectShadows = false`

Carries item B second half (proposal size distribution), item G2 (the
proposal-recall morphology sweep) and the coverage curves.

## Stage 1b-E, extraction

HELD. Runs once, at the ruled parameters. If a ruling moves the upstream
operating point materially, 1b-M's sweeps are re-run first.

## Stage 2, classifier matrix

HELD. Four arms, one init, three seeds, twelve runs.

## Test partition

Unreachable outside an explicit `allow_test=True` call. One declared
exception exists, the gopro continuity check in `tools/inventory.py --stage
continuity`, which reads container metadata for `gopro_004` and appends to
`reports/test_access_log.txt`. No frames are decoded and no annotation file is
read. Any deeper continuity test is a real test-set access and needs its own
ruling.
