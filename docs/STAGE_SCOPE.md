# Stage boundaries

Ruled 11 August 2026, revised 17 and 19 August 2026. This file is the
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

**The item F repeat was CUT on 18 August**, reversing the
12 August ruling. The bound set then already established that cue counts cannot
confirm or falsify Claim A in either direction, and the three-method pass now
supplies the negative-difficulty evidence the repeat was meant to probe.

**Item G1** answers whether a kernel of a given size erases a target of a given
size. That is geometry, not a sweep. Its honest claim is that a kernel of 7 or
above costs about an eighth of the targets. **It does not establish that k=3 is
safe**, because it measures ground-truth boxes rather than real masks, and a
real mask is smaller and holier than its bounding box. That caution was
vindicated on 18 August: a 3x3 opening cost 43 points of proposal recall and
erased the drone entirely for one box in seven.

**Item D, correction outstanding.** The ceiling comparison in the item D output
is degenerate for stationary runs beginning at frame 0, where the elapsed-frame
count is 1 and the computed ceiling is 0.152 frames, which no run can satisfy.
Thirteen of thirty videos annotate from frame 0, so the reported "30 of 46
tracks fit inside the ceiling" is inflated by construction. **The write-up must
exclude start-of-clip runs from that comparison.** The history ruling does not
rest on this figure and is unaffected.

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
finding rather than a gap. See the item D correction above before quoting the
supporting counts.

**The three-method result.** MOG2, ViBe and KNN on identical frames. At
coverage 0.5, recall is 0.7455, 0.3654 and 0.5770, and the union of all three
is 0.7572. The union beats the best single method by 1.17 points and ViBe finds
nothing the others miss, so the roughly 24 per cent missed by all three is a
property of motion proposals at this target scale rather than of MOG2. MOG2 is
the ruled subtractor: it leads in every scene, both resolutions, and at matched
and at lower cost.

## Stage 1b-E, extraction

DONE 18 to 19 August 2026. Ran once at the ruled parameters, which live in the
`ruled` block of `configs/ch4.yaml` and are read through `src/ruled.py`.

Proposal recall: train 0.7120, val 0.8363, test 0.8745. Positives 13,183,
4,384 and 6,014. Training negatives drawn for seeds 1, 2 and 3 at ratio 2,
size-stratified on deciles of the positive normalised area. 89,727 tubelets
materialised, zero dropped.

**Edge rule, ruled 18 August: replicate-pad, never clamp.** The crop window
stays centred everywhere and edge pixels are replicated where it leaves the
frame. Clamping would make the target's position within the crop a function of
its position in the frame, and edge proposals cluster where the subtractor sees
jitter, so the shift would be a scene-correlated confound. Condition attached:
every tubelet records its synthetic-pixel fraction. Measured result, 9.82 per
cent of tubelets carry any synthetic pixels, median zero.

**Box-versus-blob asymmetry, for the write-up.** The area limits derive from
annotated box areas but are applied to blob areas. On the lower limit blobs are
smaller than boxes, which `analyse_size_split.py` already flags. On the **upper**
limit the asymmetry reverses, since a blob merging a drone with adjacent motion
can exceed any annotated box, so `max_area` is tighter than intended. One
sentence in the method section, and the cost is measurable from the 1b-M dumps.

## Stage 2, classifier matrix

HELD, pending the training harness, which is not yet written.

Four arms, one init, three seeds, twelve runs. `T = 8`, stride 1, padding
`p = 0.3`, batch 162, all ruled 12 August on measurement rather than inherited.
Training values ruled 18 August: SGD with momentum 0.9 and weight decay 1e-4,
base LR 1e-3 divided by 10 at epochs 10 and 20, a 30-epoch ceiling, early
stopping on subset val AP with patience 5, seeds 1, 2 and 3, and augmentation
of horizontal flip at p=0.5 plus brightness and contrast in [0.9, 1.1] drawn
once per tubelet.

**The schedule is identical for all four arms.** That guarantees an identical
protocol, which is Chen et al.'s own prescription, and NOT per-arm optimality.
The chapter must state this, name the under-tuned-3D objection, and scope any
result to the matched protocol.

## Test partition

Unreachable outside an explicit `allow_test=True` call through
`src/splits.py`. **All data access goes through that module, benchmarks
included**, after a decode benchmark read the split CSV directly on 18 August
and touched test without the flag.

**Exception one, taken.** The gopro continuity check in `tools/inventory.py
--stage continuity` reads container metadata for `gopro_004`. No frames
decoded, no annotation read. Ruled 12 August that no deeper continuity test
runs, since it cannot change the split.

**Exception two, self-reported fault.** `tools/check_decode_4k.py` decoded 400
frames from three test clips without the flag. Throughput depends on codec and
resolution rather than frame content, so no output is tainted. Logged with its
cause and its fix.

**Exception three, taken.** `tools/extract_proposals.py --include-test` decoded
the test partition and read its annotations, authorised as part of 1b-E. Note
that the first run of 19 August is **VOID**: it read ground truth from an item B
output covering train and val only, so every test proposal was labelled
negative. The second run is the valid one.

The deferred size-distribution split on test is now moot: the operating point
was frozen from train and val before any test extraction ran, and the per-size
figures the results section needs come from the extraction output.

Any other test access needs its own ruling and its own log line.

## Open, not yet ruled

Recorded here so that the boundary and the argument about it are not confused.

- **Negative concentration.** The training negative draw is stratified by size
  band and nothing else, and two videos supply about 56 per cent of the
  negatives. The 12 August item F ruling required negatives to be stratified by
  SCENE for exactly this reason, and that reasoning was not carried across.
  Fixing it means re-drawing and re-materialising.
- **The capping ratio against the natural one.** Training runs at 2:1 while
  evaluation is uncapped at roughly 342:1, which is a train-test prior mismatch
  that shifts the score distribution.
- **Results framing.** Whether proposal recall leads the results section as the
  per-partition, per-scene ceiling before any arm comparison.
