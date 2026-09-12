# Test pass, the val-to-test shift

Written 12 September 2026 from `reports/test_final_results.json` and
`reports/val_final_results.json`. Descriptive throughout. No mechanism is
claimed for anything below.

## 1. Scene composition differs between the two splits

The pooled figures for val and test are not weighted averages over the same
mixture, and they are not taken against the same per-scene ceilings.

| Split | Total GT | gopro_fisheye | industrial | meadow |
| --- | --- | --- | --- | --- |
| val | 5,242 | 804 (15.3%) | 2,517 (48.0%) | 1,921 (36.6%) |
| test | 6,877 | 671 (9.8%) | 4,642 (67.5%) | 1,564 (22.7%) |

| Split | gopro_fisheye ceiling | industrial ceiling | meadow ceiling |
| --- | --- | --- | --- |
| val | 0.4664 | 0.9781 | 0.8053 |
| test | 0.9344 | 0.8695 | 0.9316 |

Industrial carries 67.5 per cent of test against 48.0 per cent of val, and its
proposal ceiling falls from 0.9781 to 0.8695. Fisheye and meadow ceilings both
rise. Any pooled val-to-test difference is therefore entangled with the change
in scene weighting and in per-scene ceiling, and cannot be read as a property
of an arm alone.

## 2. The per-scene view, which is the honest one

Arm means across three seeds, val to test, with the difference.

| Arm | gopro_fisheye | industrial | meadow |
| --- | --- | --- | --- |
| 2d_single | 0.4649 -> 0.9344 (+0.4695) | 0.9548 -> 0.6656 (-0.2892) | 0.5953 -> 0.9014 (+0.3061) |
| 2d_tframe_logitavg | 0.4654 -> 0.9344 (+0.4690) | 0.9755 -> 0.5868 (-0.3886) | 0.5140 -> 0.8721 (+0.3581) |
| tsm | 0.4664 -> 0.9344 (+0.4681) | 0.9691 -> 0.6269 (-0.3422) | 0.6401 -> 0.9019 (+0.2618) |
| r3d_18 | 0.4661 -> 0.9334 (+0.4674) | 0.9383 -> 0.7604 (-0.1779) | 0.7450 -> 0.9176 (+0.1726) |

Every arm moves in the same direction in every scene: up in fisheye, down in
industrial, up in meadow. The per-scene directions do not differ by arm. What
differs by arm is magnitude, and the per-scene ceilings moved too.

## 3. The pooled shift is systematic by arm, and is reported with that caveat

| Arm | Seed 1 | Seed 2 | Seed 3 | Mean | Direction within arm |
| --- | --- | --- | --- | --- | --- |
| 2d_single | +0.0163 | -0.0070 | +0.0398 | +0.0164 | split, two up one down |
| 2d_tframe_logitavg | -0.0591 | -0.0539 | -0.0416 | -0.0515 | all three down |
| tsm | -0.0435 | -0.0472 | -0.0383 | -0.0430 | all three down |
| r3d_18 | +0.0194 | +0.0328 | +0.0163 | +0.0228 | all three up |

Stated individually by arm, not grouped: `r3d_18` gains on all three seeds.
`tsm` loses on all three seeds. `2d_tframe_logitavg` loses on all three seeds.
`2d_single` nets positive at +0.0164 with seeds split two to one, seed 2 moving
-0.0070.

Nine of the twelve runs agree in direction within their arm; `2d_single` does
not. The claim that all twelve agree within arm is not supported by the table
above.

No grouping of `tsm` with `2d_tframe_logitavg` is made here. They shift in the
same direction and by similar amounts, and that is where the statement stops:
`tsm` shifts channels across the temporal dimension while
`2d_tframe_logitavg` averages per-frame logits, so a shared label describing
them as the arms that average over eight frames would assert a mechanism these
measurements do not establish.

These pooled shifts are reported beside section 2 and not instead of it. Given
the weighting change in section 1, a pooled shift is not separable from the
scene mixture, and the per-scene table is the view that does not confound the
two.

## 4. Test fisheye is saturated at its ceiling

Test fisheye comprises a single video, `gopro_004`, carrying 671 ground-truth
boxes, whose proposal stream contains **1.885 proposals per frame against
47.262 to 559.401 for the other six test videos**, so the scene is
proposal-limited at a ceiling of 0.9344 which three arms meet exactly and
`r3d_18` misses by 0.0010; the scene therefore separates no arm on test.

Measured 12 September 2026 from `reports/extraction_stats.csv`, per-video
counts already committed by the authorised 1b-E extraction. No video was
decoded and no annotation file was read.

| Video | Scene | Frames | Proposals/frame | Negatives/frame | GT boxes/frame |
| --- | --- | --- | --- | --- | --- |
| **gopro_004** | **gopro_fisheye** | **751** | **1.885** | **0.989** | **0.893** |
| 2019_08_19_C0001_5319_phantom | industrial | 2,951 | 559.401 | 558.074 | 0.953 |
| 2019_08_19_GOPR5869_1530_phantom | industrial | 1,000 | 437.091 | 435.885 | 0.960 |
| 2019_08_19_GP015869_1520_inspire | industrial | 876 | 420.215 | 417.904 | 0.992 |
| GOPR5842_002 | meadow | 612 | 85.711 | 84.703 | 0.972 |
| GOPR5842_005 | meadow | 526 | 56.711 | 55.686 | 0.935 |
| GOPR5842_007 | meadow | 511 | 47.262 | 46.313 | 0.933 |

Proposals per frame is positive plus negative plus ignored over decoded frames;
negatives per frame is the negative count over decoded frames; ground-truth
boxes per frame is `gt_boxes_in_valid_window` over decoded frames.

`gopro_004` is an outlier on the proposal stream by a factor of 25 against the
lowest of the other six, and by a factor of 297 against the highest. It is NOT
an outlier on target density: its 0.893 ground-truth boxes per frame sits just
below a range of 0.933 to 0.992 across the other six. The contrast is in how
many negatives the ranking must separate the targets from, not in how many
targets there are.

The val fisheye ceiling of 0.4664 is not explained here. Nothing in this table
bears on it and no explanation is offered.
