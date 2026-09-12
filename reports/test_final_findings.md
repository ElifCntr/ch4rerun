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

Test fisheye carries a proposal ceiling of 0.9344. Three arms reach exactly
0.9344 and `r3d_18` reaches 0.9334, so the scene is proposal-limited for every
arm and separates none of them. The scene is a single video, `gopro_004`,
carrying 671 ground-truth boxes, per
`data/splits/dvb_splits_v2.0-static.csv` line 8.

Why that video's ceiling is so much higher than val fisheye's 0.4664 is not
established by any committed artefact in this repository. No explanation is
offered here.
