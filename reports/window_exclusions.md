# The 28-box and 17-box deficits, derived exactly

16 September 2026. Both close exactly. Nothing is approximated below.

Reproduce with `python tools/derive_window_exclusions.py`, which asserts the
closure and exits non-zero on any shortfall. Artefacts:
`reports/window_exclusions.csv`, `reports/window_exclusions.log`.

## The rule as implemented

`configs/ch4.yaml:90` rules `window.T: 8`. `src/streaming_eval.py:41-43`:

    T = 8
    LEAD = (T - 1) // 2      # 3
    TRAIL = T - 1 - LEAD     # 4

The denominator is built by `count_ground_truth`, `src/streaming_eval.py:168`,
whose frame loop at **`src/streaming_eval.py:176`** is the exclusion:

    for f in range(LEAD, n - TRAIL):

For a video of `n` decoded frames the evaluated centres are frames 3 to n-5
inclusive. Boxes on the first `LEAD` = 3 and the last `TRAIL` = 4 decoded
frames are never counted, **7 frames per video**, because those frames cannot
sit at the centre of a full T = 8 window. The hypothesis put to this check,
window eligibility at video boundaries, is what the code does.

It is the same arithmetic the scoring loop uses at
`src/streaming_eval.py:135-136`, where a centre is emitted only once the
rolling buffer is full:

    if len(buf) == T:
        centre = i - TRAIL

so the denominator and the scored set are eligible on identical terms. The
asymmetry, 3 before and 4 after, follows from `(T-1)//2` with T even, and is
the ruled convention recorded in the module docstring at lines 21 to 23.

`n` is `decoded_frames`, not the container frame count:
`tools/evaluate_final.py:278-280` reads it from `reports/extraction_stats.csv`,
and `tools/evaluate_final.py:287` passes it to `count_ground_truth`.

## Why a committed artefact can answer this

`reports/extraction_stats.csv` carries `gt_boxes_in_valid_window` per video.
`tools/extract_proposals.py:256-257` builds that window as

    first_valid = max(lead, warmup)
    last_valid  = n_dec - 1 - trail

with `lead` and `trail` derived identically at `tools/extract_proposals.py:170-171`,
and `bgs.warmup_frames` ruled **0** at `configs/ch4.yaml:105`. So
`first_valid = max(3, 0) = 3` and `last_valid = n - 5`: the identical window,
over the identical `n`. The column is therefore the evaluator's own
denominator, already on disk, and no new decode or annotation read is needed.

## Test: 6,905 - 28 = 6,877

Evaluator's reported total is `n_gt_boxes` in `reports/test_final_results.json`.

| video | scene | split boxes | decoded | centres | counted | excluded |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2019_08_19_C0001_5319_phantom | industrial | 2820 | 2951 | 2944 | 2813 | 7 |
| 2019_08_19_GOPR5869_1530_phantom | industrial | 964 | 1000 | 993 | 960 | 4 |
| 2019_08_19_GP015869_1520_inspire | industrial | 876 | 876 | 869 | 869 | 7 |
| GOPR5842_002 | meadow | 598 | 612 | 605 | 595 | 3 |
| GOPR5842_005 | meadow | 492 | 526 | 519 | 492 | 0 |
| GOPR5842_007 | meadow | 481 | 511 | 504 | 477 | 4 |
| gopro_004 | gopro_fisheye | 674 | 751 | 744 | 671 | 3 |
| **TOTAL** | | **6905** | | | **6877** | **28** |

Per scene the counted column gives industrial 2813+960+869 = 4642, meadow
595+492+477 = 1564, fisheye 671, which are the three `per_scene.n_gt` values in
`reports/test_final_results.json` exactly.

## Val: 5,259 - 17 = 5,242

| video | scene | split boxes | decoded | centres | counted | excluded |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2019_11_14_C0001_3922_matrice | industrial | 2524 | 2601 | 2594 | 2517 | 7 |
| GOPR5843_002 | meadow | 393 | 424 | 417 | 389 | 4 |
| GOPR5843_005 | meadow | 524 | 571 | 564 | 524 | 0 |
| GOPR5846_002 | meadow | 629 | 686 | 679 | 629 | 0 |
| GOPR5846_005 | meadow | 379 | 450 | 443 | 379 | 0 |
| gopro_005 | gopro_fisheye | 810 | 811 | 804 | 804 | 6 |
| **TOTAL** | | **5259** | | | **5242** | **17** |

Per scene: industrial 2517, meadow 389+524+629+379 = 1921, fisheye 804, which
are the three `per_scene.n_gt` values in `reports/val_final_results.json`
exactly.

## Independent frame-level derivation, val

The tables above rest on an aggregate column. For val there is also a
committed per-box record, `reports/instances_boxes.csv`, holding all 5,259 val
boxes with their frame numbers. Counting boxes on frames `< 3` or `> n-5`
directly reproduces the aggregate video by video, and names the frames:

| video | head | tail | total | excluded frames carrying boxes |
| --- | ---: | ---: | ---: | --- |
| 2019_11_14_C0001_3922_matrice | 3 | 4 | 7 | 0, 1, 2, 2597, 2598, 2599, 2600 |
| GOPR5843_002 | 0 | 4 | 4 | 420, 421, 422, 423 |
| GOPR5843_005 | 0 | 0 | 0 | none |
| GOPR5846_002 | 0 | 0 | 0 | none |
| GOPR5846_005 | 0 | 0 | 0 | none |
| gopro_005 | 3 | 3 | 6 | 0, 1, 2, 807, 808, 809 |
| **TOTAL** | | | **17** | |

This is what establishes the rule empirically rather than by assumption. The
four videos with a zero deficit simply carry no annotated box in their first 3
or last 4 frames, and the boxes lost are exactly those the window cannot
centre.

## What is and is not established

Established: both deficits close exactly, 28 and 17, with no residual and no
video unaccounted for; the rule is a single frame-range expression at
`src/streaming_eval.py:176`; and for val the per-video split of the deficit is
confirmed independently at frame level against a second committed artefact.

Not established to the same standard: the **test** per-video split rests on
`gt_boxes_in_valid_window` rather than a frame-level count, because no
committed artefact carries per-frame test boxes.
`reports/instances_boxes.csv` covers train and val only, and
`data/proposals/proposals.csv` cannot substitute, since proposals exist only
inside the valid window by construction and so contain no excluded-frame rows
at all. A frame-level test derivation would require reading the test
annotation files, which is a fresh test access and was not taken. The test
arithmetic is exact, and it applies through shared code a rule verified at
frame level on val.

One incidental observation, not part of the deficit: the split's `boxes` and
`annotated_frames` columns differ by one for `GOPR5842_007`, 481 against 480,
so one frame there carries two boxes. Test boxes total 6,905 against 6,904
annotated frames for that reason.
