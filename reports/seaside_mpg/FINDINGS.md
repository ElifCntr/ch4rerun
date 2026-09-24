# 00_02_45_to_00_03_10_cut.mpg, 16 September 2026

Train partition, sole seaside session, train-only. No test access is involved
and none was taken. The annotation read here is a train annotation.

Reproduce with `python tools/investigate_seaside_mpg.py`. Artefacts:
`reports/seaside_mpg/` (`run.log`, `fleet_containers.csv`,
`contact_sheet.png`).

No verdict on chapter wording is drawn here.

## Headline

**Nothing is lost.** The annotation covers frames 0 to 399 and the file decodes
400 frames. **Zero boxes lie beyond the decoded range.** The shortfall is
against the filename only, and the filename is the one in the session that
announces itself as a cut.

## 1. Name arithmetic

| quantity | value |
| --- | --- |
| name parses as | 00:02:45 to 00:03:10, 165 s to 190 s |
| source-timestamp span | 25 s |
| rate | 50.000 fps, inventory and ffprobe agree |
| frames implied by the name | 25 x 50 = 1,250 |
| frames decoded | 400 |
| **shortfall** | **400/1,250 = 32.00% present** |
| absent against the name | 850 frames, 17.00 s |
| decoded duration | 8.00 s |

### The name is a source-span label, and only this file departs from it

A shortfall against a name means something only if the name is a reliable
duration label. The rest of the session tests that:

| video | named span | decoded | fps | duration | ratio | suffix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00_01_52_to_00_01_58 | 6 s | 175 | 29.970 | 5.84 s | 97.32% | none |
| **00_02_45_to_00_03_10_cut** | **25 s** | **400** | **50.000** | **8.00 s** | **32.00%** | **_cut** |
| 00_06_10_to_00_06_27 | 17 s | 497 | 29.970 | 16.58 s | 97.55% | none |
| 00_09_30_to_00_10_09 | 39 s | 1165 | 29.970 | 38.87 s | 99.67% | none |
| 00_10_09_to_00_10_40 | 31 s | 924 | 29.970 | 30.83 s | 99.45% | none |

Every seaside name without the `_cut` suffix predicts its duration to within a
second. The one name carrying `_cut` predicts 25 s and delivers 8. It is also
the only file in the session at 50 fps rather than 29.97.

## 2. Annotation coverage, the decisive check

Read from the committed annotation file through `read_boxes` in
`tools/measure_instances.py`, the same parser `tools/build_split.py` uses.

| quantity | value |
| --- | --- |
| frame entries | 400 |
| annotated frame range | 0 .. 399 |
| **maximum annotated frame** | **399** |
| **total boxes** | **399** |
| frames carrying a box | 399 |
| frames with zero boxes | 1, frame 399 |
| missing indices inside the range | 0 |
| decoded frames | 400, indices 0 to 399 |
| **boxes on frames >= 400** | **0** |

The annotation is contiguous over frames 0 to 399 with no gaps, and its last
line is `399 0`, a zero-object entry at the final decoded frame. **The
annotation ends exactly where the decode ends.** Nothing the annotator marked
is invisible to the pipeline, and no box is lost to the 8 s duration.

The trace rule is satisfied without a deficit to report: the quantity the
pipeline never sees is **zero boxes**, not an estimate.

Cross-checked against the committed record, all exact:

| artefact | committed | recount | |
| --- | ---: | ---: | --- |
| split `boxes` | 399 | 399 | match |
| split `annotated_frames` | 399 | 399 | match |
| `gt_boxes_in_valid_window` | 393 | 393 | match |

The 6-box difference between 399 and 393 is the ordinary T=8 window rule, the
first 3 and last 4 decoded frames, documented in `reports/window_exclusions.md`.
Frame 399 carries no box, which is why it is 6 and not 7.

This is the load-bearing result. The annotation was produced against the
400-frame file as distributed, not against a 25 s source span, so the file and
its labels agree and the named span never entered the labelling.

## 3. Contiguity

Packet timestamps, MPEG-PS, time_base 1/90000, 1800 ticks per frame at 50 fps.

| quantity | value |
| --- | --- |
| packets | 400, matching the 400 decoded |
| packet durations | all 1800 ticks |
| PTS first / last | 46800 / 765000 = 0.520 s .. 8.500 s |
| packets with no PTS | 1, at index 142 |
| PTS value absent from the ladder | 306000 |
| DTS deltas | 399 deltas, all exactly 1800 |
| DTS monotonic | yes, no backward step |
| DTS last / expected | 763200 / 763200 |
| keyframes | 3, at packet 0, 199, 397 |
| GOP gaps | 199 and 198 |
| **contiguous** | **yes, no gap and no reset** |

One packet carries no PTS, which puts a single 3600 gap in the sorted PTS
ladder. That is a missing timestamp, not a discontinuity: the DTS ladder is
complete across it, all 400 values with every delta exactly 1800, and
restoring the single absent value 306000 completes the PTS ladder. The
reordering visible around index 142 is ordinary B-frame reordering, the stream
declaring `has_b_frames=1`.

The container reports `start_time=0.520000`, so the timeline begins at 0.52 s
rather than 0. That is an offset, not a gap, and it is consistent end to end:
0.520 + 8.000 = 8.520, and the last frame's PTS of 8.500 plus one frame of
0.020 gives the same.

**Visual inspection**, `contact_sheet.png`, frames 0, 66, 133, 200, 266, 332,
399, the first and last among them. A single coastal shot throughout: white
chalk cliffs at the left, open sea, blue sky above a continuous band of low
cloud. Sea state, light and cloud structure are consistent frame to frame and
evolve smoothly, with no cut, no scene jump and no exposure step. The cliff's
apparent position and scale shift modestly across the 8 s. The 8 s reads as
one contiguous cut.

## 4. The seaside session, and the +2 overcount

| video | ext | container frames | decoded | over | fps | codec | encoder |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 00_01_52_to_00_01_58 | .mp4 | 177 | 175 | **+2** | 29.970 | mpeg2video | Lavf57.56.101 |
| **00_02_45_to_00_03_10_cut** | **.mpg** | **400** | **400** | **0** | **50.000** | **mpeg2video** | **none** |
| 00_06_10_to_00_06_27 | .mp4 | 499 | 497 | **+2** | 29.970 | mpeg2video | Lavf57.56.101 |
| 00_09_30_to_00_10_09 | .mp4 | 1167 | 1165 | **+2** | 29.970 | mpeg2video | Lavf57.56.101 |
| 00_10_09_to_00_10_40 | .mp4 | 926 | 924 | **+2** | 29.970 | mpeg2video | Lavf57.56.101 |

**The .mpg does not carry the overcount.** Its container frame count and its
decoded count are both 400.

Across the whole 37-video static subset exactly four videos carry an
overcount, all +2, and they are the four seaside `.mp4` files above. No other
video in the fleet has one.

**The anomalies are not independent. They share a precise signature:**

| codec, extension, encoder | videos | overcounts |
| --- | ---: | --- |
| h264, .mp4, Lavf57.56.101 | 24 | all 0 |
| h264, .mp4, Lavf58.20.100 | 8 | all 0 |
| **mpeg2video, .mp4, Lavf57.56.101** | **4** | **all +2** |
| mpeg2video, .mpg, none | 1 | 0 |

The overcount occurs in every file that is MPEG-2 video muxed into MP4, and in
no other file. It is not a property of MPEG-2 as such, since the `.mpg` is also
MPEG-2 and counts correctly, nor of the `Lavf57.56.101` mux as such, since 24
h264 files share that encoder tag and count correctly. It tracks the
combination: MPEG-2 in an MP4 container. The four are one signature, not four
coincidences.

## 4b. The +2 signature accounts for the 31,646 against 31,654 discrepancy

Recorded 24 September 2026, on the ruling that the mechanism in section 4
joins these findings as the explanation of that discrepancy.

| quantity | value | source |
| --- | ---: | --- |
| sum of `decoded_frames`, 37 static videos | 31,646 | `reports/extraction_stats.csv` |
| `frames` in the extraction metadata | 31,646 | `data/proposals/extraction_meta.json` |
| sum of `inventory_frames`, 37 static videos | 31,654 | `reports/extraction_stats.csv` |
| sum of `cv2_frame_count`, 37 static videos | 31,654 | `reports/video_inventory.csv` |
| **difference** | **8** | |
| sum of per-video overcount | 8 | `reports/seaside_mpg/fleet_containers.csv` |

The difference is 8 and the overcount total is 8. They are the same eight
frames. Four files carry +2 each, and they are the four seaside `.mp4` files
listed in section 4, every one of them MPEG-2 video muxed into MP4:

    00_01_52_to_00_01_58     +2
    00_06_10_to_00_06_27     +2
    00_09_30_to_00_10_09     +2
    00_10_09_to_00_10_40     +2

The account closes exactly, with no residual and no video unexplained. The
container frame count overstates by two in each of those four files and by
nothing anywhere else in the subset. The `.mpg` contributes nothing to the
discrepancy: its container and decoded counts are both 400.

Note which figure is the honest one. 31,646 is the count of frames actually
decoded and is what the pipeline saw; 31,654 is what the containers claim.
The eight-frame gap is a container-metadata artefact, not missing data.

## 4c. Why the name is wrong rather than the file

Recorded 24 September 2026, on the ruling that the reliability table in
section 1 is the warrant for that reading.

The table in section 1 is what licenses calling the filename wrong rather than
the file truncated. Every seaside name without the `_cut` suffix predicts its
own duration to within 3 per cent:

| video | named span | duration | ratio | deviation |
| --- | ---: | ---: | ---: | ---: |
| 00_01_52_to_00_01_58 | 6 s | 5.84 s | 97.32% | 2.68% |
| 00_06_10_to_00_06_27 | 17 s | 16.58 s | 97.55% | 2.45% |
| 00_09_30_to_00_10_09 | 39 s | 38.87 s | 99.67% | 0.33% |
| 00_10_09_to_00_10_40 | 31 s | 30.83 s | 99.45% | 0.55% |
| **00_02_45_to_00_03_10_cut** | **25 s** | **8.00 s** | **32.00%** | **68.00%** |

Four of five names are duration labels accurate to within 3 per cent. The
fifth is out by 68 per cent, and it is the only one carrying the `_cut`
suffix, and the only one at 50 fps.

This is the warrant, not the proof. What settles the matter is section 2: the
annotation covers frames 0 to 399 and stops, so the labels were made against
the 400-frame file and the named 25 s span never entered the labelling. The
table establishes that a seaside filename is normally a reliable duration
label, which is what makes its failure here a fact about the name.

## 5. The .mpg against the fleet

| property | 00_02_45_to_00_03_10_cut | the other 36 |
| --- | --- | --- |
| container | MPEG-PS (`mpeg`) | MP4 (`mov,mp4,m4a,3gp,3g2,mj2`), all 36 |
| extension | `.mpg` | `.mp4`, all 36 |
| codec | mpeg2video | h264 (32), mpeg2video (4) |
| encoder tag | **absent** | Lavf57.56.101 (28), Lavf58.20.100 (8) |
| frame rate | 50.000 | 25.0 (10), 29.97 (13), 29.971 (12), 29.969 (1) |
| probe score | 26 | 100 |
| creation_time | absent | absent, all 36 |

It is the only MPEG-PS container, the only `.mpg`, the only file with no
encoder tag, and the only file at 50 fps. **It did not go through the Lavf
re-encode that produced every other file in the subset.** It is the one file
still in a delivery-native container.

Bearing on upstream against local: the file was not re-encoded locally, so its
8 s of content is what was distributed, and the annotation supplied with the
dataset matches those 8 s exactly. Both the video and its labels are
internally consistent as received. There is no evidence of a local truncation,
and no evidence of any truncation relative to the annotation. What remains is
a filename whose span does not describe the file, on the one file in the
session explicitly suffixed `_cut`.

## An unrelated correction this turned up

`reports/gopro_continuity_findings.md` states: "Every one of the 77 files
carries `encoder=Lavf57.56.101` and no `creation_time`". Within the 37-video
static subset that is **half right**.

- No `creation_time` survives on any of the 37. That half holds.
- The single-encoder half does not. 28 carry `Lavf57.56.101`, 8 industrial
  files carry **`Lavf58.20.100`**, and this `.mpg` carries none.

`reports/gopro_continuity.csv` records that the check behind that sentence
measured 9 videos, `gopro_000` to `gopro_008`, so the claim about 77 was a
generalisation past its evidence. The fisheye conclusions that rest on it are
unaffected, because those nine files do all carry `Lavf57.56.101` and no
`creation_time`, as `reports/fisheye_session/ffprobe_evidence.txt` shows
directly. The same overstated sentence is repeated in
`reports/fisheye_session/FINDINGS.md` section 2.

**Both were corrected on 24 September 2026**, on the ruling of that date, by
dated correction rather than silent rewrite. Neither original sentence was
altered; each is marked in place and answered in a dated section:
`reports/gopro_continuity_findings.md`, "Correction, 24 September 2026", and
`reports/fisheye_session/FINDINGS.md` section 7. The corrected statement is
that of the 37 static videos 28 carry `Lavf57.56.101`, 8 industrial files
carry `Lavf58.20.100` and the `.mpg` carries no encoder tag, with no
`creation_time` on any of the 37, so every timestamp conclusion stands and the
fisheye conclusions are unaffected.
