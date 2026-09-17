# Fisheye session verification, 16 September 2026

Are `gopro_004` (test) and `gopro_005` (val) one recording occasion?

Authorised provenance verification. This follows
`reports/gopro_continuity_findings.md` (12 Aug 2026), which reached
"inconclusive" on container metadata alone and recorded that a frame
comparison "would be a real test-set access and needs its own ruling". That
ruling is the authorisation for this run. 55 frames of `gopro_004` were
decoded, no annotation file was read, and the access is logged in
`reports/test_access_log.txt`.

**Verdict, 16 September 2026: `gopro_004` and `gopro_005` are distinct
recording occasions.** The author's, single rater, consistent with the
chapter's other judgement labels. Recorded in
`data/occasion_judgements.csv`; the reasoning is in section 6.

The measurement sections below were written before that verdict and are
unchanged by it. Nothing in sections 2 to 5 was revised, reweighted or
reordered to agree with the verdict, and no number moved. The verdict rests on
the author's visual inspection and precedes the measurement; the measurement
corroborates it and did not produce it.

Everything here is reproduced by:

    python tools/check_fisheye_session.py --allow-test

Artefacts: `reports/fisheye_session/` (`run.log`, `ffprobe_evidence.txt`,
`pair_similarity.csv`, `skyline_similarity.csv`, the median PNGs,
`median_side_by_side.png`, `skyline_overlay.png`, `contact_sheet.png`).

## 1. Filenames from the committed split

`data/splits/dvb_splits_v2.0-static.csv`, read through `src.splits.load_split`
with `allow_test=True`. The expectation held; actual names are as expected.

| video | session | split | boxes | annotated frames |
| --- | --- | --- | --- | --- |
| gopro_000 | gopro_treesite | train | 960 | 480 |
| gopro_001 | gopro_treesite | train | 955 | 480 |
| gopro_002 | gopro_treesite | train | 636 | 318 |
| gopro_003 | gopro_treesite | train | 554 | 554 |
| **gopro_004** | **gopro_004** | **test** | **674** | **674** |
| **gopro_005** | **gopro_005** | **val** | **810** | **810** |
| gopro_006 | gopro_path | train | 926 | 926 |
| gopro_007 | gopro_path | train | 3480 | 3480 |
| gopro_008 | gopro_path | train | 990 | 990 |

The two videos in question are the only fisheye videos whose session column is
a singleton. The split therefore asserts they are separate occasions without
carrying evidence for it on its face.

## 2. Container and stream metadata

Full dumps in `ffprobe_evidence.txt`. Every field that could bear on recording
time is either absent or identical between the two files.

| field | gopro_004 | gopro_005 |
| --- | --- | --- |
| size (bytes) | 62,835,805 | 67,896,881 |
| container duration (s) | 25.058000 | 27.060000 |
| stream duration (s) | 25.057989 | 27.059991 |
| nb_frames | 751 | 811 |
| bytes per second | 2,507,614.5 | 2,509,123.5 |
| bytes per frame | 83,669.5 | 83,720.0 |
| codec / profile / level | h264 High 4.1 | h264 High 4.1 |
| resolution, SAR, pix_fmt | 1920x1080, 1:1, yuvj420p | 1920x1080, 1:1, yuvj420p |
| colour | bt709 / bt709 / bt709, pc range | bt709 / bt709 / bt709, pc range |
| r_frame_rate | 29971/1000 | 29971/1000 |
| encoder tag | Lavf57.56.101 | Lavf57.56.101 |
| major_brand | isom | isom |
| creation_time | absent | absent |
| SMPTE timecode | 00:00:00:00 | 00:00:00:00 |
| GOP | I every 15, 50 gaps, IPPP.. no B | I every 15, 54 gaps, IPPP.. no B |
| filesystem mtime | 2020-03-06 00:13:32 | 2020-03-06 00:13:40 |

Bytes per second agree to a ratio of 1.000602 and bytes per frame to
1.000603. This is **not** evidence of a shared occasion. Both files are
constant-rate re-encodes at about 20.06 Mbit/s, so the rate is a property of
the transcode preset and cannot vary with content. For the same reason the
identical GOP structure, colour tags, profile and level carry no information
about recording: they are a signature of one packaging pass over the whole
dataset. The prior finding already established that all 77 files share
`Lavf57.56.101` with no `creation_time`.

The two filesystem mtimes are 8 seconds apart. That is the time the files were
written during packaging, not recording time, and the ordering follows the
filename sequence.

Nothing in the container metadata separates the two files, and nothing in it
joins them either.

## 3. Background comparison

Method, plainly. For each fisheye video, 50 frames were sampled at evenly
spaced indices across the whole file and the per-pixel, per-channel median
taken. With a static rig and a small moving subject, that median is the scene
with the subject removed: the background. All nine fisheye videos are labelled
`static` in `data/dvb_camera_motion.csv`, so the premise holds for every video
compared. Similarity was measured on the greyscale medians as SSIM, mean
absolute difference in 8-bit levels, RMSE and Pearson correlation.

Sampling is deterministic, evenly spaced rather than drawn; the numpy seed is
set regardless. `cv2`'s frame count is one higher than the last seekable index
on both videos, so the reader steps back to the nearest readable frame and
asserts the requested sample count, rather than silently returning 49 of 50.

### The reference bands, and a correction to them

A similarity number is not interpretable alone, so the same measure was
computed for all 36 pairs of the nine fisheye videos. Seven of the nine carry
multi-video session labels, which supplies pairs whose occasion status is
already settled, in the same scene, camera and packaging pipeline.

The session column could not be used as it stands. `reports/gopro_continuity_findings.md`
established that session `gopro_treesite` holds `gopro_000` and `gopro_001` at
25 fps and `gopro_002` and `gopro_003` at 29.97 fps, and that four clips at two
frame rates cannot be one continuous recording. Reference groups are therefore
(session label, frame rate), with the rate read from the committed
`reports/gopro_continuity.csv`:

| group | members | role |
| --- | --- | --- |
| gopro_treesite@25.0fps | gopro_000, gopro_001 | reference |
| gopro_treesite@30.0fps | gopro_002, gopro_003 | reference |
| gopro_path@30.0fps | gopro_006, gopro_007, gopro_008 | reference |
| gopro_004@30.0fps | gopro_004 | the open question |
| gopro_005@30.0fps | gopro_005 | the open question |

This correction matters to the result. Taking the raw session column would put
the four `gopro_000`/`gopro_001` against `gopro_002`/`gopro_003` pairs into the
same-occasion band, which would widen it to SSIM 0.5405..0.9536 and overlap
the different-occasion band, leaving the comparison unable to separate
anything. Under the corrected grouping those four pairs fall inside the
different-occasion band on both measures, which is independent support for the
prior finding's reading of them.

### Result

| band | n | SSIM | MAE (levels) |
| --- | --- | --- | --- |
| known SAME occasion | 5 | 0.6269 .. 0.9536 | 3.10 .. 16.99 |
| known DIFFERENT occasion | 16 | 0.4453 .. 0.5661 | 58.52 .. 70.97 |
| **gopro_004 vs gopro_005** | 1 | **0.6828** | **38.08** |

The bands do not overlap on either measure. The query pair does not sit
cleanly in either:

- SSIM 0.6828 is inside the same-occasion band, above every one of the 16
  different-occasion pairs.
- MAE 38.08 is in the gap, 21.1 levels above the same-occasion maximum and
  20.4 levels below the different-occasion minimum. Almost exactly midway.

RMSE 47.35 and Pearson r 0.6299. Full table in `pair_similarity.csv`.

### Supplementary: skyline agreement

The whole-image measures are dominated by global layout. Every video in this
scene is a smooth bright region above, a dark band near the middle and texture
below, so they score alike whatever the scene contains, and SSIM in particular
rewards that shared layout. The horizon skyline does not behave that way: its
shape is set by where the rig stood and which way it pointed, and it is
unchanged by cloud, sun angle, exposure or whether a field has been ploughed,
all of which can differ freely within one occasion.

Taken as the strongest vertical gradient per column on the blurred greyscale
median, compared column by column. The median absolute difference is the
statistic to read, not the mean: the per-column argmax latches onto cloud
edges in the overcast videos, and `gopro_000`/`gopro_001` are one occasion by
every other measure yet score a mean of 276.7 px against a median of 18 px.

| band | n | median abs difference |
| --- | --- | --- |
| known SAME occasion | 5 | 0 .. 29 px |
| known DIFFERENT occasion | 16 | 200 .. 334 px |
| **gopro_004 vs gopro_005** | 1 | **57 px** |

Also in the gap, but not centrally: 28 px above the same-occasion maximum and
143 px below the different-occasion minimum. `skyline_overlay.png` shows the
two traces on the blended medians; `gopro_004`'s horizon sits consistently
below `gopro_005`'s across the frame, and the two arcs differ in curvature as
well as offset.

## 4. Visual inspection

`contact_sheet.png`, 5 evenly spaced frames per video, `gopro_004` on the top
row and `gopro_005` on the bottom. `median_side_by_side.png` carries the two
medians and their amplified absolute difference.

What is visible, reported as observation:

- **Sky.** `gopro_004` is flat overcast throughout, uniform grey with thin
  high cloud and no blue anywhere in any of the 5 frames. `gopro_005` is broken
  cumulus against clear blue with strong bright and dark modelling in the
  cloud. The difference is consistent across every sampled frame of each, not
  a momentary change.
- **Sun and shadow.** No directional shadow is readable in `gopro_004`, which
  is consistent with the overcast. `gopro_005` shows sunlit cloud tops.
- **Ground cover.** `gopro_004` foreground is continuous green pasture.
  `gopro_005` foreground is bare ploughed brown soil with a green margin strip
  in the immediate foreground. This is the largest single contributor to the
  MAE.
- **Vegetation.** Deciduous trees are bare in both, on the right edge of
  `gopro_004` and along the treeline of `gopro_005`. Both are consistent with
  late winter or early spring.
- **Horizon furniture.** `gopro_004` shows an orange plant vehicle and what
  appear to be parked vehicles and buildings on the skyline. `gopro_005` shows
  a low treeline with pale objects along it. No object was matched across the
  two.
- **Rig.** `gopro_005` has a dark object intruding into the top right corner
  of every sampled frame, absent from `gopro_004`. The fisheye distortion
  profile and the corner vignette look alike in both.
- **Drone livery.** The subject is small in both. In `gopro_004` it is
  resolvable as a multirotor in 2 of the 5 sampled frames. In `gopro_005` it
  is not resolvable in any sampled frame at contact-sheet scale. No livery
  comparison is possible from these frames, and none is claimed.

## 5. What this does and does not establish

Established on disk:

- The two filenames are as expected, and the split gives each its own session.
- No timestamp, timecode or encoder field survives that could date either
  recording or order them. Every metadata agreement between the two files is
  explained by the shared transcode.
- Against reference bands built from pairs of known status in the same scene,
  the pair sits outside the different-occasion band on all three measures, and
  outside the same-occasion band on two of the three.
- The backgrounds differ substantially in sky state and in ground cover, and
  the horizon sits at a different height.

Not established, and not resolvable from the files:

- Whether the differences observed are larger than one occasion can produce.
  Ground cover changing from pasture to ploughed soil is not something a
  single occasion produces, but the two videos need not show the same field:
  the rig may have been turned. Nothing here fixes the bearing.
- Whether the rig was moved between the two, or the same rig repositioned.
- Any absolute date for either recording.

The measures disagree about where the pair falls, and the direction of that
disagreement is systematic: the measures most sensitive to shared layout place
it nearest the same-occasion band, the measures most sensitive to scene content
place it further away. Which of those better matches the author's meaning of
"one recording occasion" is a judgement, not a measurement.

## 6. The verdict, and how it reads the measurement

**`gopro_004` and `gopro_005` are distinct recording occasions.** Author, single
rater, 16 September 2026. Judgement label, recorded in
`data/occasion_judgements.csv` with basis `inspection+measurement`.

The verdict rests on the author's visual inspection of both videos:

- different cloud and light,
- different background composition,
- the operator vehicle present only in `gopro_004`.

It is corroborated by the background comparison in section 3, specifically the
ground state, pasture in `gopro_004` against ploughed soil in `gopro_005`.

The verdict PRECEDES the measurement and is not derived from it. This matters
for how section 3 should be read, because that section's headline number does
not point the same way. SSIM 0.6828 placed the pair inside the same-occasion
reference band. The author reads that layout similarity as **similar rig
siting, not a shared occasion**: the two arcs, the common horizon placement and
the shared fisheye geometry say the camera stood in a comparable position and
pointed a comparable way, which is what a whole-image structural measure is
sensitive to, and not that the two recordings happened on one occasion.

That reading selects the content-sensitive side of the systematic disagreement
section 5 identified, rather than resolving it. On that reading the MAE of
38.08 and the 57 px skyline offset are the informative figures and the SSIM is
measuring the rig rather than the occasion. This is an interpretation of the
measurement, offered as the author's, and it is not itself a measured result.

Section 5's account of what is and is not established on disk stands unchanged.
In particular the disk still does not fix the camera bearing, so it remains
true that the two videos need not show the same field, and the verdict does not
rest on their doing so.
