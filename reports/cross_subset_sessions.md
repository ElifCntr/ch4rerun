# Cross-subset session table, val against test

16 September 2026. From the committed split and inventory only; no video was
decoded and no annotation read. Reproduce with
`python tools/cross_subset_sessions.py`. Artefacts:
`reports/cross_subset_sessions.csv`, `reports/cross_subset_sessions.log`.

**Two kinds of statement appear below and must not be read alike.** A FLAG is
measured: it says the committed metadata cannot separate two sessions, not that
they are one. A VERDICT is the author's judgement, single rater, recorded in
`data/occasion_judgements.csv` with its rater, date and basis. No verdict here
is derived from a flag or from any measurement; where a measurement
corroborates a verdict, the verdict came first and the register says so.
`tools/cross_subset_sessions.py` draws no verdict of its own, and prints the
flag and the verdict side by side so the two can always be read apart.

## The table

Source: `data/splits/dvb_splits_v2.0-static.csv` for scene, session and split;
`reports/video_inventory.csv` for frame rate, resolution and duration.

| video | scene | session | split | fps | duration s | resolution | filename date | camera token |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| gopro_004 | gopro_fisheye | gopro_004 | **test** | 29.970 | 25.058 | 1920x1080 | none | gopro_004 |
| gopro_005 | gopro_fisheye | gopro_005 | **val** | 29.970 | 27.060 | 1920x1080 | none | gopro_005 |
| 2019_08_19_C0001_5319_phantom | industrial | ind_2019_08_19 | **test** | 25.000 | 118.040 | 3840x2160 | 2019-08-19 | C0001 |
| 2019_08_19_GOPR5869_1530_phantom | industrial | ind_2019_08_19 | **test** | 25.000 | 40.000 | 1920x1080 | 2019-08-19 | GOPR5869 |
| 2019_08_19_GP015869_1520_inspire | industrial | ind_2019_08_19 | **test** | 25.000 | 35.040 | 1920x1080 | 2019-08-19 | GP015869 |
| 2019_11_14_C0001_3922_matrice | industrial | ind_2019_11_14 | **val** | 25.000 | 104.040 | 3840x2160 | 2019-11-14 | C0001 |
| GOPR5842_002 | meadow | meadow_5842 | **test** | 29.971 | 20.420 | 1920x1080 | none | GOPR5842 |
| GOPR5842_005 | meadow | meadow_5842 | **test** | 29.970 | 17.551 | 1920x1080 | none | GOPR5842 |
| GOPR5842_007 | meadow | meadow_5842 | **test** | 29.971 | 17.050 | 1920x1080 | none | GOPR5842 |
| GOPR5843_002 | meadow | meadow_5843 | **val** | 29.969 | 14.148 | 1920x1080 | none | GOPR5843 |
| GOPR5843_005 | meadow | meadow_5843 | **val** | 29.971 | 19.052 | 1920x1080 | none | GOPR5843 |
| GOPR5846_002 | meadow | meadow_5846 | **val** | 29.971 | 22.889 | 1920x1080 | none | GOPR5846 |
| GOPR5846_005 | meadow | meadow_5846 | **val** | 29.970 | 15.015 | 1920x1080 | none | GOPR5846 |

`creation_time` is empty for all 13, as it is for all 77 files in the dataset.
Only the four industrial videos carry a date at all, and they carry it in the
filename.

## What each session assignment rests on

`data/provenance/dvb_splits_v1_README.md` records that sessions and scenes were
"documented from the 2026-08-04 filename + thumbnail audit".
`tools/build_split.py` carries the session column through from that manifest
and asserts zero session overlap between partitions, so the integrity
guarantee is exactly as strong as the label the audit produced and no stronger.

| session | scene | split | videos | corroborating metadata |
| --- | --- | --- | ---: | --- |
| gopro_004 | gopro_fisheye | test | 1 | **none, audit only** |
| gopro_005 | gopro_fisheye | val | 1 | **none, audit only** |
| ind_2019_08_19 | industrial | test | 3 | shared filename date 2019-08-19 |
| ind_2019_11_14 | industrial | val | 1 | filename date 2019-11-14 |
| meadow_5842 | meadow | test | 3 | shared camera file number 5842 |
| meadow_5843 | meadow | val | 2 | shared camera file number 5843 |
| meadow_5846 | meadow | val | 2 | shared camera file number 5846 |

The meadow entries need a distinction. What the shared file number corroborates
is that the videos *within* one session belong together. It does not
corroborate the *boundary* between meadow_5842 and meadow_5843, which rests
entirely on the premise that a new camera file number begins a new occasion.
That premise is not tested anywhere in the record, and the calibration below
shows it does not hold in general.

## Flags

Criteria as asked: a filename date shared across the boundary within a scene,
adjacent camera numbering across the boundary, or a session resting on the
thumbnail audit alone with no corroborating metadata.

### gopro_fisheye: FLAGGED, on two criteria

- **ADJACENT.** gopro_004 (test) and gopro_005 (val) differ by 1.
- **AUDIT.** Both sessions are singletons with no filename date and no shared
  camera number. They are the only two evaluated sessions resting on the audit
  alone.

Numbering across all splits: 0 train, 1 train, 2 train, 3 train, **4 test**,
**5 val**, 6 train, 7 train, 8 train.

Caveat on this flag: `gopro_000` to `gopro_008` are dataset-assigned names, and
no committed artefact records whether that order reflects camera recording
order. The adjacency therefore rests on an ordering whose provenance is not
established. What has been measured on this pair is in
`reports/gopro_continuity_findings.md` and `reports/fisheye_session/FINDINGS.md`.

### meadow: CLOSED. Distinct occasions.

**Verdict, 16 September 2026.** GOPR5842 and GOPR5843 are distinct recording
occasions. The scene conditions differ.

Basis: the author's visual inspection. Judgement label, single rater, 16
September 2026, recorded on the same footing as the chapter's other judgement
labels, `data/dvb_camera_motion.csv` among them. The verdict is the author's
and it closes the flag; it is not a measured result and is not presented as
one. No background comparison was run on the meadow pair, unlike the fisheye
pair, so inspection is the whole of the basis.

What the flag had been raised on, retained as the record of why the question
was asked:

- **ADJACENT.** GOPR5842 (test) and GOPR5843 (val) differ by 1.

Numbering across all splits: **5842 test**, **5843 val**, 5844 train, 5845
train, **5846 val**, 5848 train. The scene's session labels are one per camera
file number over a contiguous run, and that run is distributed across all three
partitions. 5847 is the same run and is labelled scene `lake`, in train, which
means the camera numbering continues across a scene change.

No filename date exists on any meadow video, so nothing separated 5842 from
5843 on disk except the file number itself. The adjacency was therefore
unresolvable from metadata, and it is the inspection above that resolves it.

### industrial: NOT FLAGGED

The filename dates separate the two sides: test is 2019-08-19 throughout, val
is 2019-11-14, 87 days apart.

Worth recording because it looks like a flag and is not: C0001 appears on both
sides of the boundary, in `2019_08_19_C0001_5319_phantom` (test) and
`2019_11_14_C0001_3922_matrice` (val). The dates differ, so the numbering
coincidence is not evidence of a shared occasion.

A loose end inside test, which does not affect the val/test boundary and is
recorded only so it is not mistaken for settled. `2019_08_19_GOPR5869_1530_phantom`
and `2019_08_19_GP015869_1520_inspire` share the camera file number 5869, which
on the usual GoPro convention would make them chapter 1 and a later chapter of
one continuous recording. Their filenames disagree with that reading: they
carry different airframes, phantom against inspire, and different times, 1530
against 1520, with the later chapter bearing the earlier time. Both sit in one
session and one partition, so nothing here turns on it, but the session
grouping of these two is not explained by the chaptering alone.

## How much an ADJACENT flag is worth

Only the industrial videos carry dates, so they are the only place camera
numbering can be calibrated against a known date. Two facts, pulling opposite
ways, both from the committed split:

- **The counter resets.** `2019_08_19_C0001_5319_phantom` and
  `2019_11_14_C0001_3922_matrice` share the token C0001 yet are 87 days apart.
  Equal numbers do not imply one occasion.
- **The counter also persists across outings.**
  `2019_08_19_GOPR5869_1530_phantom` and `2019_09_02_GOPR5871_1058_solo` are 2
  apart in GOPR numbering yet are 14 days apart.
  `2019_08_19_C0001_5319_phantom` and `2019_09_02_C0002_2527_inspire` are 1
  apart and 14 days apart. A gap of 1 demonstrably spans more than one
  occasion in this dataset.

So a small numbering gap is consistent with one occasion and does not
establish it, and the meadow and fisheye videos carry no date at all against
which to check. Both flags mark what cannot be separated on disk, not what is
joined.

## Status: COMPLETE

Every val/test scene boundary in the three evaluated scenes is resolved, and
every one now carries a verdict. All three read distinct occasions.

Dispositions are recorded in `data/occasion_judgements.csv` and reproduced by
`tools/cross_subset_sessions.py`, which prints the metadata flag and the
author's disposition side by side so the measured position and the judged
position can always be read apart.

| scene | metadata flag | basis | verdict |
| --- | --- | --- | --- |
| industrial | none | documentary, dates 87 days apart | distinct occasions |
| meadow | ADJACENT | inspection, author, 16 Sep 2026 | **distinct occasions** |
| gopro_fisheye | ADJACENT, AUDIT | inspection, corroborated by measurement, author, 16 Sep 2026 | **distinct occasions** |

- **industrial** is settled documentarily. The filename dates separate the
  partitions, 2019-08-19 against 2019-11-14, and no judgement was needed.
- **meadow** is closed by the author's visual inspection above, single rater,
  16 September 2026: GOPR5842 and GOPR5843 are distinct recording occasions.
- **gopro_fisheye** is closed by the author's visual inspection, single rater,
  16 September 2026: gopro_004 and gopro_005 are distinct recording occasions.
  The basis is different cloud and light, different background composition, and
  the operator vehicle present only in gopro_004. The verdict is corroborated
  by the committed background comparison in
  `reports/fisheye_session/FINDINGS.md`, on the ground state, pasture against
  ploughed soil. The verdict precedes that measurement and is not derived from
  it, which the register records as basis `inspection+measurement`.

  One point of that comparison needs reading with the verdict, and section 6 of
  `FINDINGS.md` sets it out: SSIM placed the pair inside the same-occasion
  reference band. The author reads that layout similarity as similar rig siting
  rather than a shared occasion. That selects the content-sensitive side of the
  disagreement the measurement reported; it does not overturn the measurement,
  and no figure in that report was changed.

Every flagged scene carries a recorded verdict, so the cross-subset session
check is complete.

This bears on the structure noted in `docs/STAGE_SCOPE.md`: val was carved to
mirror test's scene mix, so both partitions hold videos from each of the three
evaluated scenes by design. That is what created the opportunity for a shared
occasion to straddle the boundary in each of them, and it is why the session
column is doing the entire load-bearing work of the separation. The check
above is what tests that column rather than assuming it.
