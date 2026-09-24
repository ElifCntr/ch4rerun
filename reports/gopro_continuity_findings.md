# gopro continuity check, 12 August 2026

Ruled 11 Aug, to test whether `gopro_004` (test) and
`gopro_005` (val) are one continuous recording split across the partition
boundary.

## Result: inconclusive on the question asked

Creation timestamps did not survive the dataset packaging. Every one of the 77
files carries `encoder=Lavf57.56.101` and no `creation_time`, `gopro_003`
included, so the re-encode stripped them and the encoder signature
distinguishes nothing. The check cannot confirm continuity.

> **The encoder half of the sentence above is wrong.** Corrected 24 September
> 2026; see "Correction, 24 September 2026" at the foot of this file. The
> `creation_time` half holds, and the conclusion of this section is unchanged.

It did not falsify it either. `gopro_004` and `gopro_005` are both 1920x1080
at 29.97 fps, so nothing rules them out.

## Separate finding, which bears on the same question

Frame rates across the nine fisheye clips:

| video | fps | duration (s) | v1.0 session |
| --- | --- | --- | --- |
| gopro_000 | 25.0 | 19.240 | gopro_treesite |
| gopro_001 | 25.0 | 19.240 | gopro_treesite |
| gopro_002 | 29.970 | 10.644 | gopro_treesite |
| gopro_003 | 29.971 | 18.519 | gopro_treesite |
| gopro_004 | 29.970 | 25.058 | (alone, test) |
| gopro_005 | 29.970 | 27.060 | (alone, val) |
| gopro_006 | 29.971 | 45.078 | gopro_path |
| gopro_007 | 29.971 | 116.147 | gopro_path |
| gopro_008 | 29.970 | 33.066 | gopro_path |

`gopro_000` and `gopro_001` run at 25 fps while `gopro_002` and `gopro_003`
run at 29.97, yet the v1.0 manifest assigns all four to one session,
`gopro_treesite`. Four clips at two different frame rates cannot be one
continuous recording, so the manifest's session labels were not derived from
recording metadata.

`gopro_000` and `gopro_001` also share a duration to the millisecond, 19.240,
which is the pattern produced by cutting one recording into equal chunks.

## Why this matters

The zero-session-overlap assertion in `build_split.py` treats the manifest's
session column as authoritative. That column is now shown to group
inconsistently in at least one place. The direction that would hurt Chapter 4
is the opposite one: a single real recording labelled as two sessions,
`gopro_004` and `gopro_005`, sitting either side of the val/test boundary.
Nothing here shows that has happened, and nothing here rules it out.

## What was not done

No frames were decoded and no annotation file was read. A frame comparison
across the `004`/`005` join, or a check on whether the annotation track
continues across it, would be a real test-set access and needs its own ruling.

## Correction, 24 September 2026

The sentence under "Result" claims that every one of the 77 files carries
`encoder=Lavf57.56.101` and no `creation_time`. The encoder half is wrong.

Measured over the 37 videos of the static subset, on 24 September 2026, by
`tools/investigate_seaside_mpg.py`, output at
`reports/seaside_mpg/run.log` and `reports/seaside_mpg/fleet_containers.csv`:

| encoder tag | files |
| --- | ---: |
| `Lavf57.56.101` | 28 |
| `Lavf58.20.100` | 8, all industrial |
| none | 1, `00_02_45_to_00_03_10_cut.mpg` |

No `creation_time` survives on any of the 37.

Scope of this correction. It measures the 37 static videos, not all 77. The
original claim was about 77 and the check behind it read 9, `gopro_000` to
`gopro_008`, as `reports/gopro_continuity.csv` records. The claim therefore
went past its evidence in both directions: it was never measured across 77,
and it is false across the 37 that have now been measured. The 40 dynamic
videos remain unmeasured on this point.

What is unaffected.

- Every timestamp conclusion stands. No `creation_time` survives on any of
  the 37, which is the half of the sentence the continuity argument rests on.
- The conclusion of this file is unchanged. The check remains inconclusive on
  whether `gopro_004` and `gopro_005` are one recording, and the encoder
  signature still distinguishes nothing, since a tag shared by 28 files
  separates none of them.
- The fisheye conclusions are unaffected. All nine `gopro_*` files do carry
  `Lavf57.56.101` and no `creation_time`, shown directly at
  `reports/fisheye_session/ffprobe_evidence.txt`, so the nine-video basis of
  that work was always sound.

The same overstated sentence was repeated in
`reports/fisheye_session/FINDINGS.md` section 2 and is corrected there too.
Neither original sentence has been rewritten.
