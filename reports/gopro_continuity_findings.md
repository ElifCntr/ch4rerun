# gopro continuity check, 12 August 2026

Ruled 11 Aug, to test whether `gopro_004` (test) and
`gopro_005` (val) are one continuous recording split across the partition
boundary.

## Result: inconclusive on the question asked

Creation timestamps did not survive the dataset packaging. Every one of the 77
files carries `encoder=Lavf57.56.101` and no `creation_time`, `gopro_003`
included, so the re-encode stripped them and the encoder signature
distinguishes nothing. The check cannot confirm continuity.

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
