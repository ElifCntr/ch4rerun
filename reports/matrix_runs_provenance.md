# Matrix run artefacts, what is committed and what is not

Written 12 September 2026.

The twelve pretrained matrix runs are recorded by their per-run `summary.json`
and `epochs.csv` under `runs/`, committed so that every figure in
`reports/matrix_results.txt` traces to a per-run artefact rather than to the
generated table alone. Those files are added against the `runs/` rule in
`.gitignore` by explicit override.

## Excluded binaries, recorded rather than committed

Checkpoints are excluded by the `*.pt` rule and score arrays by the `*.npz`
rule. The two final-pass score arrays are held on disk at:

| Path | Size | mtime |
| --- | --- | --- |
| `reports/val_final_scores.npz` | 38,444,935 bytes | 2026-08-26 22:22:36 +0100 |
| `reports/test_final_scores.npz` | 142,171,170 bytes | 2026-09-04 01:19:41 +0100 |

Twelve per-run val score arrays are also held on disk, one per run at
`runs/<tag>/val_scores.npz`, 4,661,251 to 4,872,329 bytes, mtimes 2026-08-24
04:00:06 to 2026-08-25 09:55:24 +0100.

Twelve per-run checkpoints are held at `runs/<tag>/best.pt`. None of the above
is in git, and none is reproducible without a re-run, which is closed.

## Console logs

The twelve per-run console logs under `runs/`, `log_2d_single_{1,2,3}.txt`,
`log_2d_tframe_logitavg_{1,2,3}.txt`, `log_tsm_{1,2,3}.txt` and
`log_r3d_18_{1,2,3}.txt`, together with `runs/queue.log`, remain uncommitted.
They were not within the scope of the 12 September ruling and no ruling covers
them.
