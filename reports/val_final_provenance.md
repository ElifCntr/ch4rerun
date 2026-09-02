# Provenance, val final pass

Dated 30 August 2026. Written alongside `reports/val_final_results.json`, which
is committed, and `runs/log_val_rehearsal.txt`, which is committed by explicit
override of the `runs/` rule in `.gitignore`.

## The scores array is not committed

`reports/val_final_scores.npz` is excluded by the binary rule in `.gitignore`
(`*.npz`). It is held at that path in the working tree:

- path: `reports/val_final_scores.npz`
- mtime: 2026-08-26 22:22:36.201073464 +0100
- size: 38,444,935 bytes
- contents: 17 arrays, one score vector per run tag, plus `labels`, `scenes`,
  `gt_key`, `area_frac` and `videos`, each of length 712,925

It regenerates from the committed checkpoints and proposals by re-running the
pass. The operating thresholds in `val_final_results.json` were selected from
the per-run `val_scores.npz` files written at training time, not from this
array.

## What the pass was

Twelve checkpoints scored in one decode pass over the six val videos on
26 August 2026, finishing 22:22:36. Every run reports 712,208 scored and 717
ignored, which sums to the full val proposal count of 712,925, so the pass is
uncapped and complete rather than a subset. Wall clock 35,099.4 s, that is
585.0 min.

## Inference batch

The pass was run at inference batch 256. This was determined by flush
arithmetic rather than by file timestamp: a flush is emitted only when the
pending list reaches the batch size, so each per-video count in
`runs/log_val_rehearsal.txt` is the cumulative total truncated to a multiple of
the batch. Reconstructing the pushed counts from `data/proposals/proposals.csv`
and the window arithmetic predicts 77,312, 294,656 and 712,704 at batch 256 and
77,504, 294,848 and 712,896 at batch 64. The logged counts are the former on
all three. The batch-64 prediction is corroborated by the training-time
streaming logs, which pass batch 64 and print exactly those figures, for
instance in `runs_scratch/log_tsm_1.txt`.

The file timestamp could not be used because `tools/evaluate_final.py` was
patched and reverted on 30 August, which overwrote its mtime.

Per-proposal cost was measured flat between the two settings, 33,151 us at
batch 64 against 32,951 us at batch 256, so the batch size is a grouping choice
and not a result-bearing one. The twelve APs from this pass agree with the
training-time full-val APs, which were computed at batch 64 through
`src/streaming_eval.py`, to a maximum absolute difference of 2.469e-07.

`tools/evaluate_final.py` now carries batch 256 as its committed default, so
the test pass runs the byte-identical rehearsed path.
