# Terminated r3d_18 scratch attempts

Sole surviving records of three `r3d_18` from-scratch attempts terminated by
CUDA out of memory after one completed epoch each, batch 162, under code
predating `008b7bf`.

Preserved 27 August 2026, before the re-run queue overwrote the originals in
`runs_scratch/`. The seed 1 `epochs.csv` and `best.pt` had already been
overwritten when the copy was taken and are not recoverable; the seed 1 console
log is all that survives of that attempt.

## What is here

| Path | What it is |
| --- | --- |
| `log_r3d_18_1.txt` | Seed 1 console log. Epoch 1, then the traceback |
| `log_r3d_18_2.txt` | Seed 2 console log. Epoch 1, then the traceback |
| `log_r3d_18_3.txt` | Seed 3 console log. Epoch 1, then the traceback |
| `r3d_18_seed2/epochs.csv` | Seed 2, one data row |
| `r3d_18_seed3/epochs.csv` | Seed 3, one data row |
| `r3d_18_seed2/best.pt` | Seed 2 epoch-1 checkpoint, 132,742,788 bytes |
| `r3d_18_seed3/best.pt` | Seed 3 epoch-1 checkpoint, 132,742,788 bytes |

The two checkpoints are held on disk and are NOT committed, `*.pt` being
excluded repository-wide by `.gitignore`. Everything else here is committed.

## What the logs record

Each run completed epoch 1 and then raised `torch.OutOfMemoryError` inside
`loss.backward()` during epoch 2. None stopped on patience and none reached the
epoch ceiling. The allocation refused was 994.00 MiB against a total card
capacity of 15.88 GiB in all three.

| Seed | Epochs completed | Epoch 1 loss | Epoch 1 subset AP | Ended |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.5760 | 0.0084 | CUDA OOM in epoch 2 |
| 2 | 1 | 0.5809 | 0.0051 | CUDA OOM in epoch 2 |
| 3 | 1 | 0.5814 | 0.0104 | CUDA OOM in epoch 2 |

These are single epoch-1 measurements from terminated runs. They are not
convergence outcomes and no convergence claim is supported by them.

## Undetermined

Batch 162 raised CUDA out of memory before 23 August 2026 and completed without
it on 24 to 27 August 2026 for the same arm at the same batch. The artefacts do
not determine why.

## Code resolution

The runs are bounded before 2026-08-23 20:04:54, the mtime carried by every
preserved file. `tools/train_arm.py` was then at `a8c243c` and `src/tubelets.py`
was then at `a8c243c`, which carries the constant horizontal flip later
corrected by `008b7bf`.
