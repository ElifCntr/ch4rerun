# Correction note, r3d_18 BatchNorm diagnostic

Dated 27 August 2026. This file is a sibling to `reports/r3d_bn_diagnostic.json`,
which admits no comments. It carries the same correction that was appended to
`reports/r3d_bn_diagnostic.txt`, and it records where the corrected values sit
and where they do not.

## The correction

Three subset AP values, 0.0084, 0.0051 and 0.0104, have been quoted as though
they were matrix outcomes. Each is a single epoch-1 measurement taken from an
`r3d_18` from-scratch run that was then terminated by `torch.OutOfMemoryError`
inside `loss.backward()` during epoch 2. None of the three completed a second
epoch, stopped on patience, or reached the epoch ceiling. They are not matrix
outcomes and no convergence claim in either direction is supported by them.

The same applies to the losses quoted alongside them as 0.576, 0.581 and 0.581.
The surviving records are preserved in `runs_terminated/`.

## Where the values appear, and where they do not

| File | Carries the three values | Status |
| --- | --- | --- |
| `reports/r3d_bn_diagnostic.json` | No | Unaffected. See below |
| `reports/r3d_bn_diagnostic.txt` | Yes | Corrected by appendix within that file |
| `tools/diagnose_r3d_bn.py`, docstring line 6 | Yes | Left unaltered. Corrected by this note |
| `tools/diagnose_r3d_bn.py`, report text line 194 | Yes | Left unaltered. See the re-entry warning below |

`reports/r3d_bn_diagnostic.json` carries only the diagnostic's own
measurements: the frozen, unfrozen and reset losses of 0.5774516964061729,
0.24913546887123816 and 0.24913546887123816, the BatchNorm layer count of 20,
and the first-layer activation statistics. None of the three corrected values is
present in it and none of its stored values requires correction.

## Re-entry warning

The sentence carrying the three values is hardcoded in the producing script at
`tools/diagnose_r3d_bn.py` line 194, inside the block that writes the report
text. Re-running the script therefore regenerates
`reports/r3d_bn_diagnostic.txt` with the uncorrected sentence restored and the
appended correction absent. The script has been left unaltered so that the
committed report and the script that produced it remain the same artefact. Any
re-run is to be treated as reintroducing the error until that line is ruled on.

## What is unaffected

The frozen against unfrozen against reset comparison is unaffected. Those three
losses were measured by `tools/diagnose_r3d_bn.py` in its own single-epoch run
and do not depend on the three corrected values.

The comparison figure quoted in the script docstring for `2d_single`, loss 0.328
and subset AP 0.617 at epoch 1, is also unaffected. It traces to
`runs_scratch/2d_single_seed1/epochs.csv`, which records 0.3281469940495543 and
0.6172454902683201 for a run that completed 12 epochs and stopped on patience.
