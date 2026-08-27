# ch4rerun conventions

## Numbers
- Never invent, estimate or round a measurement, a source, or a
  figure attributed to either. If a value is missing, say so.
- Choosing a parameter value is allowed when the rationale and its
  basis are stated in the same commit message or docstring.
- Every reported number must trace to a committed run artefact
  (CSV, log, or config). Cite the path.

## Claims
- Label judgement as judgement. Do not present a reasoned choice
  as an established result.
- Citations in prose are {CITE Author Year} placeholders only.

## Scripts
- Assertions, not checks. Exit non-zero with a diff on mismatch.
- Never reconcile a recomputed figure silently.
- Seeds set explicitly in torch, numpy and the DataLoader.

## Writing
- British academic English. No em-dashes. Fact first.

## Where the record lives
- Stage and item scope: docs/STAGE_SCOPE.md
- Ruled parameters: the `ruled` block of configs/ch4.yaml, read via src/ruled.py
- A committed report always beats the README.
