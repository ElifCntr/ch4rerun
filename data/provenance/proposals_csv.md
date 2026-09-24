# data/proposals/proposals.csv, provenance

Recorded 2026-09-24.

The stratifier input. Excluded from the repository at `.gitignore:31`
for its size, so a committed pipeline depends on a file the repository
does not carry. It is pinned by hash here.

| field | value |
| --- | --- |
| location | `data/proposals/proposals.csv` |
| size (bytes) | 751,435,447 |
| sha256 | `07506ed0da782a609776319425eb4217c26cb7961d7425da5bdf8bf9c1aa6ddd` |
| recorded | 2026-09-24 |
| committed | no, excluded at `.gitignore:31` |
| generator | `tools/extract_proposals.py` |
| generator metadata | `data/proposals/extraction_meta.json`, committed |

## What depends on it

- `tools/draw_negatives.py:60` reads it to draw the training negatives,
  producing `data/negatives/negatives_seed{1,2,3}.csv`,
  `data/negatives/negatives_meta.json` and `reports/negative_draw.txt`.
- `tools/record_band_edges.py` reads it to record
  `reports/stratification_band_edges.txt`, where this same hash is
  restated beside the derived edges.

## Checking a copy

```
sha256sum data/proposals/proposals.csv
```

A copy whose digest differs from the one above is not the file the
committed draw was made from, and the band edges in
`reports/stratification_band_edges.txt` do not describe it. Regenerating
the input is not guaranteed to reproduce this digest; the record of what
was actually used is the digest, not the regeneration recipe.
