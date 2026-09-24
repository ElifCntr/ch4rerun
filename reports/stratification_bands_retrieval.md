# Stratification band retrieval, negative draw

24 September 2026. Read-only retrieval from the committed negative-draw config
and the draw tool's committed output. Quotes and locations only; no
interpretation is offered and no verdict is drawn.

Section 1 was superseded in part later the same day; see the note at its head.
The rest of this retrieval stands as written.

Sources, all committed:

| artefact | role |
| --- | --- |
| `configs/ch4.yaml` | the ruled block, band count and edge rule |
| `tools/draw_negatives.py` | the draw tool |
| `reports/negative_draw.txt` | its text output, the only per-cell record |
| `data/negatives/negatives_meta.json` | its machine-readable output |

## 1. The band edges were not in any committed artefact

SUPERSEDED IN PART, 24 September 2026, on the same day and by the ruling this
retrieval prompted. The derived edge values are now recorded at
`reports/stratification_band_edges.txt`, written by
`tools/record_band_edges.py`, together with the rule, the units, and the
sha256 of the input they were derived from. Everything in this section
remains true of `tools/draw_negatives.py` itself, which still does not write
them; what has changed is that a separate dated report now does. The ten
bands, eleven edges and nine cut points are quoted in that report.

The edge values are computed at runtime by the draw and never written to disk
by it.

- `tools/draw_negatives.py:109` `edges = np.percentile(pos_area, np.linspace(0, 100, n_bands + 1))`
- `tools/draw_negatives.py:110` `inner = edges[1:-1]`

Lines 111 and 112 consume `inner` through `np.digitize`. Those four lines, 109
to 112, are every mention of `edges` and `inner` in the file.

Both committed outputs carry the RULE as prose and not the values:

- `reports/negative_draw.txt:5` "Band edges are global deciles of the positive
  normalised area."
- `data/negatives/negatives_meta.json`, key `band_edge_rule`: "global deciles
  of the positive normalised area", written at `tools/draw_negatives.py:204`.

The meta JSON carries fifteen top-level keys: `split`, `ratio`, `bands`,
`stratified_by`, `band_edge_rule`, `seeds`, `positives`, `coverage_zero_pool`,
`quota_total`, `drawn_total`, `shortfall_total`, `cells`, `per_seed_drawn`,
`union_across_seeds`, `note`. The only key matching "edge" is
`band_edge_rule`, which is a string.

The edges were also not recomputable from committed data. The tool's input is
`data/proposals/proposals.csv` (`tools/draw_negatives.py:60`), which is
excluded from the repository at `.gitignore:31`. That input is now pinned by
sha256 at `data/provenance/proposals_csv.md`, so a future copy can be checked
against the one the committed draw was made from, though the input itself
remains uncommitted.

## 2. Units, and the quantity the bands are defined on

Defined on NORMALISED AREA, a dimensionless fraction of frame area. Not
equivalent side.

- `tools/draw_negatives.py:91` and `:94` bin the `area_frac` column of
  `proposals.csv`, for positives and negatives respectively.
- `tools/extract_proposals.py:309` writes that column as
  `round(int(st[k, cv2.CC_STAT_AREA]) / fa, 9)`, the connected-component blob
  area in pixels over the frame area.
- `tools/extract_proposals.py:244` `fa = float(fw * fh_)`.

No equivalent-side quantity enters the draw. `equiv_side` and
`equiv_side_frac` exist only as separate columns of
`reports/instances_boxes.csv`, fields 20 and 21 of its header, computed at
`tools/measure_instances.py:213` and `:214`.

Note that `area_frac` in `reports/instances_boxes.csv` is a different
construction, `w * h / frame_area` at `tools/measure_instances.py:212`. The
draw does not read that file. Both are normalised areas.

## 3. Number of bands, and the ruled parameters

From `configs/ch4.yaml`, the `ruled` block:

- `configs/ch4.yaml:120` `ratio: 2`
- `configs/ch4.yaml:121` `bands: 10`
- `configs/ch4.yaml:122` `band_edge_rule: "global_deciles_of_positive_area_frac"`
- `configs/ch4.yaml:123` `stratify_by: ["size_band", "scene"]`

Read at `tools/draw_negatives.py:67` and `:68`. Confirmed in the output at
`reports/negative_draw.txt:2`:

    split train, ratio 2x, 10 bands, 5 scenes, seeds [1, 2, 3]

## 4. Per band: positives, quota, drawn, shortfall

### How the artefacts hold these figures

Per band and scene CELL, POOLED ACROSS SEEDS, not per seed. The drawn count
per cell is `drawable = min(q, idx.size)` at `tools/draw_negatives.py:136`,
which does not depend on the seed. The seeds differ in WHICH rows are drawn,
at `tools/draw_negatives.py:157`, not in how many. Per-seed figures appear
only as three identical grand totals.

The quota rule is `q = npos * ratio` at `tools/draw_negatives.py:135`.

All 49 cells are at `reports/negative_draw.txt:8-56`. The table below
aggregates those lines by band. Every figure in it is a sum over the committed
per-cell lines cited in its final column.

| band | cells | positives | quota | drawn | shortfall | lines |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 5 | 1,334 | 2,668 | 2,668 | 0 | 8-12 |
| 1 | 5 | 1,260 | 2,520 | 2,520 | 0 | 13-17 |
| 2 | 5 | 1,408 | 2,816 | 2,816 | 0 | 18-22 |
| 3 | 5 | 1,332 | 2,664 | 2,664 | 0 | 23-27 |
| 4 | 5 | 1,334 | 2,668 | 2,668 | 0 | 28-32 |
| 5 | 5 | 1,332 | 2,664 | 2,664 | 0 | 33-37 |
| 6 | 5 | 1,338 | 2,676 | 2,671 | 5 | 38-42 |
| 7 | 5 | 1,339 | 2,678 | 2,368 | 310 | 43-47 |
| 8 | 5 | 1,334 | 2,668 | 1,786 | 882 | 48-52 |
| 9 | 4 | 1,336 | 2,672 | 1,331 | 1,341 | 53-56 |
| **total** | **49** | **13,347** | **26,694** | **24,156** | **2,538** | |

Band 9 holds four cells rather than five. `lake` is absent from it; its cells
are `gopro_fisheye`, `industrial`, `meadow` and `seaside` at
`reports/negative_draw.txt:53-56`. That is why the cell count is 49 and not
50, matching `cells` in the meta JSON, which holds 49 entries. Cells with zero
positives are skipped at `tools/draw_negatives.py:132-133`.

### Cells carrying a shortfall

Seven of the 49, in four bands, 6 to 9.

| line | band | scene | quota | drawn | shortfall |
| --- | ---: | --- | ---: | ---: | ---: |
| `negative_draw.txt:42` | 6 | seaside | 590 | 585 | 5 |
| `negative_draw.txt:46` | 7 | meadow | 624 | 497 | 127 |
| `negative_draw.txt:47` | 7 | seaside | 656 | 473 | 183 |
| `negative_draw.txt:51` | 8 | meadow | 646 | 276 | 370 |
| `negative_draw.txt:52` | 8 | seaside | 760 | 248 | 512 |
| `negative_draw.txt:55` | 9 | meadow | 294 | 61 | 233 |
| `negative_draw.txt:56` | 9 | seaside | 1,200 | 92 | 1,108 |

The remaining 42 cells carry a shortfall of zero.

### Totals

`reports/negative_draw.txt:58`, verbatim:

    TOTAL quota 26,694, drawable 24,156, shortfall 2,538

Matching the meta JSON: `quota_total: 26694`, `drawn_total: 24156`,
`shortfall_total: 2538`.

`reports/negative_draw.txt:59`, verbatim:

    SHORTFALL RECORDED AND NOT BACKFILLED. A deficit is not redistributed to
    other cells, because that would undo the profile matching the
    stratification exists to produce.

Written at `tools/draw_negatives.py:148-151`.

### Per seed

`reports/negative_draw.txt:61-63`, verbatim:

    seed 1: 24,156 negatives drawn
    seed 2: 24,156 negatives drawn
    seed 3: 24,156 negatives drawn

Meta JSON: `per_seed_drawn: {"1": 24156, "2": 24156, "3": 24156}`,
`union_across_seeds: 63335`.

### Population

`reports/negative_draw.txt:3`, verbatim:

    positives 13,347, coverage-0 pool 4,513,722

### Where drawn and shortfall live

The meta JSON's `cells` entries carry only `band`, `scene`, `positives`,
`quota` and `available`, written at `tools/draw_negatives.py:210-212`. The
per-cell `drawn` and `short` figures exist ONLY in
`reports/negative_draw.txt`, emitted at `tools/draw_negatives.py:142-143`.

## 5. A distinction the source draws

`tools/draw_negatives.py:30-31`, verbatim:

    These are STRATIFICATION bands and are not the benchmark-derived REPORTING
    bands.

A separate `band_edges` parameter is taken by `src/metrics.py:142`,
`size_band_recall(labels, area_fracs, band_edges, n_gt_by_band)`, and applied
at `src/metrics.py:151`. That parameter is not the object retrieved above.
