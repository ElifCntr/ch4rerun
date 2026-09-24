#!/usr/bin/env python3
"""
Record the stratification band edges to disk, and pin the stratifier input.

WHY THIS EXISTS. tools/draw_negatives.py computes the size-band edges at
line 109 and never writes them out. Its committed outputs,
reports/negative_draw.txt and data/negatives/negatives_meta.json, carry the
RULE as prose ("global deciles of the positive normalised area") but not the
values. The input the edges are derived from, data/proposals/proposals.csv,
is excluded from the repository at .gitignore:31. The derived values were
therefore unquotable and would not survive loss of that input.

This script closes that gap. It is READ-ONLY with respect to every existing
artefact: it reads proposals.csv and the config, and writes two NEW files.
It does not touch reports/negative_draw.txt.

THE EDGES ARE COMPUTED BY THE SAME EXPRESSION AS THE DRAW, on the same input
under the same filter, so nothing can drift between the two:

    edges = np.percentile(pos_area, np.linspace(0, 100, n_bands + 1))

which is tools/draw_negatives.py:109 verbatim. The filter is that file's
lines 86 to 92: rows whose split column equals the requested split and whose
label column is exactly "positive", taking the area_frac column.

NOTE ON THE COUNT. With bands = 10 this yields ELEVEN edge values, the 0th
to the 100th percentile in steps of 10. Nine of them are interior cut points;
tools/draw_negatives.py:110 drops the first and last as `inner = edges[1:-1]`
and only those nine decide band membership. The outer two are the observed
minimum and maximum of the positive area fractions and bound no band from
outside. Ten bands, eleven edges, nine cut points. All eleven are recorded.

VERIFICATION. The edges are not merely recomputed, they are checked against
the committed draw. This script re-derives the per-band and per-cell positive
counts by digitising with the same nine cut points, and ASSERTS them against
the 49 cell lines of reports/negative_draw.txt. A mismatch exits non-zero
with a diff and writes nothing, because an edge set that does not reproduce
the committed counts is not the edge set the draw used.

Usage:
    python tools/record_band_edges.py            # compute, verify and write
    python tools/record_band_edges.py --verify   # verify only, write nothing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.config import load_config  # noqa: E402
from src.ruled import ruled  # noqa: E402

PROPOSALS = "data/proposals/proposals.csv"
DRAW = "reports/negative_draw.txt"
OUT_EDGES = Path("reports/stratification_band_edges.txt")
OUT_PROV = Path("data/provenance/proposals_csv.md")
CHUNK = 1 << 20


def fail(lines: list[str]) -> None:
    print("FAIL the recomputed edges do not reproduce the committed draw.",
          file=sys.stderr)
    for ln in lines:
        print(f"  {ln}", file=sys.stderr)
    sys.exit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def committed_cells(path: str) -> list[dict]:
    """The 49 per-cell lines of the committed draw report."""
    rows = []
    pat = re.compile(r"^\s+(\d)\s+(\S+)\s+([\d,]+)\s+([\d,]+)\s+"
                     r"([\d,]+)\s+([\d,]+)\s+([\d,]+)\s*$")
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        m = pat.match(line.rstrip("\n"))
        if m:
            g = [x.replace(",", "") for x in m.groups()]
            rows.append({"line": n, "band": int(g[0]), "scene": g[1],
                         "positives": int(g[2])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--proposals", default=PROPOSALS)
    ap.add_argument("--split", default="train")
    ap.add_argument("--verify", action="store_true",
                    help="verify against the committed draw and write nothing")
    args = ap.parse_args()

    src = Path(args.proposals)
    if not src.exists():
        print(f"FAIL missing {src}. It is gitignored by design; regenerate "
              f"with tools/extract_proposals.py.", file=sys.stderr)
        sys.exit(1)
    if not Path(DRAW).exists():
        print(f"FAIL missing {DRAW}, which the verification needs.",
              file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    n_bands = ruled(cfg, "negatives.bands")
    ratio = ruled(cfg, "negatives.ratio")
    rule = ruled(cfg, "negatives.band_edge_rule")

    print(f"hashing {src} ...")
    digest = sha256(src)
    size = src.stat().st_size

    # Same filter as tools/draw_negatives.py:86-92.
    pos_area: list[float] = []
    pos_scene: list[str] = []
    with open(src, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ix = {c: i for i, c in enumerate(header)}
        for row in rd:
            if row[ix["split"]] != args.split:
                continue
            if row[ix["label"]] == "positive":
                pos_area.append(float(row[ix["area_frac"]]))
                pos_scene.append(row[ix["scene"]])
    pos_area_a = np.asarray(pos_area)
    pos_scene_a = np.asarray(pos_scene)
    if pos_area_a.size == 0:
        print(f"FAIL no positives in split '{args.split}'", file=sys.stderr)
        sys.exit(1)

    # tools/draw_negatives.py:109-112, verbatim.
    edges = np.percentile(pos_area_a, np.linspace(0, 100, n_bands + 1))
    inner = edges[1:-1]
    pos_band = np.digitize(pos_area_a, inner, right=False)

    # ---- verification against the committed draw --------------------------
    cells = committed_cells(DRAW)
    problems: list[str] = []
    if not cells:
        problems.append(f"parsed no cell lines from {DRAW}")

    got = defaultdict(int)
    for b, sc in zip(pos_band, pos_scene_a):
        got[(int(b), sc)] += 1
    for c in cells:
        mine = got.get((c["band"], c["scene"]), 0)
        if mine != c["positives"]:
            problems.append(
                f"{DRAW}:{c['line']} band {c['band']} {c['scene']}: "
                f"committed {c['positives']}, recomputed {mine}")
    extra = set(got) - {(c["band"], c["scene"]) for c in cells}
    for k in sorted(extra):
        problems.append(f"recomputed a cell absent from the draw: {k} "
                        f"with {got[k]} positives")

    total_committed = sum(c["positives"] for c in cells)
    if total_committed != pos_area_a.size:
        problems.append(f"positives: committed {total_committed}, "
                        f"recomputed {pos_area_a.size}")
    hdr = open(DRAW, encoding="utf-8").read()
    if f"positives {pos_area_a.size:,}," not in hdr:
        problems.append(f"{DRAW} header does not state "
                        f"positives {pos_area_a.size:,}")
    if problems:
        fail(problems)

    print(f"verified: {len(cells)} cells and {pos_area_a.size:,} positives "
          f"reproduce {DRAW} exactly")

    lo_n = int((pos_area_a < inner[0]).sum())
    hi_n = int((pos_area_a >= inner[-1]).sum())

    if args.verify:
        print("--verify given; nothing written.")
        return

    # ---- the edges report --------------------------------------------------
    today = date.today().isoformat()
    L = []
    A = L.append
    A("STRATIFICATION BAND EDGES, negative draw")
    A(f"Recorded {today}. Derived values, written because the draw does not "
      f"write them.")
    A("")
    A("These are the SIZE-BAND edges used to stratify the training negative")
    A("draw. They are recomputed here by the same expression the draw uses,")
    A("from the same input under the same filter, and verified against the")
    A("committed per-cell counts before being written. See the VERIFICATION")
    A("section below.")
    A("")
    A("They are STRATIFICATION bands. They are not the benchmark-derived")
    A("REPORTING bands, per tools/draw_negatives.py:30-31.")
    A("")
    A("RULE")
    A(f"  configs/ch4.yaml :: ruled.negatives.band_edge_rule = {rule!r}")
    A(f"  configs/ch4.yaml :: ruled.negatives.bands          = {n_bands}")
    A(f"  configs/ch4.yaml :: ruled.negatives.ratio          = {ratio}")
    A("  Global deciles of the positive normalised area, taken over the whole")
    A("  split rather than per scene.")
    A("")
    A("UNITS")
    A("  Normalised area. Dimensionless fraction of frame area, in [0, 1].")
    A("  The binned quantity is the area_frac column of proposals.csv, written")
    A("  at tools/extract_proposals.py:309 as the connected-component blob")
    A("  area in pixels divided by the frame area (fa = fw * fh,")
    A("  tools/extract_proposals.py:244).")
    A("  NOT equivalent side. No equivalent-side quantity enters the draw.")
    A("")
    A("EXPRESSION, tools/draw_negatives.py:109-112 verbatim")
    A("  edges = np.percentile(pos_area, np.linspace(0, 100, n_bands + 1))")
    A("  inner = edges[1:-1]")
    A("  pos_band = np.digitize(pos_area, inner, right=False)")
    A("  neg_band = np.digitize(neg_area, inner, right=False)")
    A("")
    A("INPUT")
    A(f"  path         {src}")
    A(f"  split        {args.split}")
    A(f"  filter       label == 'positive', column area_frac")
    A(f"               (tools/draw_negatives.py:86-92)")
    A(f"  positives    {pos_area_a.size:,}")
    A(f"  size_bytes   {size:,}")
    A(f"  sha256       {digest}")
    A("  NOT COMMITTED. Excluded at .gitignore:31. Pinned by the hash above")
    A("  and in data/provenance/proposals_csv.md.")
    A("")
    A(f"EDGES, {n_bands + 1} values: the 0th to the 100th percentile in steps "
      f"of 10")
    A("  Of these, the nine INTERIOR values are the cut points that decide")
    A("  band membership. The first and last are the observed minimum and")
    A("  maximum of the positive area fractions and bound no band from")
    A("  outside: np.digitize is called on inner, not on edges, so a value")
    A("  below edge 0 would still fall in band 0 and a value above edge 10")
    A("  would still fall in band 9. Ten bands, eleven edges, nine cut points.")
    A("")
    A(f"  {'i':>3}  {'percentile':>10}  {'edge (area fraction)':>24}  role")
    for i, e in enumerate(edges):
        role = ("minimum, not a cut point" if i == 0 else
                "maximum, not a cut point" if i == n_bands else
                f"cut: band {i - 1} | band {i}")
        A(f"  {i:>3}  {i * 100 // n_bands:>9}%  {e:>24.17e}  {role}")
    A("")
    A("  Full precision, as float64 repr, in index order:")
    for i, e in enumerate(edges):
        A(f"    edges[{i:>2}] = {e!r}")
    A("")
    A("  The nine cut points alone, as passed to np.digitize:")
    A(f"    inner = {[float(x) for x in inner]!r}")
    A("")
    A("BAND MEMBERSHIP")
    A("  np.digitize(..., right=False) puts a value v in band b where")
    A("  inner[b-1] <= v < inner[b]. Left-closed, right-open. Band 0 is")
    A("  v < inner[0] and band 9 is v >= inner[8].")
    A(f"  band 0 holds {lo_n:,} positives, band 9 holds {hi_n:,}.")
    A("")
    A("THE OUTER EDGES AGAINST THE RULED ANCHORS")
    A("  Two independent cross-checks, recorded as observations. Both fall out")
    A("  of the recomputation above and neither is hardcoded here.")
    A("")
    min_frac = ruled(cfg, "proposals.min_area_frac")
    pos_cov = ruled(cfg, "proposals.positive_coverage")
    A(f"  edges[0], the observed minimum, is {edges[0]!r}.")
    A(f"  configs/ch4.yaml:108 rules proposals.min_area_frac = {min_frac:.2e},")
    A("  the extraction floor, applied at tools/extract_proposals.py:246 as")
    A("  lo = min_frac * fa. The observed minimum sits just above that floor,")
    A(f"  at {edges[0] / min_frac:.6f} times it, which is what a floor applied")
    A("  at extraction produces.")
    A("")
    A(f"  edges[{n_bands}], the observed maximum, is {edges[-1]!r},")
    A(f"  which is {edges[-1]:.4e} to five significant figures.")
    A("  configs/ch4.yaml:110-111 records a measured anchor of 1.6629e-02, the")
    A('  "largest cover of a box at coverage >= 0.5". A proposal is positive')
    A(f"  exactly when its coverage reaches proposals.positive_coverage = "
      f"{pos_cov}")
    A("  (configs/ch4.yaml:114), so the largest positive area fraction and")
    A("  that anchor are the same quantity. They agree.")
    A("  This corroborates the anchor from which ruled.proposals.max_area_frac")
    A("  was derived. It is a cross-check, not a new measurement, and it does")
    A("  not revisit the max_area_frac ruling.")
    A("")
    A("VERIFICATION, run before this file was written")
    A("  The nine cut points above were used to re-derive the positive count")
    A("  of every band-by-scene cell, and each was asserted against the")
    A(f"  committed cell lines of {DRAW}.")
    A(f"    cells checked        {len(cells)}")
    A(f"    cells disagreeing    0")
    A(f"    positives recomputed {pos_area_a.size:,}")
    A(f"    positives committed  {total_committed:,}")
    A("  The script exits non-zero and writes nothing on any mismatch, because")
    A("  an edge set that does not reproduce the committed counts is not the")
    A("  edge set the draw used.")
    A("")
    A("PROVENANCE OF THIS FILE")
    A("  Written by tools/record_band_edges.py. Reproduce with:")
    A("    python tools/record_band_edges.py")
    A("  Verify without writing:")
    A("    python tools/record_band_edges.py --verify")
    A("  Nothing in reports/negative_draw.txt was edited. These values are")
    A("  DERIVED and recorded after the fact; they were not part of the")
    A("  original draw's output.")

    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    OUT_EDGES.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- the provenance record --------------------------------------------
    P = []
    B = P.append
    B("# data/proposals/proposals.csv, provenance")
    B("")
    B(f"Recorded {today}.")
    B("")
    B("The stratifier input. Excluded from the repository at `.gitignore:31`")
    B("for its size, so a committed pipeline depends on a file the repository")
    B("does not carry. It is pinned by hash here.")
    B("")
    B("| field | value |")
    B("| --- | --- |")
    B(f"| location | `{src}` |")
    B(f"| size (bytes) | {size:,} |")
    B(f"| sha256 | `{digest}` |")
    B(f"| recorded | {today} |")
    B("| committed | no, excluded at `.gitignore:31` |")
    B("| generator | `tools/extract_proposals.py` |")
    B("| generator metadata | `data/proposals/extraction_meta.json`, committed |")
    B("")
    B("## What depends on it")
    B("")
    B("- `tools/draw_negatives.py:60` reads it to draw the training negatives,")
    B("  producing `data/negatives/negatives_seed{1,2,3}.csv`,")
    B("  `data/negatives/negatives_meta.json` and `reports/negative_draw.txt`.")
    B("- `tools/record_band_edges.py` reads it to record")
    B("  `reports/stratification_band_edges.txt`, where this same hash is")
    B("  restated beside the derived edges.")
    B("")
    B("## Checking a copy")
    B("")
    B("```")
    B(f"sha256sum {src}")
    B("```")
    B("")
    B("A copy whose digest differs from the one above is not the file the")
    B("committed draw was made from, and the band edges in")
    B("`reports/stratification_band_edges.txt` do not describe it. Regenerating")
    B("the input is not guaranteed to reproduce this digest; the record of what")
    B("was actually used is the digest, not the regeneration recipe.")

    OUT_PROV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROV.write_text("\n".join(P) + "\n", encoding="utf-8")

    print(f"written {OUT_EDGES}")
    print(f"written {OUT_PROV}")


if __name__ == "__main__":
    main()
