#!/usr/bin/env python3
"""
What does the upper area limit cost?

Ruled 19 August. The area limits derive from annotated BOX areas but are
applied to BLOB areas. On the lower limit the asymmetry is known and measured:
blobs are smaller than boxes, and a limit of 15 px2 costs 1.7 points against
9 px2.

On the UPPER limit it reverses. A blob merging a drone with adjacent motion can
EXCEED any annotated box, so a maximum set at the largest annotated box is
TIGHTER than intended and may reject legitimate covering blobs.

This counts them, offline, from the committed 1b-M dump, which recorded each
ground-truth box's best-covering blob area with NO size filter applied. A box
whose best-covering blob exceeds the per-resolution maximum is one the upper
limit removes, and if that box has no smaller alternative it is a recall loss.

The dump stores only the BEST covering blob per box, so this measures boxes
whose best cover is rejected. Where that happens, a smaller second-best blob
might still have covered the box, which this cannot see. So the figure is an
UPPER BOUND on the loss, and is labelled as one.

Usage:
    python tools/check_max_area_cost.py
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.ruled import ruled  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--dump", default="reports/bgs_pass_mog2/matched.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--coverage", default="0.5,0.7")
    args = ap.parse_args()

    covs = [float(c) for c in args.coverage.split(",")]
    cfg = load_config(args.config)
    max_px = ruled(cfg, "proposals.max_area_px")
    min_frac = ruled(cfg, "proposals.min_area_frac")

    if not os.path.exists(args.dump):
        print(f"FAIL missing {args.dump}")
        sys.exit(1)

    rows = []
    with open(args.dump, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["variant"] != "none":
                continue
            res = r["resolution"]
            w, h = (float(x) for x in res.split("x"))
            rows.append({
                "res": res, "scene": r["scene"], "fa": w * h,
                "cov": float(r["best_coverage"]),
                "area": float(r["matched_blob_area"]),
            })
    if not rows:
        print("FAIL no rows read")
        sys.exit(1)

    L = ["UPPER AREA LIMIT, WHAT IT COSTS",
         f"Source {args.dump}, {len(rows):,} ground-truth boxes, no size "
         "filter applied in the dump.",
         "",
         "A box is COUNTED as lost when its best-covering blob reaches the "
         "coverage threshold but EXCEEDS the per-resolution maximum, so the "
         "upper limit rejects it.",
         "",
         "UPPER BOUND, not an exact loss: the dump stores only the BEST "
         "covering blob per box, so a smaller second-best blob might still "
         "have covered a box counted here. Nothing in the committed dumps can "
         "see that.",
         ""]

    for res in sorted(max_px):
        L.append(f"  RESOLUTION {res}, maximum {float(max_px[res]):,.0f} px2")
        sub = [r for r in rows if r["res"] == res]
        if not sub:
            L.append("    no boxes at this resolution in the dump")
            continue
        over = [r for r in sub if r["area"] > float(max_px[res])]
        L.append(f"    boxes {len(sub):,}, best-covering blob over the "
                 f"maximum {len(over):,} ({len(over) / len(sub):.4%})")
        for c in covs:
            lost = [r for r in over if r["cov"] >= c]
            L.append(f"    at coverage >= {c:g}: {len(lost):,} lost "
                     f"({len(lost) / len(sub):.4%} of this resolution)")
        L.append("")

    total_over = sum(1 for r in rows if r["area"] > float(max_px[r["res"]]))
    L.append(f"POOLED: {total_over:,} of {len(rows):,} boxes "
             f"({total_over / len(rows):.4%}) have a best cover above the "
             "maximum.")
    for c in covs:
        lost = sum(1 for r in rows
                   if r["area"] > float(max_px[r["res"]]) and r["cov"] >= c)
        L.append(f"  at coverage >= {c:g}: {lost:,} "
                 f"({lost / len(rows):.4%} of all boxes)")
    L.append("")

    by_scene = defaultdict(lambda: [0, 0])
    for r in rows:
        by_scene[r["scene"]][1] += 1
        if r["area"] > float(max_px[r["res"]]) and r["cov"] >= covs[0]:
            by_scene[r["scene"]][0] += 1
    L.append(f"BY SCENE, at coverage >= {covs[0]:g}")
    for sc in sorted(by_scene):
        lost, tot = by_scene[sc]
        L.append(f"  {sc:<15} {lost:>6,} of {tot:>6,}  {lost / tot:.4%}")
    L.append("")

    L.append("READING. A negligible figure makes this a method-section "
             "footnote. A material one is a named recall loss, and reopening "
             "the upper limit becomes a decision with a measured basis, "
             "requiring re-extraction.")
    L.append(f"For reference the lower limit sits at {min_frac:g} of frame "
             "area and its cost is already measured at 1.7 points.")

    txt = "\n".join(L)
    path = os.path.join(args.out, "max_area_cost.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
