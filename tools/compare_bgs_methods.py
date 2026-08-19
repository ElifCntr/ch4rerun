#!/usr/bin/env python3
"""
Three-method comparison, and the union test that decides what the ceiling means.

THE QUESTION THIS EXISTS TO ANSWER. Each method has its own recall ceiling.
That alone does not distinguish two very different claims:

  (a) MOG2 has a ceiling, and another method might not.
  (b) MOTION PROPOSALS have a ceiling at this target scale.

Per-method curves cannot separate them. If the three methods miss DIFFERENT
boxes, the union recall is much higher than any single method and the ceiling
is method-specific, which is (a). If they miss the SAME boxes, the union is
close to the best single method and the ceiling is intrinsic, which is (b).

That union figure is the whole reason the ViBe and KNN passes were restored,
so it is computed first and reported first.

Also reports, per method: recall against the normalised lower limit, blobs per
frame, the coverage distribution, and per-scene recall at the ruled limit.

Reads reports/bgs_pass_<method>/ dumps. Nothing is re-run.

Usage:
    python tools/compare_bgs_methods.py
    python tools/compare_bgs_methods.py --limit 7.23e-06 --coverage 0.5
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

METHODS = ["mog2", "vibe", "knn"]
FRAC_GRID = [0.0, 2e-6, 5e-6, 7.23e-6, 1.2e-5, 2e-5, 3.5e-5, 6e-5, 1e-4]


def frame_area(res):
    w, h = res.split("x")
    return float(w) * float(h)


def load(dump):
    """Returns (by_key, hist, frames). by_key maps a ground-truth box to its
    coverage and matched blob area, so methods can be joined box by box."""
    m = os.path.join(dump, "matched.csv")
    h = os.path.join(dump, "hist.csv")
    if not (os.path.exists(m) and os.path.exists(h)):
        return None, None, None
    by_key = {}
    with open(m, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["video"], int(r["frame"]),
                   r["gt_x"], r["gt_y"], r["gt_w"], r["gt_h"])
            by_key[key] = {
                "scene": r["scene"], "res": r["resolution"],
                "fa": frame_area(r["resolution"]),
                "cov": float(r["best_coverage"]),
                "area": float(r["matched_blob_area"]),
            }
    hist, frames = [], {}
    with open(h, newline="") as fh:
        for r in csv.DictReader(fh):
            hist.append((float(r["bin_lo"]), int(r["count"]),
                         frame_area(r["resolution"])))
            frames[r["video"]] = int(r["frames_processed"])
    return by_key, hist, frames


def found(rec, frac, cov_thr):
    return rec["cov"] >= cov_thr and rec["area"] >= frac * rec["fa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports")
    ap.add_argument("--limit", type=float, default=7.23e-6,
                    help="the ruled normalised lower limit")
    ap.add_argument("--coverage", default="0.5,0.7")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    covs = [float(c) for c in args.coverage.split(",")]

    data = {}
    for m in METHODS:
        by_key, hist, frames = load(os.path.join(args.root, f"bgs_pass_{m}"))
        if by_key is None:
            print(f"NOTE no dump for {m}, skipping")
            continue
        data[m] = {"k": by_key, "h": hist, "f": frames}
    if len(data) < 2:
        print("FAIL need at least two method dumps to compare")
        sys.exit(1)

    present = [m for m in METHODS if m in data]
    keys = set.intersection(*[set(data[m]["k"]) for m in present])
    L = ["THREE-METHOD COMPARISON",
         f"Methods present: {present}",
         f"Boxes common to all: {len(keys):,}",
         f"Ruled normalised lower limit: {args.limit:g}",
         ""]

    for m in present:
        n = len(data[m]["k"])
        if n != len(keys):
            L.append(f"NOTE {m} has {n:,} boxes against {len(keys):,} common; "
                     "the union analysis uses the common set only.")
    L.append("")

    # ---------------- the union test, first because it is the point
    L.append("1. UNION TEST. Is the ceiling method-specific or intrinsic?")
    for cov_thr in covs:
        L.append(f"  At coverage {cov_thr:g}, limit {args.limit:g}:")
        per = {}
        for m in present:
            per[m] = {k for k in keys if found(data[m]["k"][k], args.limit,
                                               cov_thr)}
            L.append(f"    {m:<6} alone          {len(per[m]) / len(keys):.4f}")
        union = set().union(*per.values())
        inter = set.intersection(*per.values())
        best = max(len(per[m]) for m in present)
        L.append(f"    UNION (any method)     {len(union) / len(keys):.4f}")
        L.append(f"    INTERSECTION (all)     {len(inter) / len(keys):.4f}")
        L.append(f"    union minus best single {(len(union) - best) / len(keys):+.4f}")
        L.append("    A small gain means the methods miss the SAME boxes and "
                 "the ceiling is intrinsic to motion proposals.")
        L.append("    A large gain means it is method-specific.")
        # what only one method finds
        for m in present:
            only = per[m] - set().union(*[per[o] for o in present if o != m])
            L.append(f"    found ONLY by {m:<6} {len(only) / len(keys):.4f}")
        missed_all = keys - union
        L.append(f"    missed by ALL          {len(missed_all) / len(keys):.4f}")
        if missed_all:
            by_scene = defaultdict(int)
            tot_scene = defaultdict(int)
            for k in keys:
                tot_scene[data[present[0]]["k"][k]["scene"]] += 1
            for k in missed_all:
                by_scene[data[present[0]]["k"][k]["scene"]] += 1
            L.append("    boxes missed by all, by scene (share of that scene):")
            for sc in sorted(tot_scene):
                L.append(f"      {sc:<15} {by_scene[sc]:>6} / "
                         f"{tot_scene[sc]:<6} = "
                         f"{by_scene[sc] / tot_scene[sc]:.4f}")
        L.append("")

    # ---------------- per-method curves
    L.append("2. RECALL AGAINST THE NORMALISED LOWER LIMIT")
    for m in present:
        tot_frames = sum(data[m]["f"].values())
        L.append(f"  {m.upper()}  ({len(data[m]['k']):,} boxes, "
                 f"{tot_frames:,} frames)")
        L.append("    min_frac    blobs/frame  " +
                 "  ".join(f"recall@{c:g}" for c in covs))
        for f in FRAC_GRID:
            bpf = sum(c for lo, c, fa in data[m]["h"] if lo >= f * fa)
            bpf = bpf / tot_frames if tot_frames else 0
            recs = [np.mean([found(v, f, c) for v in data[m]["k"].values()])
                    for c in covs]
            L.append(f"    {f:>9.3g}  {bpf:>11.2f}  " +
                     "  ".join(f"{r:>10.4f}" for r in recs))
        L.append("")

    # ---------------- coverage distributions
    L.append("3. COVERAGE DISTRIBUTION, no size limit")
    for m in present:
        c = np.array([v["cov"] for v in data[m]["k"].values()])
        L.append(f"  {m:<6} " + "  ".join(
            f"p{p}={np.percentile(c, p):.3f}"
            for p in [5, 25, 50, 75, 95]) +
            f"   zero-coverage {float((c == 0).mean()):.4f}")
    L.append("")

    # ---------------- per scene at the ruled limit
    L.append(f"4. PER SCENE at the ruled limit {args.limit:g}, "
             f"coverage {covs[0]:g}")
    scenes = sorted({v["scene"] for v in data[present[0]]["k"].values()})
    L.append("  scene            n      " + "  ".join(f"{m:>7}" for m in present)
             + "     union")
    for sc in scenes:
        ks = [k for k in keys if data[present[0]]["k"][k]["scene"] == sc]
        if not ks:
            continue
        cells, sets = [], []
        for m in present:
            s = {k for k in ks if found(data[m]["k"][k], args.limit, covs[0])}
            sets.append(s)
            cells.append(f"{len(s) / len(ks):>7.4f}")
        u = len(set().union(*sets)) / len(ks)
        L.append(f"  {sc:<15} {len(ks):>5}  " + "  ".join(cells) +
                 f"   {u:>7.4f}")
    L.append("")

    L.append("5. PER RESOLUTION at the ruled limit")
    for res in sorted({v["res"] for v in data[present[0]]["k"].values()}):
        ks = [k for k in keys if data[present[0]]["k"][k]["res"] == res]
        if not ks:
            continue
        cells, sets = [], []
        for m in present:
            s = {k for k in ks if found(data[m]["k"][k], args.limit, covs[0])}
            sets.append(s)
            cells.append(f"{m} {len(s) / len(ks):.4f}")
        u = len(set().union(*sets)) / len(ks)
        L.append(f"  {res:<11} n={len(ks):>6}  " + "  ".join(cells) +
                 f"   union {u:.4f}")

    txt = "\n".join(L)
    path = os.path.join(args.out, "bgs_method_comparison.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
