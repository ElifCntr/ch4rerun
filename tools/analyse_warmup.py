#!/usr/bin/env python3
"""
Does a warm-up help? Measured, not assumed.

Reads reports/bgs_pass/matched.csv only. Nothing is re-run.

THE QUESTION. Skipping the first N frames of each video should help if the
subtractor's early instability is what produces the blob flood. It should not
help if the flood is codec noise, which is persistent.

WHAT IT REPORTS
  1. Blobs per frame and coverage against frame index, binned. If warm-up
     helps, both improve over the first bins and then flatten.
  2. The DENOMINATOR COST of each candidate warm-up: how many ground-truth
     boxes it discards, and which videos and scenes lose most.
  3. Recall computed BOTH WAYS at each candidate, on the full box set and on
     the surviving set, so a rise from discarding hard frames is visible
     rather than hidden.
  4. First annotated frame per video, which is the warm-up that costs nothing.

LIMITATION, stated rather than worked around. matched.csv holds only frames
that contain an annotated drone, so frames before the first annotation are
invisible here. The blob-count-against-index curve therefore starts at each
video's first annotated frame, not at frame 0. The frame-0 initialisation
blob is known separately from the histogram and is not visible in this file.

Usage:
    python tools/analyse_warmup.py
    python tools/analyse_warmup.py --coverage 0.5
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

BINS = [0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 900, 1400, 10**9]
WARMUPS = [0, 10, 25, 50, 75, 100, 150, 200, 300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default="reports/bgs_pass/matched.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--variant", default="none")
    args = ap.parse_args()

    if not os.path.exists(args.matched):
        print(f"FAIL missing {args.matched}")
        sys.exit(1)

    rows = []
    with open(args.matched, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["variant"] != args.variant:
                continue
            rows.append({
                "video": r["video"], "scene": r["scene"],
                "res": r["resolution"], "frame": int(r["frame"]),
                "cov": float(r["best_coverage"]),
                "area": float(r["matched_blob_area"]),
                "nblobs": int(r["n_blobs_in_frame"]),
            })
    if not rows:
        print(f"FAIL no rows for variant {args.variant}")
        sys.exit(1)

    L = [f"WARM-UP CHECK, variant {args.variant}, "
         f"coverage threshold {args.coverage:g}",
         f"Source {args.matched}, {len(rows)} annotated boxes",
         "",
         "LIMITATION: only frames containing an annotated drone appear here, "
         "so each video's curve starts at its first annotation, not frame 0.",
         ""]

    # 1. blobs and coverage against frame index
    L.append("1. AGAINST FRAME INDEX. If warm-up helps, the early bins are "
             "worse and then flatten.")
    L.append("   frame range        n     blobs/frame   median cov   "
             f"recall@{args.coverage:g}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sub = [r for r in rows if lo <= r["frame"] < hi]
        if not sub:
            continue
        nb = np.mean([r["nblobs"] for r in sub])
        cv = np.median([r["cov"] for r in sub])
        rc = np.mean([r["cov"] >= args.coverage for r in sub])
        label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
        L.append(f"   {label:<16} {len(sub):>5}   {nb:>11.1f}   "
                 f"{cv:>10.3f}   {rc:>10.4f}")
    L.append("")

    # 2 and 3. cost and effect of each candidate warm-up
    total = len(rows)
    base_hits = sum(1 for r in rows if r["cov"] >= args.coverage)
    L.append("2. CANDIDATE WARM-UPS. Recall both ways, so a rise from "
             "discarding hard frames is visible.")
    L.append("   warmup   boxes kept   dropped   recall on kept   "
             "recall on ALL boxes   blobs/frame")
    for w in WARMUPS:
        kept = [r for r in rows if r["frame"] >= w]
        if not kept:
            continue
        hits = sum(1 for r in kept if r["cov"] >= args.coverage)
        on_kept = hits / len(kept)
        on_all = hits / total          # dropped boxes count as misses
        nb = np.mean([r["nblobs"] for r in kept])
        L.append(f"   {w:>6}   {len(kept):>10}   {total - len(kept):>7}   "
                 f"{on_kept:>14.4f}   {on_all:>19.4f}   {nb:>11.1f}")
    L.append("")
    L.append(f"   Baseline recall with no warm-up: {base_hits / total:.4f}")
    L.append("   'recall on ALL boxes' charges discarded boxes as misses, "
             "which is the honest figure if the chapter still claims to "
             "detect drones in those frames.")
    L.append("")

    # 4. free warm-up per video
    L.append("3. FIRST ANNOTATED FRAME PER VIDEO. A warm-up up to this "
             "value costs that video nothing.")
    first = {}
    for r in rows:
        v = r["video"]
        if v not in first or r["frame"] < first[v]:
            first[v] = r["frame"]
    for v in sorted(first, key=lambda k: first[k]):
        L.append(f"   {first[v]:>5}   {v}")
    vals = np.array(list(first.values()))
    L.append(f"   median first annotated frame: {np.median(vals):.0f}, "
             f"min {vals.min()}, max {vals.max()}")
    L.append("")

    # 5. who pays, by scene, at a mid candidate
    L.append("4. WHO PAYS. Boxes dropped by each warm-up, by scene.")
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    L.append("   scene            n      " +
             "  ".join(f"w={w}" for w in [25, 50, 100, 200]))
    for sc in sorted(by_scene):
        sub = by_scene[sc]
        cells = [f"{sum(1 for r in sub if r['frame'] < w) / len(sub):.3f}"
                 for w in [25, 50, 100, 200]]
        L.append(f"   {sc:<15} {len(sub):>5}  " +
                 "  ".join(f"{c:>5}" for c in cells))

    txt = "\n".join(L)
    path = os.path.join(args.out, "warmup_check.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
