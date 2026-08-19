#!/usr/bin/env python3
"""
Read the background-subtractor dump and produce the operating curves.

Runs offline on reports/bgs_pass/{matched,hist}.csv. The subtractor is never
re-run, which is the whole point of the two-table design.

WHAT IT ANSWERS
  - drone recall against the lower size limit, for both cleanup variants
  - blobs per frame against the same limit
  - the two joined, which is the operating curve the limit is read off
  - the same in AREA-FRACTION units, which is the normalisation arm
  - what the upper size limit removes
  - tubelet count and hours per epoch at each operating point, from the
    measured 0.170 s per tubelet per epoch for the whole twelve-run matrix

Recall counts a ground-truth box as found when its covering blob reaches the
coverage threshold AND survives the size limits. Coverage thresholds are
reported as a small set rather than one, since no threshold is ruled.

Nothing is chosen here. Every number is read off the dump.

Usage:
    python tools/analyse_bgs_pass.py
    python tools/analyse_bgs_pass.py --coverage 0.5,0.7,0.9
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

SEC_PER_TUBELET_PER_EPOCH = 0.170   # measured 12 Aug, whole 12-run matrix, T=8
ABS_GRID = [0, 4, 9, 15, 24, 40, 64, 100, 160, 250, 400, 640, 1000]
FRAC_GRID = [0.0, 2e-6, 5e-6, 7.23e-6, 1.2e-5, 2e-5, 3.5e-5, 6e-5, 1e-4]


def frame_area(res):
    w, h = res.split("x")
    return float(w) * float(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="reports/bgs_pass")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--coverage", default="0.5,0.7")
    args = ap.parse_args()
    covs = [float(c) for c in args.coverage.split(",")]

    mpath = os.path.join(args.dump, "matched.csv")
    hpath = os.path.join(args.dump, "hist.csv")
    for p in (mpath, hpath):
        if not os.path.exists(p):
            print(f"FAIL missing {p}")
            sys.exit(1)

    # matched: one row per GT box per variant
    rows = defaultdict(list)          # variant -> list of dicts
    with open(mpath, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["variant"]].append({
                "scene": r["scene"], "res": r["resolution"],
                "fa": frame_area(r["resolution"]),
                "cov": float(r["best_coverage"]),
                "area": float(r["matched_blob_area"]),
                "gt_area": float(r["gt_area"]),
            })

    # hist: blob-area histogram per video per variant
    hist = defaultdict(list)          # variant -> list of (lo, hi, count, fa)
    frames = defaultdict(dict)        # variant -> video -> frames_processed
    with open(hpath, newline="") as fh:
        for r in csv.DictReader(fh):
            v = r["variant"]
            hist[v].append((float(r["bin_lo"]), float(r["bin_hi"]),
                            int(r["count"]), frame_area(r["resolution"])))
            frames[v][r["video"]] = int(r["frames_processed"])

    variants = sorted(rows)
    L = ["BGS PASS OPERATING CURVES",
         f"Source {args.dump}, variants {variants}",
         "Read off the dump. Nothing is chosen here.",
         ""]

    def blobs_per_frame(var, thr, frac=False):
        """Blobs at or above the threshold, per frame. Uses bin_lo, so this
        UNDERCOUNTS within the straddling bin. Bin width is about 3 per cent."""
        n = 0
        for lo, hi, c, fa in hist[var]:
            t = thr * fa if frac else thr
            if lo >= t:
                n += c
        tot = sum(frames[var].values())
        return n / tot if tot else 0.0

    def recall(var, thr, cov_thr, frac=False, sub=None):
        rs = sub if sub is not None else rows[var]
        if not rs:
            return float("nan"), 0
        ok = 0
        for r in rs:
            t = thr * r["fa"] if frac else thr
            if r["cov"] >= cov_thr and r["area"] >= t:
                ok += 1
        return ok / len(rs), len(rs)

    # ---- absolute-pixel arm
    for var in variants:
        L.append(f"VARIANT {var.upper()}  "
                 f"({len(rows[var])} boxes, "
                 f"{sum(frames[var].values())} frames)")
        head = ("  min_area   blobs/frame  " +
                "  ".join(f"recall@{c:g}" for c in covs) +
                "   tubelets/frame   h/epoch")
        L.append(head)
        for thr in ABS_GRID:
            bpf = blobs_per_frame(var, thr)
            recs = [recall(var, thr, c)[0] for c in covs]
            # tubelets per frame = surviving blobs; hours for the whole matrix
            n_tub = bpf * sum(frames[var].values())
            hpe = SEC_PER_TUBELET_PER_EPOCH * n_tub / 3600.0
            L.append(f"  {thr:>8}   {bpf:>10.2f}  " +
                     "  ".join(f"{r:>9.4f}" for r in recs) +
                     f"   {bpf:>13.2f}   {hpe:>7.2f}")
        L.append("")

    # ---- area-fraction arm, the normalisation question
    L.append("AREA-FRACTION ARM (single normalised limit across resolutions)")
    for var in variants:
        L.append(f"  VARIANT {var.upper()}")
        L.append("    min_frac     blobs/frame  " +
                 "  ".join(f"recall@{c:g}" for c in covs))
        for thr in FRAC_GRID:
            bpf = blobs_per_frame(var, thr, frac=True)
            recs = [recall(var, thr, c, frac=True)[0] for c in covs]
            L.append(f"    {thr:>9.3g}   {bpf:>10.2f}  " +
                     "  ".join(f"{r:>9.4f}" for r in recs))
        L.append("")

    # ---- per resolution and per scene, at the absolute floor
    L.append("PER RESOLUTION, absolute limit, recall at each coverage")
    for var in variants:
        by = defaultdict(list)
        for r in rows[var]:
            by[r["res"]].append(r)
        for res in sorted(by):
            L.append(f"  {var:<6} {res:<11} n={len(by[res])}")
            for thr in [0, 9, 15, 24, 40, 100]:
                recs = [recall(var, thr, c, sub=by[res])[0] for c in covs]
                L.append(f"    min_area {thr:>4}  " +
                         "  ".join(f"{r:.4f}" for r in recs))
        L.append("")

    L.append("PER SCENE, absolute limit, recall at the first coverage "
             f"threshold ({covs[0]:g})")
    for var in variants:
        by = defaultdict(list)
        for r in rows[var]:
            by[r["scene"]].append(r)
        L.append(f"  VARIANT {var.upper()}")
        L.append("    scene            n      " +
                 "  ".join(f"ma={t}" for t in [0, 9, 15, 24, 40, 100]))
        for sc in sorted(by):
            vals = [recall(var, t, covs[0], sub=by[sc])[0]
                    for t in [0, 9, 15, 24, 40, 100]]
            L.append(f"    {sc:<15} {len(by[sc]):>5}  " +
                     "  ".join(f"{v:.4f}" for v in vals))
        L.append("")

    # ---- coverage distribution, no size limit
    L.append("COVERAGE DISTRIBUTION, no size limit")
    for var in variants:
        c = np.array([r["cov"] for r in rows[var]])
        L.append(f"  {var:<6} n={c.size}  " +
                 "  ".join(f"p{p}={np.percentile(c, p):.3f}"
                           for p in [0, 5, 10, 25, 50, 75, 95, 100]))
        L.append(f"         fraction with zero coverage: "
                 f"{float((c == 0).mean()):.4f}")
    L.append("")

    # ---- upper limit
    L.append("UPPER LIMIT, what it removes")
    L.append("  Blobs per frame at or above each fraction of the frame")
    for var in variants:
        cells = []
        for f in [2.65e-3, 4.10e-2, 0.1, 0.5, 0.9]:
            cells.append(f"{f:.3g}:{blobs_per_frame(var, f, frac=True):.4f}")
        L.append(f"    {var:<6} " + "   ".join(cells))
    L.append("  4K max annotated fraction is 2.65e-03, 1080p is 4.10e-02.")
    L.append("")

    L.append("NOTES")
    L.append("  blobs/frame uses bin_lo, so it undercounts inside the "
             "straddling bin. Bins are about 3 per cent wide.")
    L.append("  h/epoch applies 0.170 s per tubelet per epoch, measured "
             "12 Aug for the whole twelve-run matrix at T=8, and assumes "
             "every surviving blob becomes a tubelet. It is an UPPER BOUND "
             "on training tubelets, since negative capping has not been "
             "applied and is unruled.")
    L.append("  Frames here are train+val only, 30 videos.")

    txt = "\n".join(L)
    path = os.path.join(args.out, "bgs_operating_curves.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
