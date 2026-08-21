#!/usr/bin/env python3
"""
Find the two anchors for the re-ruled upper area limit.

Ruled 19 August. max_area becomes a SINGLE frame-area-normalised value placed
ABOVE the largest observed best-covering blob and BELOW the frame-0
initialisation blob, which is the only thing the limit demonstrably needs to
reject. The previous anchor, the largest annotated BOX, was wrong: a legitimate
covering blob that merges a drone with adjacent motion exceeds any box, which
is why 199 good covers were rejected at 4K.

THE TWO ANCHORS, both measured, neither chosen.

  LOWER anchor, from matched.csv. The largest best-covering blob of any train
  or val ground-truth box, as a fraction of frame area. Anything at or below
  this must survive, or the limit destroys a cover we know exists. Reported
  for all boxes and separately for boxes covered at 0.5 and at 0.7, since the
  labelling threshold is 0.5 and only those become positives.

  UPPER anchor, from hist.csv. The initialisation blob. It is identified not by
  assumption but by COUNT: blobs above it number at most one per video, which
  is the signature of a once-per-video artefact rather than scene content. The
  script walks the upper tail and reports where the per-frame count reaches
  that floor.

THE PROPOSED VALUE is the GEOMETRIC MIDPOINT of the two anchors. That is the
point of maximum multiplicative margin on both sides and requires no invented
factor. It is a proposal; the value is declared elsewhere.

IF THE ANCHORS OVERLAP the script says so and proposes nothing, per the
ruling: stop and report rather than choose inside an overlap.

Usage:
    python tools/find_max_area_anchors.py
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

GRID = [1e-3, 2e-3, 2.65e-3, 5e-3, 1e-2, 2e-2, 4.10e-2, 6e-2, 8e-2,
        1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 9e-1]


def frame_area(res):
    w, h = res.split("x")
    return float(w) * float(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="reports/bgs_pass_mog2")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--min-frame", type=int, default=3,
                    help="lowest frame that can be a T=8 window centre; "
                         "extraction never proposes below it")
    args = ap.parse_args()

    mp = os.path.join(args.dump, "matched.csv")
    hp = os.path.join(args.dump, "hist.csv")
    for p in (mp, hp):
        if not os.path.exists(p):
            print(f"FAIL missing {p}")
            sys.exit(1)

    # LOWER anchor: covering-blob areas as a fraction of frame area
    cov, frac, res_of, scene_of = [], [], [], []
    excl_frames, excl_max = [], 0.0
    with open(mp, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["variant"] != "none":
                continue
            fa = frame_area(r["resolution"])
            if int(r["frame"]) < args.min_frame:
                # cannot be a T=8 window centre, so extraction never sees it.
                # Frame 0 carries the initialisation blob, which "covers" any
                # drone present at 1.0 and is not a legitimate cover.
                excl_frames.append((r["video"], int(r["frame"]),
                                    float(r["matched_blob_area"]) / fa))
                excl_max = max(excl_max, float(r["matched_blob_area"]) / fa)
                continue
            cov.append(float(r["best_coverage"]))
            frac.append(float(r["matched_blob_area"]) / fa)
            res_of.append(r["resolution"])
            scene_of.append(r["scene"])
    cov = np.asarray(cov)
    frac = np.asarray(frac)
    res_of = np.asarray(res_of)
    scene_of = np.asarray(scene_of)

    # UPPER anchor: upper tail of every blob, and the per-video floor
    tail = []
    frames_per_video = {}
    with open(hp, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["variant"] != "none":
                continue
            fa = frame_area(r["resolution"])
            tail.append((float(r["bin_lo"]) / fa, int(r["count"])))
            frames_per_video[r["video"]] = int(r["frames_processed"])
    n_videos = len(frames_per_video)
    n_frames = sum(frames_per_video.values())

    def blobs_per_frame(f):
        return sum(c for lo, c in tail if lo >= f) / n_frames

    L = ["UPPER AREA LIMIT, THE TWO MEASURED ANCHORS",
         f"Source {args.dump}, {cov.size:,} ground-truth boxes, "
         f"{n_videos} videos, {n_frames:,} frames (train and val).",
         "Nothing is chosen here beyond the stated midpoint proposal.",
         ""]

    L.append(f"EXCLUDED: {len(excl_frames)} boxes on frames below "
             f"{args.min_frame}, which cannot be a T=8 window centre and are "
             "never proposed by extraction. Largest covering blob among them "
             f"{excl_max:.4e}. This is where the initialisation blob lives; "
             "including it makes the two anchors the same object.")
    for v, f, x in sorted(excl_frames, key=lambda e: -e[2])[:8]:
        L.append(f"    {v:<44} frame {f:>2}  {x:.4e}")
    L.append("")
    L.append("LOWER ANCHOR, largest best-covering blob as a frame fraction")
    lo_anchor = None
    for label, mask in (("all boxes", np.ones_like(cov, bool)),
                        ("covered >= 0.5", cov >= 0.5),
                        ("covered >= 0.7", cov >= 0.7)):
        if not mask.any():
            continue
        f = frac[mask]
        L.append(f"  {label:<16} n={mask.sum():>6,}  max={f.max():.4e}  "
                 f"p99.9={np.percentile(f, 99.9):.4e}  "
                 f"p99={np.percentile(f, 99):.4e}")
        if label == "covered >= 0.5":
            lo_anchor = float(f.max())
    L.append("  The 0.5 row is the binding one: those boxes become positives, "
             "so their covers must survive the limit.")
    L.append("")
    L.append("  largest covering blob by resolution and scene, covered >= 0.5")
    m = cov >= 0.5
    for key, arr in (("resolution", res_of), ("scene", scene_of)):
        for g in sorted(set(arr[m])):
            sub = frac[m & (arr == g)]
            if sub.size:
                L.append(f"    {key:<11} {g:<15} max={sub.max():.4e}")
    L.append("")

    L.append("UPPER ANCHOR, upper tail of ALL blobs")
    L.append("  The initialisation blob is identified by COUNT, not by "
             "assumption: at most one per video is the signature of a "
             f"once-per-video artefact. That floor is {1 / n_frames * n_videos:.6f} "
             "blobs per frame.")
    L.append("   frame fraction   blobs/frame   blobs total   per video")
    floor = n_videos / n_frames
    hi_anchor = None
    for f in GRID:
        b = blobs_per_frame(f)
        tot = b * n_frames
        L.append(f"   {f:>13.3g}   {b:>11.5f}   {tot:>11,.0f}   "
                 f"{tot / n_videos:>9.2f}")
        if hi_anchor is None and b <= floor * 1.05:
            hi_anchor = f
    L.append("")

    if lo_anchor is None or hi_anchor is None:
        L.append("ANCHOR NOT FOUND. Reporting without a proposal.")
    elif hi_anchor <= lo_anchor:
        L.append("ANCHORS OVERLAP. The largest covering blob "
                 f"({lo_anchor:.4e}) is at or above the point where blobs "
                 f"reach one per video ({hi_anchor:.4e}). No value separates "
                 "them, so none is proposed, per the ruling. The limit cannot "
                 "reject the initialisation blob without also rejecting a "
                 "known good cover.")
    else:
        mid = math.sqrt(lo_anchor * hi_anchor)
        L.append("ANCHORS ARE SEPARATED.")
        L.append(f"  lower  {lo_anchor:.4e}   largest cover of a box at "
                 "coverage >= 0.5")
        L.append(f"  upper  {hi_anchor:.4e}   blobs reach one per video")
        L.append(f"  gap    {hi_anchor / lo_anchor:.1f}x")
        L.append("")
        L.append(f"  PROPOSED max_area_frac = {mid:.4e}")
        L.append(f"    margin above the largest known cover: "
                 f"{mid / lo_anchor:.1f}x")
        L.append(f"    margin below the initialisation blob: "
                 f"{hi_anchor / mid:.1f}x")
        L.append("    Geometric midpoint, so the multiplicative margin is "
                 "equal on both sides and no factor is invented.")
        L.append(f"    At this value the limit would reject "
                 f"{blobs_per_frame(mid) * n_frames:,.0f} blobs in train and "
                 f"val, {blobs_per_frame(mid):.5f} per frame.")
        L.append("")
        L.append("  CAVEAT CARRIED: the lower anchor comes from a dump that "
                 "records only the BEST cover per box, and it is train and "
                 "val only. Test was not in the 1b-M dumps and is 4K-heavy, "
                 "so a larger cover may exist there. The margin above the "
                 "known maximum is what absorbs that.")

    txt = "\n".join(L)
    path = os.path.join(args.out, "max_area_anchors.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
