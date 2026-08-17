#!/usr/bin/env python3
"""
1b-M step 3, task 2. Item D stationarity statistic.

Authorised 15 Aug as the anchor for the MOG2 absorption target, replacing
item C's median track length. Item C established that a track is the
annotated span of the clip rather than how long a drone stays visible, and
MOG2 absorption bites on PIXEL STATIONARITY, not on visibility.

Statistic, scale-free, invents no constant.

  Primary, ANCHORED. Longest run of consecutive frames such that every box
  centre in the run lies inside the box of the run's FIRST frame. That is
  "the drone has not moved off itself", the condition under which its pixels
  stay covered and the mixture can absorb it.

  Diagnostic, PAIRWISE. Longest run where each consecutive centre step stays
  within half the current box's width and height. Separates slow drift from
  per-frame jitter. Reported alongside, never instead.

Also reports each run's START FRAME, because the 15 Aug learningRate
measurement showed MOG2's absorption ceiling is set by ELAPSED FRAMES at the
moment the object stops, not by clip length. A run's start frame is
therefore as load-bearing as its length.

Reported BOTH WAYS, with and without associated instances (12 Aug ruling),
per scene, and per rate group in frames and seconds.

stdlib csv plus numpy only. No pandas, deliberately, so the verified torch
environment is not perturbed for an analysis script.

Usage:
    python tools/measure_stationarity.py
    python tools/measure_stationarity.py --splits train,val,test
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

PCTS = [0, 10, 25, 50, 75, 90, 95, 99, 100]
CF = 0.10  # confirmed 15 Aug from the OpenCV source comment, cf = 1 - 0.9
NEED = ["video", "split", "scene", "fps", "frame", "track_id",
        "associated", "x", "y", "w", "h", "cx", "cy"]


def as_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def anchored_run(cx, cy, x, y, w, h):
    """Longest run whose centres all lie inside the box of the run's first
    frame. Returns (length, start_offset)."""
    n = len(cx)
    if n == 0:
        return 0, 0
    best, best_s = 1, 0
    for s in range(n):
        inside = ((cx[s:] >= x[s]) & (cx[s:] <= x[s] + w[s]) &
                  (cy[s:] >= y[s]) & (cy[s:] <= y[s] + h[s]))
        bad = np.flatnonzero(~inside)
        run = int(bad[0]) if bad.size else int(n - s)
        if run > best:
            best, best_s = run, s
        if best >= n - s:
            break
    return best, best_s


def pairwise_run(cx, cy, w, h):
    """Longest run with each consecutive centre step inside half the current
    box. Returns (length, start_offset)."""
    n = len(cx)
    if n == 0:
        return 0, 0
    best, best_s, cur, cur_s = 1, 0, 1, 0
    for i in range(1, n):
        if abs(cx[i] - cx[i - 1]) <= w[i] / 2.0 and \
           abs(cy[i] - cy[i - 1]) <= h[i] / 2.0:
            cur += 1
        else:
            cur, cur_s = 1, i
        if cur > best:
            best, best_s = cur, cur_s
    return best, best_s


def ceiling_frames(stop_frame, cf=CF):
    """Slowest absorption MOG2 can deliver for an object that stops at
    stop_frame, from alpha = 1/(2*nframes). Measured 15 Aug: alpha is set by
    elapsed frames, not by history, whenever history exceeds 2*nframes."""
    n = max(int(stop_frame), 1)
    alpha = 1.0 / (2.0 * n)
    return math.log(1.0 - cf) / math.log(1.0 - alpha)


def summarise(vals, fps=None):
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {}
    out = {"n": int(a.size)}
    for p in PCTS:
        out[f"p{p}"] = round(float(np.percentile(a, p)), 1)
        if fps:
            out[f"p{p}_sec"] = round(out[f"p{p}"] / fps, 2)
    return out


def line(label, s, sec=False):
    if not s:
        return f"  {label:<24} (empty)"
    suf = "_sec" if sec else ""
    return (f"  {label:<24} n={s['n']:<4} " +
            "  ".join(f"p{p}={s['p' + str(p) + suf]:g}" for p in PCTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default="reports/instances_boxes.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wanted = {s.strip() for s in args.splits.split(",")}
    tracks = defaultdict(list)
    with open(args.boxes, newline="") as fh:
        rd = csv.DictReader(fh)
        missing = [c for c in NEED if c not in (rd.fieldnames or [])]
        if missing:
            print(f"FAIL {args.boxes} missing: {missing}")
            print(f"     present: {rd.fieldnames}")
            sys.exit(1)
        for r in rd:
            if r["split"] not in wanted:
                continue
            tracks[(r["video"], r["track_id"])].append(r)

    if not tracks:
        print(f"FAIL no rows for splits {sorted(wanted)}")
        sys.exit(1)

    out_rows = []
    for (vid, tid), rows in sorted(tracks.items()):
        rows.sort(key=lambda r: int(r["frame"]))
        f = np.array([int(r["frame"]) for r in rows])
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["y"]) for r in rows])
        w = np.array([float(r["w"]) for r in rows])
        h = np.array([float(r["h"]) for r in rows])
        cx = np.array([float(r["cx"]) for r in rows])
        cy = np.array([float(r["cy"]) for r in rows])
        # runs are over CONSECUTIVE frames only; split on any annotation gap
        segs = np.split(np.arange(len(rows)), np.where(np.diff(f) != 1)[0] + 1)
        ba, bas, bp, bps = 0, 0, 0, 0
        for idx in segs:
            if idx.size == 0:
                continue
            la, sa = anchored_run(cx[idx], cy[idx], x[idx], y[idx],
                                  w[idx], h[idx])
            if la > ba:
                ba, bas = la, int(f[idx[sa]])
            lp, sp = pairwise_run(cx[idx], cy[idx], w[idx], h[idx])
            if lp > bp:
                bp, bps = lp, int(f[idx[sp]])
        fps = float(rows[0]["fps"])
        ceil_a = ceiling_frames(bas)
        out_rows.append({
            "video": vid, "track_id": tid,
            "scene": rows[0]["scene"], "split": rows[0]["split"],
            "fps": fps,
            "associated": any(as_bool(r["associated"]) for r in rows),
            "track_frames": len(rows),
            "n_segments": len([s for s in segs if s.size]),
            "anchored_run_frames": ba,
            "anchored_run_start_frame": bas,
            "anchored_run_sec": round(ba / fps, 3),
            "pairwise_run_frames": bp,
            "pairwise_run_start_frame": bps,
            "pairwise_run_sec": round(bp / fps, 3),
            "mog2_ceiling_frames_at_run_start": round(ceil_a, 1),
            "mog2_ceiling_sec_at_run_start": round(ceil_a / fps, 3),
            "ceiling_covers_run": ceil_a >= ba,
        })

    per_path = os.path.join(args.out, "item_d_stationarity_tracks.csv")
    with open(per_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)

    def col(rows, k):
        return [r[k] for r in rows]

    L = ["ITEM D STATIONARITY STATISTIC",
         f"Source {args.boxes}, splits {sorted(wanted)}, "
         f"{len(out_rows)} tracks",
         "",
         "ANCHORED (primary): longest run of consecutive frames with the box "
         "centre inside the box of the run's first frame.",
         "PAIRWISE (diagnostic): longest run with each consecutive centre "
         "step inside half the current box.",
         "No threshold is set and no absorption target is chosen here.",
         ""]

    def block(title, rows):
        L.append(title)
        for name, k in (("anchored (primary)", "anchored_run_frames"),
                        ("pairwise (diagnostic)", "pairwise_run_frames")):
            L.append(line(name, summarise(col(rows, k))))
        L.append("")

    block("ALL TRACKS, frames", out_rows)
    block("EXCLUDING associated tracks, frames",
          [r for r in out_rows if not r["associated"]])
    block("ASSOCIATED tracks only, frames",
          [r for r in out_rows if r["associated"]])

    by_rate = defaultdict(list)
    for r in out_rows:
        by_rate[round(r["fps"], 2)].append(r)
    for rate in sorted(by_rate):
        s = summarise(col(by_rate[rate], "anchored_run_frames"), fps=rate)
        L.append(f"RATE GROUP {rate:g} fps, anchored")
        L.append(line("frames", s))
        L.append(line("seconds", s, sec=True))
        L.append("")

    by_scene = defaultdict(list)
    for r in out_rows:
        by_scene[r["scene"]].append(r)
    L.append("BY SCENE, anchored, frames")
    for scene in sorted(by_scene):
        L.append(line(scene, summarise(
            col(by_scene[scene], "anchored_run_frames"))))
    L.append("")

    L.append("MOG2 CEILING AGAINST THE MEASURED RUNS")
    L.append("Ceiling is the slowest absorption reachable for an object that "
             f"stops at frame n, from alpha = 1/(2n) at cf={CF}. Measured "
             "15 Aug, not assumed.")
    covered = sum(1 for r in out_rows if r["ceiling_covers_run"])
    L.append(f"  Tracks whose longest stationary run fits inside the ceiling "
             f"at its own start frame: {covered} of {len(out_rows)}")
    L.append(line("ceiling at run start, s",
                  summarise(col(out_rows, "mog2_ceiling_sec_at_run_start"))))
    L.append(line("anchored run, s",
                  summarise(col(out_rows, "anchored_run_sec"))))
    L.append("")
    L.append("Read this as the bound on any absorption target stated in "
             "seconds. A target above the ceiling column is unreachable on "
             "those tracks whatever history is set to.")

    txt = "\n".join(L)
    rep = os.path.join(args.out, "item_d_stationarity.txt")
    with open(rep, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {rep} and {per_path}")


if __name__ == "__main__":
    main()
