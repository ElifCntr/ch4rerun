#!/usr/bin/env python3
"""
Item D ceiling comparison, recomputed with start-of-clip runs excluded.

WHY THE ORIGINAL FIGURE IS WRONG. The ceiling for a stationary run beginning
at frame n is derived from alpha = 1/(2n), the value the subtractor's warm-up
ramp has reached by then. At n = 0 that gives alpha = 0.5 and a ceiling of
0.152 frames, which no run of any length can satisfy. Thirteen of thirty
videos annotate from frame 0, so every run beginning there fails the
comparison by construction rather than by measurement, and the reported
"30 of 46 tracks fit, sixteen do not" is inflated.

WHAT THIS CHANGES AND WHAT IT DOES NOT. The absorption-target derivation was
abandoned for two reasons: the ramp caps absorption regardless of history, and
a single target in seconds is not uniformly achievable across clips. Neither
rests on this count. Only the supporting figure moves.

THE EXCLUSION IS PRINCIPLED. A run beginning before frame 3 cannot be a T=8
window centre, so extraction never proposes it, and it could not have been
absorbed from a tubelet that does not exist. The same cut-off the extraction
uses is applied here, rather than a threshold chosen to improve the number.

Both figures are reported side by side. The original is not deleted, because
the corrected one only means something against it.

Usage:
    python tools/recompute_ceiling.py
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

T = 8
LEAD = (T - 1) // 2       # 3; a run starting before this is never proposed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks",
                    default="reports/item_d_stationarity_tracks.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--min-start", type=int, default=LEAD)
    args = ap.parse_args()

    if not os.path.exists(args.tracks):
        print(f"FAIL missing {args.tracks}")
        print("     regenerate with tools/measure_stationarity.py")
        sys.exit(1)

    rows = []
    with open(args.tracks, newline="") as fh:
        rd = csv.DictReader(fh)
        need = ["anchored_run_frames", "anchored_run_start_frame",
                "mog2_ceiling_frames_at_run_start", "ceiling_covers_run",
                "video", "scene", "fps"]
        missing = [c for c in need if c not in (rd.fieldnames or [])]
        if missing:
            print(f"FAIL {args.tracks} missing: {missing}")
            print(f"     present: {rd.fieldnames}")
            sys.exit(1)
        for r in rd:
            rows.append({
                "video": r["video"], "scene": r["scene"],
                "fps": float(r["fps"]),
                "run": int(r["anchored_run_frames"]),
                "start": int(r["anchored_run_start_frame"]),
                "ceiling": float(r["mog2_ceiling_frames_at_run_start"]),
            })

    def covers(r):
        return r["ceiling"] >= r["run"]

    kept = [r for r in rows if r["start"] >= args.min_start]
    dropped = [r for r in rows if r["start"] < args.min_start]

    L = ["ITEM D CEILING COMPARISON, RECOMPUTED",
         f"Source {args.tracks}, {len(rows)} tracks.",
         f"Runs beginning before frame {args.min_start} are excluded: they "
         "cannot be a T=8 window centre, so extraction never proposes them, "
         "and their computed ceiling is degenerate.",
         ""]

    L.append("AS ORIGINALLY REPORTED, every track included")
    n_ok = sum(1 for r in rows if covers(r))
    L.append(f"  {n_ok} of {len(rows)} tracks fit inside the ceiling, "
             f"{len(rows) - n_ok} do not")
    L.append("")

    L.append(f"EXCLUDED, runs starting before frame {args.min_start}")
    L.append(f"  {len(dropped)} tracks")
    for r in sorted(dropped, key=lambda r: r["start"])[:12]:
        L.append(f"    {r['video']:<44} start {r['start']:>3}  "
                 f"run {r['run']:>4}  ceiling {r['ceiling']:>7.2f}")
    if dropped:
        n_dropped_ok = sum(1 for r in dropped if covers(r))
        L.append(f"  of these, {n_dropped_ok} fit and "
                 f"{len(dropped) - n_dropped_ok} do not, which is the "
                 "artefact: a ceiling below one frame cannot be satisfied")
    L.append("")

    L.append("CORRECTED, start-of-clip runs excluded")
    if not kept:
        L.append("  no tracks remain; report the exclusion, not a ratio")
    else:
        n_ok_k = sum(1 for r in kept if covers(r))
        L.append(f"  {n_ok_k} of {len(kept)} tracks fit inside the ceiling, "
                 f"{len(kept) - n_ok_k} do not "
                 f"({(len(kept) - n_ok_k) / len(kept):.1%})")
        L.append("")
        run = np.array([r["run"] for r in kept], float)
        ceil = np.array([r["ceiling"] for r in kept], float)
        sec_run = np.array([r["run"] / r["fps"] for r in kept])
        sec_ceil = np.array([r["ceiling"] / r["fps"] for r in kept])
        for name, a in (("anchored run, frames", run),
                        ("ceiling, frames", ceil),
                        ("anchored run, s", sec_run),
                        ("ceiling, s", sec_ceil)):
            L.append(f"  {name:<22} " + "  ".join(
                f"p{p}={np.percentile(a, p):.2f}"
                for p in (10, 25, 50, 75, 90)))
        L.append("")
        by = defaultdict(lambda: [0, 0])
        for r in kept:
            by[r["scene"]][1] += 1
            if not covers(r):
                by[r["scene"]][0] += 1
        L.append("  tracks exceeding the ceiling, by scene")
        for sc in sorted(by):
            bad, tot = by[sc]
            L.append(f"    {sc:<15} {bad:>3} of {tot:>3}")

    L.append("")
    L.append("READING. The corrected count is the one the write-up quotes. "
             "The original is kept beside it because the correction only "
             "means something against it. Neither figure bears on the "
             "abandonment of the absorption-target derivation, which rests "
             "on the ramp cap and on a single target in seconds not being "
             "uniformly achievable.")

    txt = "\n".join(L)
    path = os.path.join(args.out, "item_d_ceiling_corrected.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
