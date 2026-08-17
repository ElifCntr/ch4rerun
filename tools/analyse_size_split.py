#!/usr/bin/env python3
"""
1b-M step 3, task 3. Item B per-resolution and per-scene split.

Authorised 15 Aug. Retrieval from reports/instances_boxes.csv, not a new
measurement. Needed because the pooled p1 of 4.9 px equivalent side is very
likely drawn from the gopro_fisheye end (scene median 9.9 px) and is the
wrong input to a threshold scaled by frame area, and because the max_area
sweep starts at the MEASURED MAXIMUM ANNOTATED BOX AREA PER RESOLUTION,
which does not exist anywhere yet.

BOX AREA VERSUS VISIBLE AREA. The CSV carries both, plus a clipped flag,
because annotated boxes may extend past the frame edge. A connected
component cannot. So the quantity min_area actually filters against is
VISIBLE area, while item B's recorded figures are box area. Both are
reported side by side with the clipped fraction per group. Which one the
sweep should use is a ruling, not a finding, and is not decided here.

Cross-checks against the recorded item B values, so a schema or scope
mismatch shows up as a disagreement rather than a silently different number.
Disagreements are reported, never reconciled. Exit status is 1 if any
disagreement is found.

stdlib csv plus numpy only. No pandas, deliberately, so the verified torch
environment is not perturbed for an analysis script.

Usage:
    python tools/analyse_size_split.py
    python tools/analyse_size_split.py --splits train,val,test
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

PCTS = [1, 25, 50, 75, 99]
NEED = ["video", "split", "scene", "resolution", "w", "h", "box_area",
        "visible_area", "clipped", "area_frac", "equiv_side"]

# Recorded item B figures (train+val, n=23,883). Cross-check only, never input.
RECORDED = {
    "n": 23883,
    "equiv_side": {"p1": 4.9, "p25": 11.0, "p50": 18.3, "p75": 38.0,
                   "p99": 147.3},
    "res_n": {"1920x1080": 18730, "3840x2160": 5153},
    "res_median_equiv_side": {"1920x1080": 14.4, "3840x2160": 45.2},
    "res_median_area_frac": {"1920x1080": 1.00e-04, "3840x2160": 2.46e-04},
    "scene_median_equiv_side": {"gopro_fisheye": 9.9, "meadow": 19.3,
                                "lake": 23.5, "seaside": 25.4,
                                "industrial": 41.7},
    "min_box_area": 12.0,
    "median_area_frac": 1.30e-04,
}


def as_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None
    d = {"n": int(a.size), "min": float(a.min()), "max": float(a.max())}
    for p in PCTS:
        d[f"p{p}"] = float(np.percentile(a, p))
    return d


def fmt(d, sig=4):
    if d is None:
        return "n/a"
    return "  ".join(f"{k}={d[k]:.{sig}g}"
                     for k in ["min", "p1", "p25", "p50", "p75", "p99", "max"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default="reports/instances_boxes.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--splits", default="train,val",
                    help="item B was measured on train+val")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wanted = {s.strip() for s in args.splits.split(",")}
    rows = []
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
            fw, fh_ = (float(v) for v in r["resolution"].split("x"))
            fa = fw * fh_
            va = float(r["visible_area"])
            rows.append({
                "scene": r["scene"], "resolution": r["resolution"],
                "frame_area": fa,
                "box_area": float(r["box_area"]),
                "visible_area": va,
                "equiv_side": float(r["equiv_side"]),
                "visible_equiv_side": float(np.sqrt(va)) if va > 0 else 0.0,
                "area_frac": float(r["area_frac"]),
                "visible_area_frac": va / fa,
                "clipped": as_bool(r["clipped"]),
            })

    if not rows:
        print(f"FAIL no rows for splits {sorted(wanted)}")
        sys.exit(1)

    def col(rs, k):
        return [r[k] for r in rs]

    def group(rs, key):
        g = defaultdict(list)
        for r in rs:
            g[r[key]].append(r)
        return g

    L = ["ITEM B SPLIT, per resolution and per scene",
         f"Source {args.boxes}, splits {sorted(wanted)}, n={len(rows)}",
         "Retrieval only. No threshold is set here and no value is chosen.",
         "",
         "BOX area is the annotated rectangle, which may extend past the "
         "frame edge. VISIBLE area is the part inside the frame, which is "
         "what a connected component can actually be. Both reported; which "
         "the sweep filters on is a ruling, not a finding.",
         ""]

    def emit(label, rs):
        clipped = float(np.mean([r["clipped"] for r in rs])) if rs else 0.0
        L.append(f"{label}  (n={len(rs)}, clipped={clipped:.1%})")
        for name, k in (("box area px^2", "box_area"),
                        ("vis area px^2", "visible_area"),
                        ("box equiv side px", "equiv_side"),
                        ("vis equiv side px", "visible_equiv_side")):
            L.append(f"  {name:<18} {fmt(stats(col(rs, k)))}")
        for name, k in (("box area fraction", "area_frac"),
                        ("vis area fraction", "visible_area_frac")):
            L.append(f"  {name:<18} {fmt(stats(col(rs, k)), sig=3)}")
        L.append("")

    emit("POOLED", rows)
    by_res = group(rows, "resolution")
    by_scene = group(rows, "scene")

    L.append("BY RESOLUTION\n")
    for res in sorted(by_res):
        emit(f"  {res}", by_res[res])
    L.append("BY SCENE\n")
    for sc in sorted(by_scene):
        emit(f"  {sc}", by_scene[sc])

    L.append("BY SCENE x RESOLUTION\n")
    cross = defaultdict(list)
    for r in rows:
        cross[(r["scene"], r["resolution"])].append(r)
    for k in sorted(cross):
        emit(f"  {k[0]} @ {k[1]}", cross[k])

    L.append("SWEEP ENDPOINTS, measured")
    L.append("  max_area sweeps UPWARD from the maximum annotated box area, "
             "per resolution (ruled 15 Aug):")
    for res in sorted(by_res):
        rs = by_res[res]
        fa = rs[0]["frame_area"]
        for name, k in (("box", "box_area"), ("visible", "visible_area")):
            m = max(col(rs, k))
            L.append(f"    {res} {name:<8} max={m:.0f} px^2  "
                     f"equiv_side={np.sqrt(m):.1f} px  frac={m / fa:.3e}")
    L.append("  min_area sweeps log-spaced from NO FILTER upward. The "
             "smallest annotated box per resolution is where the first real "
             "drone would be lost:")
    for res in sorted(by_res):
        rs = by_res[res]
        fa = rs[0]["frame_area"]
        for name, k in (("box", "box_area"), ("visible", "visible_area")):
            m = min(col(rs, k))
            L.append(f"    {res} {name:<8} min={m:.0f} px^2  "
                     f"equiv_side={np.sqrt(m):.1f} px  frac={m / fa:.3e}")
    L.append("")

    L.append("PER-RESOLUTION SMALL TAIL, the figure the pooled p1 hides")
    for res in sorted(by_res):
        s = stats(col(by_res[res], "equiv_side"))
        L.append(f"  {res}  n={s['n']}  min={s['min']:.2f}  p1={s['p1']:.2f}  "
                 f"p25={s['p25']:.2f} px equiv side")
    L.append("")

    L.append("CROSS-CHECK against the recorded item B values")
    diffs = []

    def check(name, got, want, tol):
        ok = got is not None and abs(got - want) <= tol
        L.append(f"  {'OK  ' if ok else 'DIFF'} {name}: recomputed "
                 f"{got:.6g} vs recorded {want:.6g}")
        if not ok:
            diffs.append(name)

    es = stats(col(rows, "equiv_side"))
    check("n", float(len(rows)), float(RECORDED["n"]), 0)
    for p, want in RECORDED["equiv_side"].items():
        check(f"pooled equiv side {p}", es[p], want, 0.15)
    check("pooled median area fraction",
          float(np.median(col(rows, "area_frac"))),
          RECORDED["median_area_frac"], 5e-7)
    check("smallest box area px^2", float(min(col(rows, "box_area"))),
          RECORDED["min_box_area"], 0)
    for res, want in RECORDED["res_n"].items():
        check(f"{res} n", float(len(by_res.get(res, []))), float(want), 0)
    for res, want in RECORDED["res_median_equiv_side"].items():
        if res in by_res:
            check(f"{res} median equiv side",
                  float(np.median(col(by_res[res], "equiv_side"))), want, 0.15)
    for res, want in RECORDED["res_median_area_frac"].items():
        if res in by_res:
            check(f"{res} median area fraction",
                  float(np.median(col(by_res[res], "area_frac"))), want, 5e-7)
    for sc, want in RECORDED["scene_median_equiv_side"].items():
        if sc in by_scene:
            check(f"{sc} median equiv side",
                  float(np.median(col(by_scene[sc], "equiv_side"))),
                  want, 0.15)

    L.append("")
    if diffs:
        L.append(f"DISAGREEMENTS FOUND ({len(diffs)}): {diffs}")
        L.append("Reported, not reconciled. A recomputed figure that differs "
                 "means the inputs differ from what was measured.")
    else:
        L.append("No disagreements. The split is consistent with the recorded "
                 "item B figures.")

    txt = "\n".join(L)
    with open(os.path.join(args.out, "item_b_split.txt"), "w") as fh:
        fh.write(txt)

    out_rows = []
    for key, g in (("resolution", by_res), ("scene", by_scene)):
        for name in sorted(g):
            rs = g[name]
            for unit, k in (("box_area_px2", "box_area"),
                            ("visible_area_px2", "visible_area"),
                            ("box_equiv_side_px", "equiv_side"),
                            ("visible_equiv_side_px", "visible_equiv_side"),
                            ("box_area_frac", "area_frac"),
                            ("visible_area_frac", "visible_area_frac")):
                s = stats(col(rs, k))
                if s:
                    out_rows.append({
                        "grouping": key, "group": name, "unit": unit,
                        "clipped_frac": round(float(
                            np.mean([r["clipped"] for r in rs])), 6), **s})
    csv_path = os.path.join(args.out, "item_b_split.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)

    print(txt)
    print(f"\nwritten to {args.out}/item_b_split.txt and {csv_path}")
    if diffs:
        sys.exit(1)


if __name__ == "__main__":
    main()
