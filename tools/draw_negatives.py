#!/usr/bin/env python3
"""
Item 5. Training negative draw, one list per run seed. Coordinates only.

Ruled 10 August and 18 August:
  - training loader only; val and test stay uncapped
  - negatives drawn ONLY from the coverage-exactly-0 pool
  - size-stratified, with band edges taken from the drone size distribution,
    quotas set so the negative size profile matches the positive one
  - ratio 2x
  - drawn once offline, committed, shared across all four arms, VARYING BY
    SEED and never by arm
  - no hard negative mining
  - shortfalls recorded, NEVER backfilled

BAND EDGES, declared with a rationale rather than invented. The edges are the
DECILES of the POSITIVE proposals' normalised area within this split. Two
consequences follow by construction rather than by tuning. The bands come from
the measured drone size distribution as ruled, since positives are drone
proposals. And a quota of ratio x (positives in band) makes the negative size
profile match the positive one exactly, which is what the ruling asks for,
without any edge being chosen by hand. Ten bands is the only free choice; it
gives about 1,300 positives per band, enough that a per-band quota is not
itself noisy.

These are STRATIFICATION BANDS. They are not the REPORTING BANDS, which are
benchmark-derived (COCO, AI-TOD) and used only for per-size-band recall. The
11 August ruling forbids the two sharing a name or edges by accident.

Two passes over proposals.csv so memory stays small: pass one reads the
normalised areas and row offsets, pass two emits only the drawn rows.

Usage:
    python tools/draw_negatives.py
    python tools/draw_negatives.py --ratio 2 --seeds 1,2,3 --bands 10
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default="data/proposals/proposals.csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ratio", type=int, default=2, help="ruled 18 Aug")
    ap.add_argument("--seeds", default="1,2,3", help="ruled 18 Aug")
    ap.add_argument("--bands", type=int, default=10)
    ap.add_argument("--out", default="data/negatives")
    ap.add_argument("--report", default="reports")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)

    if not os.path.exists(args.proposals):
        print(f"FAIL missing {args.proposals}")
        print("     It is not committed by design; regenerate with "
              "tools/extract_proposals.py")
        sys.exit(1)

    # pass 1: normalised areas for positives, and areas plus row numbers for
    # the coverage-0 negatives
    pos_area, neg_area, neg_row = [], [], []
    with open(args.proposals, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ix = {c: i for i, c in enumerate(header)}
        for n, row in enumerate(rd):
            if row[ix["split"]] != args.split:
                continue
            lab = row[ix["label"]]
            if lab == "positive":
                pos_area.append(float(row[ix["area_frac"]]))
            elif lab == "negative":
                neg_area.append(float(row[ix["area_frac"]]))
                neg_row.append(n)

    pos_area = np.asarray(pos_area, dtype=np.float64)
    neg_area = np.asarray(neg_area, dtype=np.float64)
    neg_row = np.asarray(neg_row, dtype=np.int64)
    if pos_area.size == 0 or neg_area.size == 0:
        print(f"FAIL no positives or no negatives in split '{args.split}'")
        sys.exit(1)

    # band edges: deciles of the positive normalised area
    qs = np.linspace(0, 100, args.bands + 1)
    edges = np.percentile(pos_area, qs)
    edges[0], edges[-1] = -np.inf, np.inf

    pos_counts = np.histogram(pos_area, bins=np.r_[
        pos_area.min() - 1, edges[1:-1], pos_area.max() + 1])[0]
    neg_band = np.digitize(neg_area, edges[1:-1], right=False)

    lines = ["TRAINING NEGATIVE DRAW",
             f"split {args.split}, ratio {args.ratio}x, "
             f"{args.bands} bands, seeds {seeds}",
             f"positives {pos_area.size:,}, coverage-0 pool "
             f"{neg_area.size:,}",
             "",
             "Band edges are DECILES OF THE POSITIVE normalised area, so the "
             "negative size profile matches the positive one by construction. "
             "These are STRATIFICATION bands, not the benchmark-derived "
             "REPORTING bands.",
             "",
             "   band   area_frac range                 positives   quota   "
             "available   drawn   shortfall"]

    quotas, avail = [], []
    for b in range(args.bands):
        lo = edges[b] if np.isfinite(edges[b]) else pos_area.min()
        hi = edges[b + 1] if np.isfinite(edges[b + 1]) else pos_area.max()
        q = int(pos_counts[b]) * args.ratio
        a = int((neg_band == b).sum())
        quotas.append(q)
        avail.append(a)
        lines.append(f"   {b:>4}   {lo:.3e} .. {hi:.3e}   {pos_counts[b]:>9,}"
                     f"   {q:>5,}   {a:>9,}   {min(q, a):>5,}   "
                     f"{max(0, q - a):>9,}")

    total_quota = sum(quotas)
    total_drawn = sum(min(q, a) for q, a in zip(quotas, avail))
    lines.append("")
    lines.append(f"   TOTAL quota {total_quota:,}, drawable {total_drawn:,}, "
                 f"shortfall {total_quota - total_drawn:,}")
    if total_quota != total_drawn:
        lines.append("   SHORTFALL RECORDED AND NOT BACKFILLED, per the "
                     "10 August ruling. The deficit is not redistributed to "
                     "other bands, because that would change the size "
                     "profile the stratification exists to preserve.")
    lines.append("")

    # draw per seed
    chosen = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        picks = []
        for b in range(args.bands):
            idx = np.flatnonzero(neg_band == b)
            k = min(quotas[b], idx.size)
            if k:
                picks.append(rng.choice(idx, size=k, replace=False))
        sel = np.sort(np.concatenate(picks)) if picks else np.array([],
                                                                   dtype=int)
        chosen[seed] = set(neg_row[sel].tolist())
        lines.append(f"   seed {seed}: {len(chosen[seed]):,} negatives drawn")

    overlap = set.intersection(*chosen.values()) if len(chosen) > 1 else set()
    union = set.union(*chosen.values())
    lines.append(f"   union across seeds {len(union):,}, "
                 f"common to all {len(overlap):,}")
    lines.append("   Seeds draw independently from the same pool, so some "
                 "overlap is expected and is not a fault.")

    # pass 2: emit the drawn rows
    writers, files = {}, {}
    for seed in seeds:
        p = os.path.join(args.out, f"negatives_seed{seed}.csv")
        files[seed] = open(p, "w", newline="")
        writers[seed] = csv.writer(files[seed])

    wrote = defaultdict(int)
    with open(args.proposals, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        for seed in seeds:
            writers[seed].writerow(header)
        for n, row in enumerate(rd):
            for seed in seeds:
                if n in chosen[seed]:
                    writers[seed].writerow(row)
                    wrote[seed] += 1
    for seed in seeds:
        files[seed].close()

    for seed in seeds:
        if wrote[seed] != len(chosen[seed]):
            print(f"FAIL seed {seed} wrote {wrote[seed]} rows for "
                  f"{len(chosen[seed])} selections")
            sys.exit(1)

    meta = {
        "split": args.split, "ratio": args.ratio, "bands": args.bands,
        "band_edge_rule": "deciles of the positive normalised area",
        "seeds": seeds,
        "positives": int(pos_area.size),
        "coverage_zero_pool": int(neg_area.size),
        "quota_total": int(total_quota), "drawn_total": int(total_drawn),
        "shortfall_total": int(total_quota - total_drawn),
        "per_band": [{"band": b,
                      "positives": int(pos_counts[b]),
                      "quota": int(quotas[b]),
                      "available": int(avail[b]),
                      "shortfall": int(max(0, quotas[b] - avail[b]))}
                     for b in range(args.bands)],
        "per_seed_drawn": {str(s): len(chosen[s]) for s in seeds},
        "union_across_seeds": len(union),
        "common_to_all_seeds": len(overlap),
        "note": "Coordinates only. No crops written. Negatives come solely "
                "from the coverage-exactly-0 pool. No hard negative mining.",
    }
    with open(os.path.join(args.out, "negatives_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    txt = "\n".join(lines)
    with open(os.path.join(args.report, "negative_draw.txt"), "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {args.out}/negatives_seed*.csv and "
          f"{args.report}/negative_draw.txt")


if __name__ == "__main__":
    main()
