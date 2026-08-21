#!/usr/bin/env python3
"""
Training negative draw, one list per run seed. Coordinates only.

STRATIFIED BY SIZE BAND x SCENE, ruled 19 August. The earlier version
stratified by size band alone, and negatives concentrate in whichever videos
the subtractor finds noisy, so two videos supplied about 56 per cent of the
draw while positives were spread across 24. The risk is a network that learns
two videos' background rather than "not a drone", surfacing as a val-test gap
that reads like overfitting and is not.

The same reasoning was already ruled for the item F negative sample on
12 August, where scene stratification was required because unmatched proposals
cluster in cluttered scenes for structural reasons. It simply had not been
carried into the training draw.

RULES, unchanged except for the added axis:
  - training loader only; val and test stay uncapped
  - negatives drawn ONLY from the coverage-exactly-0 pool
  - quota per cell = ratio x (positives in that cell), so BOTH the size profile
    and the scene profile of the negatives match the positives by construction
  - shortfalls recorded, NEVER backfilled; redistributing a deficit would undo
    the matching the stratification exists to produce
  - drawn once, committed, shared across all four arms, VARYING BY SEED
  - no hard negative mining

BAND EDGES are the DECILES of the positive normalised area, taken GLOBALLY
rather than per scene. Per-scene deciles would make a band mean a different
size in each scene, and the size profile being matched is a property of the
drone distribution, not of any one scene. These are STRATIFICATION bands and
are not the benchmark-derived REPORTING bands.

Ratio and band count come from the ruled block of configs/ch4.yaml.

Two passes over proposals.csv so memory stays small.

Usage:
    python tools/draw_negatives.py
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.ruled import ruled  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--proposals", default="data/proposals/proposals.csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/negatives")
    ap.add_argument("--report", default="reports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ratio = ruled(cfg, "negatives.ratio")
    n_bands = ruled(cfg, "negatives.bands")
    seeds = ruled(cfg, "training.seeds")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)
    if not os.path.exists(args.proposals):
        print(f"FAIL missing {args.proposals}")
        print("     Not committed by design; regenerate with "
              "tools/extract_proposals.py")
        sys.exit(1)

    # pass 1: positives by (area, scene); negatives by (area, scene, row)
    pos_area, pos_scene = [], []
    neg_area, neg_scene, neg_row, neg_video = [], [], [], []
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
                pos_scene.append(row[ix["scene"]])
            elif lab == "negative":
                neg_area.append(float(row[ix["area_frac"]]))
                neg_scene.append(row[ix["scene"]])
                neg_video.append(row[ix["video"]])
                neg_row.append(n)

    pos_area = np.asarray(pos_area)
    neg_area = np.asarray(neg_area)
    neg_row = np.asarray(neg_row, dtype=np.int64)
    pos_scene = np.asarray(pos_scene)
    neg_scene = np.asarray(neg_scene)
    neg_video = np.asarray(neg_video)
    if pos_area.size == 0 or neg_area.size == 0:
        print(f"FAIL no positives or negatives in split '{args.split}'")
        sys.exit(1)

    edges = np.percentile(pos_area, np.linspace(0, 100, n_bands + 1))
    inner = edges[1:-1]
    pos_band = np.digitize(pos_area, inner, right=False)
    neg_band = np.digitize(neg_area, inner, right=False)
    scenes = sorted(set(pos_scene) | set(neg_scene))

    L = ["TRAINING NEGATIVE DRAW, stratified by size band x scene",
         f"split {args.split}, ratio {ratio}x, {n_bands} bands, "
         f"{len(scenes)} scenes, seeds {seeds}",
         f"positives {pos_area.size:,}, coverage-0 pool {neg_area.size:,}",
         "",
         "Quota per cell is ratio x positives in that cell, so the negative "
         "size AND scene profiles match the positive ones by construction. "
         "Band edges are global deciles of the positive normalised area.",
         "",
         "   band  scene            positives   quota   available   drawn  "
         " short"]

    cells, quota_tot, drawn_tot = [], 0, 0
    for b in range(n_bands):
        for sc in scenes:
            pmask = (pos_band == b) & (pos_scene == sc)
            npos = int(pmask.sum())
            if npos == 0:
                continue
            idx = np.flatnonzero((neg_band == b) & (neg_scene == sc))
            q = npos * ratio
            drawable = min(q, idx.size)
            cells.append({"band": b, "scene": sc, "positives": npos,
                          "quota": q, "available": int(idx.size),
                          "idx": idx})
            quota_tot += q
            drawn_tot += drawable
            L.append(f"   {b:>4}  {sc:<15} {npos:>9,}  {q:>6,}  "
                     f"{idx.size:>10,}  {drawable:>6,}  {max(0, q - drawable):>6,}")

    L.append("")
    L.append(f"   TOTAL quota {quota_tot:,}, drawable {drawn_tot:,}, "
             f"shortfall {quota_tot - drawn_tot:,}")
    if quota_tot != drawn_tot:
        L.append("   SHORTFALL RECORDED AND NOT BACKFILLED. A deficit is not "
                 "redistributed to other cells, because that would undo the "
                 "profile matching the stratification exists to produce.")
    L.append("")

    chosen = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        picks = [rng.choice(c["idx"], size=min(c["quota"], c["idx"].size),
                            replace=False)
                 for c in cells if c["idx"].size]
        sel = np.sort(np.concatenate(picks)) if picks else np.array([], int)
        chosen[seed] = sel
        L.append(f"   seed {seed}: {sel.size:,} negatives drawn")

    # concentration check, the reason for this change
    L.append("")
    L.append("CONCENTRATION BY VIDEO, seed "
             f"{seeds[0]}. Band-only stratification put 56 per cent in two "
             "videos.")
    v, c = np.unique(neg_video[chosen[seeds[0]]], return_counts=True)
    order = np.argsort(-c)
    top = c[order][:5].sum() / c.sum()
    for i in order[:8]:
        L.append(f"   {v[i]:<44} {c[i]:>7,}  {c[i] / c.sum():>6.2%}")
    L.append(f"   top two {c[order][:2].sum() / c.sum():.2%}, "
             f"top five {top:.2%}, across {v.size} videos")
    L.append("")
    L.append("SCENE PROFILE, positives against negatives")
    for sc in scenes:
        pf = (pos_scene == sc).mean()
        nf = (neg_scene[chosen[seeds[0]]] == sc).mean()
        L.append(f"   {sc:<15} positives {pf:>7.2%}   negatives {nf:>7.2%}")

    # pass 2: emit
    want = {s: set(neg_row[chosen[s]].tolist()) for s in seeds}
    files = {s: open(os.path.join(args.out, f"negatives_seed{s}.csv"),
                     "w", newline="") for s in seeds}
    writers = {s: csv.writer(files[s]) for s in seeds}
    with open(args.proposals, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        for s in seeds:
            writers[s].writerow(header)
        for n, row in enumerate(rd):
            for s in seeds:
                if n in want[s]:
                    writers[s].writerow(row)
    for s in seeds:
        files[s].close()

    union = set().union(*want.values())
    meta = {
        "split": args.split, "ratio": ratio, "bands": n_bands,
        "stratified_by": ["size_band", "scene"],
        "band_edge_rule": "global deciles of the positive normalised area",
        "seeds": list(seeds),
        "positives": int(pos_area.size),
        "coverage_zero_pool": int(neg_area.size),
        "quota_total": int(quota_tot), "drawn_total": int(drawn_tot),
        "shortfall_total": int(quota_tot - drawn_tot),
        "cells": [{k: (int(c[k]) if k != "scene" else c[k])
                   for k in ("band", "scene", "positives", "quota",
                             "available")} for c in cells],
        "per_seed_drawn": {str(s): int(chosen[s].size) for s in seeds},
        "union_across_seeds": len(union),
        "note": "Coordinates only. Negatives from the coverage-0 pool only. "
                "No hard negative mining. Shortfalls never backfilled.",
    }
    with open(os.path.join(args.out, "negatives_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    txt = "\n".join(L)
    with open(os.path.join(args.report, "negative_draw.txt"), "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {args.out}/negatives_seed*.csv")


if __name__ == "__main__":
    main()
