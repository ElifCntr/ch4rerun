#!/usr/bin/env python3
"""
Collate the classifier matrix into the tables the chapter needs.

Reads runs/<arm>_seed<n>/summary.json and epochs.csv. Reports whatever is
present, so it can run part-way through the matrix, and says plainly which
runs are missing rather than quietly averaging over fewer seeds.

WHAT IT PRODUCES

  1. THE PROPOSAL CEILING FIRST, per partition and per scene, before any arm
     number appears. Ruled: recall at the proposal stage is the limit every
     arm inherits, so it leads rather than follows.

  2. Per-arm AP as mean and spread across seeds, beside the ceiling and the
     headroom actually used. An arm at 0.97 against a ceiling of 0.98 has
     almost nothing left to win; the same 0.97 against 0.80 would be
     impossible. Reporting AP alone hides which case you are in.

  3. THE APPENDIX RUN TABLE: best epoch, stopping epoch, and per-run wall
     clock, so the learning-rate schedule's inertness is visible rather than
     confessed. The schedule steps at epochs 10 and 20; if runs stop before
     20 the second step never fires, and this table is where a reader sees
     that.

  4. THE SUBSET-VERSUS-FULL AP GAP per run, the measured bias check on
     early-stopping selection. Both numbers already exist per run.

  4a. THE FLIP REGIME PER RUN, constant or coin-toss. The horizontal flip was
     a constant until 008b7bf, so runs trained before it flipped every
     tubelet while runs trained after it honoured hflip_p as the ruled coin
     toss. Which regime a run belongs to is a property of the run and belongs
     beside it, not in prose a reader has to carry. The column says only
     which regime; any comparison caveat lives in the chapter.

  5. Per-scene AP against per-scene ceiling, which is where the proposal-
     limited and classifier-limited scenes separate.

NO SIGNIFICANCE TESTING. Three seeds support a spread, not a p-value, and
the arms are not independent samples of anything. The spread is reported and
the reader judges.

Usage:
    python tools/collate_runs.py
"""

import argparse
import csv
import json
import os
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARMS = ["2d_single", "2d_tframe_logitavg", "tsm", "r3d_18"]

# The horizontal flip was drawn through a freshly constructed torch.Generator()
# and so returned the same value every call: every training tubelet was
# flipped. Fixed in 008b7bf, committed at the timestamp below. A run trained
# before it is CONSTANT, a run trained after it is COIN-TOSS, and the regime is
# read from the mtime of the run's epochs.csv, which is when its training loop
# last wrote. This is the same basis the 30 August ruling was made on.
FLIP_FIX = datetime(2026, 8, 26, 1, 46, 36,
                    tzinfo=timezone(timedelta(hours=1))).timestamp()

# Ruled 30 August 2026: the standing pretrained matrix under runs/ is constant
# flip throughout. Asserted rather than assumed, because a disagreement means
# either the derivation or the ruling is wrong and a person has to say which.
RULED_CONSTANT_RUNS = "runs"


def load_runs(root, seeds):
    runs = {}
    missing = []
    for arm in ARMS:
        for s in seeds:
            d = Path(root) / f"{arm}_seed{s}"
            sp = d / "summary.json"
            if not sp.exists():
                missing.append(f"{arm} seed {s}")
                continue
            with open(sp) as fh:
                r = json.load(fh)
            ep = d / "epochs.csv"
            r["_epochs"] = list(csv.DictReader(open(ep))) if ep.exists() else []
            if not ep.exists():
                print(f"FAIL {ep} is missing, so the flip regime of "
                      f"{arm} seed {s} cannot be determined")
                sys.exit(1)
            r["_flip"] = ("coin-toss" if ep.stat().st_mtime > FLIP_FIX
                          else "constant")
            runs[(arm, s)] = r
    return runs, missing


def assert_ruled_regime(root, runs):
    """The 30 August ruling recorded runs/ as constant flip throughout. Exit
    non-zero with the diff if the derivation disagrees."""
    if Path(root).name != RULED_CONSTANT_RUNS:
        return
    wrong = {f"{arm} seed {s}": r["_flip"]
             for (arm, s), r in runs.items() if r["_flip"] != "constant"}
    if wrong:
        print("FAIL the 30 August ruling records runs/ as constant flip "
              "throughout, and the derivation disagrees:")
        for k, v in sorted(wrong.items()):
            print(f"  {k}: ruled constant, derived {v}")
        print(f"  cutoff: 008b7bf at "
              f"{datetime.fromtimestamp(FLIP_FIX).isoformat()}")
        sys.exit(1)


def spread(xs):
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.4f} (1 seed)"
    return (f"{st.mean(xs):.4f} +/- {st.stdev(xs):.4f} "
            f"[{min(xs):.4f}, {max(xs):.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    runs, missing = load_runs(args.runs, seeds)
    if not runs:
        print(f"FAIL no summary.json found under {args.runs}/")
        sys.exit(1)
    assert_ruled_regime(args.runs, runs)

    L = ["CHAPTER 4 CLASSIFIER MATRIX",
         f"{len(runs)} of {len(ARMS) * len(seeds)} runs present.", ""]
    if missing:
        L.append("MISSING RUNS, averages below cover fewer seeds:")
        for m in missing:
            L.append(f"  {m}")
        L.append("")

    any_run = next(iter(runs.values()))

    # 1. the ceiling, first
    L.append("1. PROPOSAL RECALL, THE CEILING EVERY ARM INHERITS")
    L.append("   No classifier can exceed this. It is a property of the "
             "motion proposals, measured before any arm was trained.")
    L.append(f"   val, pooled      {any_run['val_proposal_ceiling']:.4f}  "
             f"over {any_run['val_gt_boxes']:,} ground-truth boxes")
    for g, d in sorted(any_run.get("per_scene", {}).items()):
        if d.get("ceiling") is not None:
            L.append(f"   val, {g:<12} {d['ceiling']:.4f}  "
                     f"over {d['n_gt']:,}")
    L.append("")

    # 2. per-arm AP
    L.append("2. VAL AP BY ARM, full uncapped pass at the best checkpoint")
    L.append("   arm                  AP across seeds                       "
             "ceiling  headroom used")
    ceil = any_run["val_proposal_ceiling"]
    for arm in ARMS:
        xs = [runs[(arm, s)]["val_ap_full"] for s in seeds if (arm, s) in runs]
        if not xs:
            continue
        used = st.mean(xs) / ceil if ceil else float("nan")
        L.append(f"   {arm:<20} {spread(xs):<38} {ceil:.4f}   {used:.1%}")
    L.append("")

    # 5. per scene
    L.append("3. VAL AP BY SCENE, mean across seeds, against each ceiling")
    scenes = sorted(any_run.get("per_scene", {}))
    L.append("   arm                  " + "  ".join(f"{s:>16}" for s in scenes))
    L.append("   ceiling              " + "  ".join(
        f"{any_run['per_scene'][s]['ceiling']:>16.4f}" for s in scenes))
    for arm in ARMS:
        cells = []
        for sc in scenes:
            xs = [runs[(arm, s)]["per_scene"][sc]["ap"]
                  for s in seeds
                  if (arm, s) in runs
                  and runs[(arm, s)]["per_scene"].get(sc, {}).get("ap")
                  is not None]
            cells.append(f"{st.mean(xs):>16.4f}" if xs else f"{'n/a':>16}")
        if any("n/a" not in c for c in cells):
            L.append(f"   {arm:<20} " + "  ".join(cells))
    L.append("   A small gap to the ceiling means the scene is "
             "PROPOSAL-limited; a large one means it is CLASSIFIER-limited.")
    L.append("")

    # 3 and 4. appendix run table
    L.append("4. APPENDIX RUN TABLE")
    L.append("   The schedule steps at epochs 10 and 20. A run stopping "
             "before 20 never reaches the second step; this table is where "
             "that is visible.")
    L.append("   arm                  seed  best  stop  subset AP  full AP  "
             "  gap   plateau_sd   train_s   eval_s  flip")
    for arm in ARMS:
        for s in seeds:
            r = runs.get((arm, s))
            if not r:
                continue
            gap = r["best_subset_ap"] - r["val_ap_full"]
            tr = sum(float(e["train_s"]) for e in r["_epochs"])
            ev = sum(float(e["eval_s"]) for e in r["_epochs"])
            # Spread of subset AP over the last five epochs. Once a run has
            # fitted the training set, val AP moves on its own, so an epoch
            # selected as best is close to arbitrary. A plateau spread near
            # the arm's seed spread means the checkpoint was a draw rather
            # than a decision.
            tail = [float(e["subset_ap"]) for e in r["_epochs"]][-5:]
            psd = st.stdev(tail) if len(tail) > 1 else 0.0
            L.append(f"   {arm:<20} {s:>4}  {r['best_epoch']:>4}  "
                     f"{r['epochs_run']:>4}  {r['best_subset_ap']:>9.4f}  "
                     f"{r['val_ap_full']:>7.4f}  {gap:>+7.4f}  "
                     f"{psd:>10.4f}  {tr:>8.0f}  {ev:>8.0f}  "
                     f"{r['_flip']}")
    L.append("")
    gaps = [r["best_subset_ap"] - r["val_ap_full"] for r in runs.values()]
    if gaps:
        L.append(f"   SUBSET BIAS: mean gap {st.mean(gaps):+.4f}"
                 + (f", sd {st.stdev(gaps):.4f}" if len(gaps) > 1 else "")
                 + f", range [{min(gaps):+.4f}, {max(gaps):+.4f}]")
        L.append("   A gap near zero means the subset was not a biased "
                 "estimate of the full pass. It is reported per run rather "
                 "than assumed.")
    L.append("")
    L.append("   FLIP REGIME: constant means every training tubelet was "
             "flipped, the defect fixed in 008b7bf; coin-toss means hflip_p "
             "was honoured. Read from each run's epochs.csv mtime against "
             "that commit.")
    L.append("")

    reached = [r for r in runs.values() if r["epochs_run"] >= 20]
    L.append(f"   Runs reaching the second learning-rate step: "
             f"{len(reached)} of {len(runs)}")
    tot = sum(sum(float(e["train_s"]) + float(e["eval_s"])
                  for e in r["_epochs"]) for r in runs.values())
    L.append(f"   MEASURED WALL CLOCK across {len(runs)} runs: "
             f"{tot / 3600:.1f} hours of training and subset evaluation, "
             "excluding the final full passes.")

    txt = "\n".join(L)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "matrix_results.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
