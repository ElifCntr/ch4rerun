#!/usr/bin/env python3
"""
Val early-stopping subset. Item 4, ruled 18 August.

Draws the frame subset used for the per-epoch early-stopping signal. The FULL
uncapped val pass still runs once per run at the end, so every reported val
number is uncapped. This affects only the signal that decides when to stop.

THE FOUR CONDITIONS, as ruled.
  1. Sample FRAMES, not proposals. Proposals within a frame are correlated and
     a tubelet needs its neighbouring frames anyway.
  2. Drawn ONCE, seeded, and committed. Identical across all four arms and all
     three seeds, or early stopping differs by arm and contaminates the
     comparison. That is why the seed here is fixed and is NOT the run seed.
  3. STRATIFIED BY VIDEO. 2,524 of val's 5,259 boxes come from
     2019_11_14_C0001_3922_matrice alone, so an unstratified draw would be
     mostly that one video.
  4. Target 10 per cent. If stratification cannot hit it cleanly, take the
     nearest achievable and REPORT IT, which this script does.

WINDOW VALIDITY. A tubelet at T=8 stride 1 spans centre-3 to centre+4, the
ruled asymmetric convention, so a frame can only be a centre if both ends
exist. Frames outside that range are excluded and the count is reported.

SPLIT ACCESS goes through src.splits.load_split, per the standing policy of
18 August. The first version of this script read the split CSV directly, which
is the same bypass that caused the decode-benchmark test access. It requests
["val"] only and never passes allow_test, so requesting test would raise.

Usage:
    python tools/build_val_subset.py
    python tools/build_val_subset.py --fraction 0.10 --seed 20260818
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

from src.splits import load_split  # noqa: E402

T = 8                    # ruled 12 Aug
LEAD = (T - 1) // 2      # 3, so the window runs centre-3 to centre+4
TRAIL = T - 1 - LEAD     # 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits",
                    default="data/splits/dvb_splits_v2.0-static.csv")
    ap.add_argument("--inventory", default="reports/video_inventory.csv")
    ap.add_argument("--boxes", default="reports/instances_boxes.csv")
    ap.add_argument("--fraction", type=float, default=0.10,
                    help="ruled 18 Aug")
    ap.add_argument("--seed", type=int, default=20260818,
                    help="FIXED. Not the run seed. The subset is identical "
                         "across all arms and all run seeds by design")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--report", default="reports")
    args = ap.parse_args()

    # val only; allow_test is never passed, so a test request would raise
    part = load_split(args.splits, ["val"])
    val_videos = {r["video"] for r in part["val"]}
    if not val_videos:
        print("FAIL src.splits.load_split returned no val videos")
        sys.exit(1)

    frames = {}
    with open(args.inventory, newline="") as fh:
        for r in csv.DictReader(fh):
            frames[r["video"]] = int(float(r["frames"]))

    boxes = defaultdict(lambda: defaultdict(int))
    with open(args.boxes, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["video"] in val_videos:
                boxes[r["video"]][int(r["frame"])] += 1

    rng = np.random.default_rng(args.seed)
    chosen, rows = {}, []
    for v in sorted(val_videos):
        n = frames.get(v)
        if n is None:
            print(f"FAIL {v} missing from {args.inventory}")
            sys.exit(1)
        valid = np.arange(LEAD, n - TRAIL)
        want = max(1, min(int(round(len(valid) * args.fraction)), len(valid)))
        pick = np.sort(rng.choice(valid, size=want, replace=False))
        chosen[v] = pick
        rows.extend({"video": v, "frame": int(f)} for f in pick)

    total_frames = sum(frames[v] for v in val_videos)
    total_valid = sum(max(0, frames[v] - LEAD - TRAIL) for v in val_videos)
    total_pick = sum(len(p) for p in chosen.values())
    pos_all = sum(sum(boxes[v].values()) for v in val_videos)
    pos_pick = sum(int(boxes[v].get(int(f), 0))
                   for v in val_videos for f in chosen[v])

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)
    sub_path = os.path.join(args.out, "val_earlystop_subset.csv")
    with open(sub_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "frame"])
        w.writeheader()
        w.writerows(rows)

    L = ["VAL EARLY-STOPPING SUBSET",
         f"seed {args.seed} (fixed, not a run seed), target fraction "
         f"{args.fraction:.0%}, T={T} lead {LEAD} trail {TRAIL}",
         "split access via src.splits.load_split, partitions ['val'], "
         "allow_test never passed",
         "",
         "PER VIDEO",
         "   video                                     frames   valid   "
         "picked    pct    boxes  boxes_in_subset"]
    for v in sorted(val_videos):
        n = frames[v]
        valid = max(0, n - LEAD - TRAIL)
        k = len(chosen[v])
        b = sum(boxes[v].values())
        bp = sum(int(boxes[v].get(int(f), 0)) for f in chosen[v])
        L.append(f"   {v:<40} {n:>7} {valid:>7} {k:>8} "
                 f"{k / valid if valid else 0:>6.1%} {b:>8} {bp:>16}")
    L.append("")
    L.append(f"   TOTAL frames {total_frames:,}, valid centres "
             f"{total_valid:,}, picked {total_pick:,}")
    L.append(f"   ACHIEVED FRACTION {total_pick / total_valid:.4%} of valid "
             f"centres, {total_pick / total_frames:.4%} of all val frames")
    L.append(f"   Target was {args.fraction:.2%}. "
             + ("On target." if abs(total_pick / total_valid - args.fraction)
                < 0.005 else
                "NEAREST ACHIEVABLE under per-video stratification, reported "
                "as ruled."))
    L.append("")
    L.append(f"   VAL BOXES total {pos_all:,}, in subset {pos_pick:,} "
             f"({pos_pick / pos_all if pos_all else 0:.2%})")
    L.append("   Positives per evaluation will be this box count times "
             "proposal recall. Using the pooled train+val figure of 0.7455 "
             "gives about "
             f"{pos_pick * 0.7455:.0f}, against the roughly 392 the ruling "
             "anticipated. VAL'S OWN RECALL MAY DIFFER, since val is not a "
             "scaled-down train at the proposal stage, so treat this as an "
             "estimate until the first real evaluation.")
    L.append("")
    L.append("   Frames excluded at each video's ends because a T=8 window "
             f"needs centre-{LEAD} to centre+{TRAIL}: "
             f"{total_frames - total_valid:,} across {len(val_videos)} videos.")
    L.append("")
    L.append("THIS SUBSET IS FIXED. It does not vary by arm or by run seed. "
             "Varying it would let early stopping differ by arm and would "
             "contaminate the comparison.")

    txt = "\n".join(L)
    rep_path = os.path.join(args.report, "val_subset.txt")
    with open(rep_path, "w") as fh:
        fh.write(txt)
    with open(os.path.join(args.out, "val_earlystop_subset.json"), "w") as fh:
        json.dump({
            "seed": args.seed, "target_fraction": args.fraction,
            "achieved_fraction_of_valid": total_pick / total_valid,
            "frames_picked": total_pick, "valid_centres": total_valid,
            "all_val_frames": total_frames,
            "val_boxes_total": pos_all, "val_boxes_in_subset": pos_pick,
            "T": T, "lead": LEAD, "trail": TRAIL,
            "stratified_by": "video",
            "split_access": "src.splits.load_split(['val'])",
            "note": "Fixed across all arms and all run seeds by design.",
        }, fh, indent=2)

    print(txt)
    print(f"\nwritten to {sub_path} and {rep_path}")


if __name__ == "__main__":
    main()
