#!/usr/bin/env python3
"""
Size the ruled storage plan.

Two figures are needed. Free disk is a shell command. This gives the
other one, per-split proposal counts, and adds a measurement that may remove
the frame store entirely.

WHAT IT REPORTS

  1. PER-SPLIT PROPOSAL COUNTS at each candidate normalised lower limit, from
     the committed histogram. Train and val are measured. Test is NOT run
     here; under the ruled plan test stores no crops, so its proposal count
     affects evaluation time rather than disk.

  2. TRAINING MATERIALISATION at each limit: positives from measured recall,
     capped negatives at several ratios, tubelets and GB. This is the only
     thing written to disk under the ruling.

  3. FRAME STORE SIZE for val and test, raw and at a lossless estimate, from
     committed frame counts and resolutions. No test access beyond the
     already-committed item A inventory.

  4. DECODE-ONLY BENCHMARK. The 5.7 fps that ruled out cropping at load time
     was the subtractor plus connected components plus a Python matching loop,
     NOT decode. If plain sequential decode is fast enough, evaluation needs a
     rolling 8-frame buffer and NO FRAME STORE AT ALL. This measures it.

Nothing is chosen here.

Usage:
    python tools/size_storage.py --data-root <path>
    python tools/size_storage.py --data-root <path> --decode-videos 3
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

FRAC_GRID = [0.0, 2e-6, 5e-6, 7.23e-6, 1.2e-5, 2e-5, 3.5e-5, 6e-5, 1e-4]
# measured recall at coverage 0.5, variant none, from bgs_operating_curves.txt
RECALL_AT = {0.0: 0.7623, 2e-6: 0.7623, 5e-6: 0.7601, 7.23e-6: 0.7455,
             1.2e-5: 0.7126, 2e-5: 0.6559, 3.5e-5: 0.5518, 6e-5: 0.4068,
             1e-4: 0.2930}
NEG_RATIOS = [1, 2, 3]
TUBELET_BYTES = 8 * 112 * 112 * 3          # T=8, 112x112, 3 channels, uint8
TRAIN_BOXES = 18624
SEC_PER_TUBELET_PER_EPOCH = 0.170
LOSSLESS_FRACTION = 0.55                   # PNG on natural footage, ESTIMATE


def gb(b):
    return b / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--hist", default="reports/bgs_pass/hist.csv")
    ap.add_argument("--inventory", default="reports/video_inventory.csv")
    ap.add_argument("--splits", default="data/dvb_splits_v2.0-static.csv")
    ap.add_argument("--variant", default="none")
    ap.add_argument("--decode-videos", type=int, default=2)
    ap.add_argument("--decode-frames", type=int, default=400)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    for p in (args.hist, args.inventory, args.splits):
        if not os.path.exists(p):
            print(f"FAIL missing {p}")
            sys.exit(1)

    # split map
    smap = {}
    with open(args.splits, newline="") as fh:
        rd = csv.DictReader(fh)
        for c in ("video", "split"):
            if c not in (rd.fieldnames or []):
                print(f"FAIL {args.splits} has no '{c}' column")
                print(f"     present: {rd.fieldnames}")
                sys.exit(1)
        for r in rd:
            smap[r["video"]] = r["split"]

    # inventory: frames and resolution per video
    inv = {}
    with open(args.inventory, newline="") as fh:
        for r in csv.DictReader(fh):
            inv[r["video"]] = {
                "frames": int(float(r["frames"])),
                "w": int(float(r["cv2_width"])),
                "h": int(float(r["cv2_height"])),
                "path": r["path"],
            }

    # histogram, per split
    bins = defaultdict(list)          # split -> (lo, count, frame_area)
    frames_seen = defaultdict(dict)   # split -> video -> frames
    with open(args.hist, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["variant"] != args.variant:
                continue
            sp = r["split"]
            w, h = (float(x) for x in r["resolution"].split("x"))
            bins[sp].append((float(r["bin_lo"]), int(r["count"]), w * h))
            frames_seen[sp][r["video"]] = int(r["frames_processed"])

    L = ["STORAGE SIZING for the ruled plan",
         f"Variant {args.variant}. Proposal counts measured on train and val "
         "only; test stores no crops under the ruling.",
         ""]

    def proposals(split, frac):
        return sum(c for lo, c, fa in bins[split] if lo >= frac * fa)

    L.append("1. PROPOSAL COUNTS PER SPLIT, at each normalised lower limit")
    L.append("   min_frac      train         val    train/frame   val/frame")
    tr_f = sum(frames_seen.get("train", {}).values())
    va_f = sum(frames_seen.get("val", {}).values())
    for f in FRAC_GRID:
        tp, vp = proposals("train", f), proposals("val", f)
        L.append(f"   {f:>9.3g}  {tp:>11,}  {vp:>10,}  "
                 f"{tp / tr_f if tr_f else 0:>12.1f}  "
                 f"{vp / va_f if va_f else 0:>10.1f}")
    L.append(f"   train frames {tr_f:,}   val frames {va_f:,}")
    L.append("")

    L.append("2. TRAINING MATERIALISATION, the only crops written to disk")
    L.append(f"   Positives = {TRAIN_BOXES:,} train boxes x measured recall "
             "at coverage 0.5")
    L.append("   min_frac    recall   positives   " +
             "   ".join(f"neg x{k}: tubelets / GB / h-per-epoch"
                        for k in [2]))
    for f in FRAC_GRID:
        rec = RECALL_AT[f]
        pos = TRAIN_BOXES * rec
        cells = []
        for k in NEG_RATIOS:
            n = pos * (1 + k)
            cells.append(f"x{k}: {n:>8,.0f} / {gb(n * TUBELET_BYTES):>6.1f} GB"
                         f" / {SEC_PER_TUBELET_PER_EPOCH * n / 3600:>4.2f} h")
        L.append(f"   {f:>9.3g}  {rec:>7.4f}  {pos:>10,.0f}   " +
                 "   ".join(cells))
    L.append("   h-per-epoch is for the WHOLE twelve-run matrix at T=8.")
    L.append("")

    L.append("3. FRAME STORE for val and test, if one is used at all")
    by_split = defaultdict(lambda: {"frames": 0, "bytes": 0, "vids": 0})
    for v, sp in smap.items():
        if v not in inv:
            continue
        d = inv[v]
        by_split[sp]["frames"] += d["frames"]
        by_split[sp]["bytes"] += d["frames"] * d["w"] * d["h"] * 3
        by_split[sp]["vids"] += 1
    for sp in ("train", "val", "test"):
        if sp not in by_split:
            continue
        d = by_split[sp]
        L.append(f"   {sp:<6} {d['vids']:>3} videos  {d['frames']:>7,} frames"
                 f"   raw {gb(d['bytes']):>8.1f} GB"
                 f"   lossless est {gb(d['bytes'] * LOSSLESS_FRACTION):>7.1f} GB")
    vt = gb((by_split["val"]["bytes"] + by_split["test"]["bytes"]))
    L.append(f"   VAL + TEST raw {vt:.1f} GB, lossless estimate "
             f"{vt * LOSSLESS_FRACTION:.1f} GB")
    L.append("   The lossless fraction is an ESTIMATE for PNG on natural "
             "footage, not a measurement. Verify on one video before "
             "committing disk.")
    L.append("")

    L.append("4. DECODE-ONLY BENCHMARK")
    L.append("   If sequential decode is fast enough, evaluation runs from a "
             "rolling 8-frame buffer and NO FRAME STORE IS NEEDED.")
    picks = [v for v in sorted(inv) if smap.get(v) in ("val", "train")]
    picks = picks[:args.decode_videos]
    total_f, total_t = 0, 0.0
    for v in picks:
        path = os.path.join(args.data_root, inv[v]["path"])
        if not os.path.exists(path):
            path = os.path.join(args.data_root, "videos",
                                os.path.basename(inv[v]["path"]))
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            L.append(f"   could not open {v}, skipped")
            continue
        n, t0 = 0, time.time()
        while n < args.decode_frames:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        dt = time.time() - t0
        cap.release()
        total_f += n
        total_t += dt
        L.append(f"   {v:<44} {inv[v]['w']}x{inv[v]['h']}  {n:>4} frames  "
                 f"{dt:>6.2f} s  {n / dt if dt else 0:>7.1f} fps")
    if total_t:
        fps = total_f / total_t
        L.append(f"   DECODE ONLY: {fps:.1f} fps")
        for sp in ("val", "test"):
            if sp in by_split:
                secs = by_split[sp]["frames"] / fps
                L.append(f"   one streaming pass over {sp}: "
                         f"{secs / 60:.1f} min at this rate")
        L.append("   Compare against 5.7 fps for the full subtractor pass, "
                 "which included connected components and a Python matching "
                 "loop and is NOT the decode cost.")

    txt = "\n".join(L)
    path = os.path.join(args.out, "storage_sizing.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
