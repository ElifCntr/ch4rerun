#!/usr/bin/env python3
"""
Background subtractor pass over the static subset, all three methods.

Replaces the 17 August version. Four changes.

  1. --method mog2 | vibe | knn, per the 18 August restoration of the ViBe and
     KNN arms. ViBe is src.vibe.ViBe, seeded and the seed recorded.
  2. MORPHOLOGY DROPPED. Ruled out 18 August, so only the raw mask is
     processed. The variant column is retained so the offline analysis reads
     unchanged. This halves the per-frame work.
  3. MATCHING VECTORISED. The first pass looped over every blob in Python for
     every ground-truth box, which is why noisy videos ran at 1.8 fps and the
     20-minute extrapolation missed by 3.6x. Coverage is now computed over all
     blobs at once.
  4. SPLIT ACCESS THROUGH src.splits.load_split, per the standing policy of
     18 August. The earlier version read the split CSV directly, which is the
     same bypass that caused the decode-benchmark test access. It requests
     ["train", "val"] and never passes allow_test, so a test request raises
     rather than being filtered out by convention.

DUMPS, unchanged so the offline analysis still applies.
  matched.csv  one row per ground-truth box: the best-covering blob's area and
               the coverage achieved. Recall at any lower limit is the
               fraction of boxes whose covering blob is at least that big.
  hist.csv     log-spaced histogram of every blob's area per video, plus the
               frame count. Blobs per frame at any limit is a bin sum.
  meta.json    method, parameters, seed, OpenCL state, throughput.

NOTE, and it matters for what comes next: THIS SCRIPT STORES NO PROPOSAL
COORDINATES. It stores areas and a per-box best match. The negative draw needs
coordinates, so extraction must emit them; this dump cannot supply them.

learningRate is NEVER passed to MOG2. Measured 15 August: passing it makes
history inert.

Usage:
    python tools/run_bgs_pass.py --data-root <path> --method mog2 --history 6960
    python tools/run_bgs_pass.py --data-root <path> --method knn  --history 6960
    python tools/run_bgs_pass.py --data-root <path> --method vibe --seed 20260818
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits import load_split  # noqa: E402

NBINS = 240
VARIANT = "none"          # morphology ruled out 18 Aug; column kept because
                          # the offline analysis filters on it


def bin_edges(frame_area):
    return np.geomspace(1.0, float(frame_area), NBINS + 1)


def best_coverage(stats, gx, gy, gw, gh):
    """Coverage is intersection over GROUND-TRUTH area, as ruled. Vectorised
    over every blob at once. Returns (coverage, blob_area)."""
    if stats.shape[0] == 0:
        return 0.0, 0
    g_area = gw * gh
    if g_area <= 0:
        return 0.0, 0
    bx = stats[:, cv2.CC_STAT_LEFT].astype(np.float64)
    by = stats[:, cv2.CC_STAT_TOP].astype(np.float64)
    bw = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    bh = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    ix = np.clip(np.minimum(bx + bw, gx + gw) - np.maximum(bx, gx), 0, None)
    iy = np.clip(np.minimum(by + bh, gy + gh) - np.maximum(by, gy), 0, None)
    cov = (ix * iy) / g_area
    j = int(np.argmax(cov))
    return float(cov[j]), int(stats[j, cv2.CC_STAT_AREA])


def make_subtractor(method, history, seed):
    if method == "mog2":
        return cv2.createBackgroundSubtractorMOG2(
            history=history, detectShadows=False), None
    if method == "knn":
        return cv2.createBackgroundSubtractorKNN(
            history=history, detectShadows=False), None
    if method == "vibe":
        from src.vibe import ViBe
        v = ViBe(seed=seed)
        return v, (v.describe() if hasattr(v, "describe") else None)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--method", required=True,
                    choices=["mog2", "vibe", "knn"])
    ap.add_argument("--history", type=int, default=None,
                    help="required for mog2 and knn, ignored for vibe")
    ap.add_argument("--seed", type=int, default=None,
                    help="required for vibe, which is non-deterministic")
    ap.add_argument("--inventory", default="reports/video_inventory.csv")
    ap.add_argument("--splits",
                    default="data/splits/dvb_splits_v2.0-static.csv")
    ap.add_argument("--boxes", default="reports/instances_boxes.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--use-opencl", action="store_true")
    args = ap.parse_args()

    if args.method in ("mog2", "knn") and args.history is None:
        print(f"FAIL --history is required for {args.method}, no default")
        sys.exit(1)
    if args.method == "vibe" and args.seed is None:
        print("FAIL --seed is required for vibe, which is non-deterministic")
        sys.exit(1)

    out = args.out or f"reports/bgs_pass_{args.method}"
    os.makedirs(out, exist_ok=True)
    cv2.ocl.setUseOpenCL(bool(args.use_opencl))

    # train and val only; allow_test never passed, so test would raise
    part = load_split(args.splits, ["train", "val"])
    allowed = {r["video"] for rows in part.values() for r in rows}

    inv = {}
    with open(args.inventory, newline="") as fh:
        for r in csv.DictReader(fh):
            inv[r["video"]] = r["path"]

    gt, meta = {}, {}
    with open(args.boxes, newline="") as fh:
        for r in csv.DictReader(fh):
            v = r["video"]
            if v not in allowed:
                continue
            gt.setdefault(v, {}).setdefault(int(r["frame"]), []).append(
                tuple(float(r[k]) for k in ("x", "y", "w", "h")))
            meta.setdefault(v, {"split": r["split"], "scene": r["scene"],
                                "resolution": r["resolution"]})

    videos = sorted(meta)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        print("FAIL no train or val videos matched")
        sys.exit(1)

    mf = open(os.path.join(out, "matched.csv"), "w", newline="")
    hf = open(os.path.join(out, "hist.csv"), "w", newline="")
    mw, hw = csv.writer(mf), csv.writer(hf)
    mw.writerow(["video", "split", "scene", "resolution", "frame", "variant",
                 "gt_x", "gt_y", "gt_w", "gt_h", "gt_area",
                 "best_coverage", "matched_blob_area", "n_blobs_in_frame"])
    hw.writerow(["video", "split", "scene", "resolution", "variant",
                 "frames_processed", "bin_lo", "bin_hi", "count"])

    tot_f, tot_t, describe = 0, 0.0, None
    for vi, name in enumerate(videos, 1):
        path = os.path.join(args.data_root, inv[name])
        if not os.path.exists(path):
            path = os.path.join(args.data_root, "videos",
                                os.path.basename(inv[name]))
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"FAIL cannot open {path}")
            sys.exit(1)

        fw, fh_ = (int(x) for x in meta[name]["resolution"].split("x"))
        edges = bin_edges(fw * fh_)
        counts = np.zeros(NBINS, dtype=np.int64)
        sub, describe = make_subtractor(args.method, args.history, args.seed)

        t0, fi, done = time.time(), 0, 0
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and done >= args.max_frames):
                break
            mask = sub.apply(frame)
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                (mask > 0).astype(np.uint8), connectivity=8)
            st = stats[1:] if n > 1 else np.empty((0, 5), dtype=np.int32)
            if st.shape[0]:
                counts += np.histogram(
                    st[:, cv2.CC_STAT_AREA].astype(np.float64), bins=edges)[0]
            for (gx, gy, gw, gh) in gt.get(name, {}).get(fi, []):
                cov, area = best_coverage(st, gx, gy, gw, gh)
                mw.writerow([name, meta[name]["split"], meta[name]["scene"],
                             meta[name]["resolution"], fi, VARIANT,
                             gx, gy, gw, gh, gw * gh,
                             round(cov, 6), area, int(st.shape[0])])
            fi += 1
            done += 1
        cap.release()
        dt = time.time() - t0
        tot_f += done
        tot_t += dt

        for b in range(NBINS):
            if counts[b]:
                hw.writerow([name, meta[name]["split"], meta[name]["scene"],
                             meta[name]["resolution"], VARIANT, done,
                             round(edges[b], 4), round(edges[b + 1], 4),
                             int(counts[b])])
        print(f"[{vi}/{len(videos)}] {name:<44} {done:>5} frames  "
              f"{dt:>7.1f} s  {done / dt if dt else 0:>6.1f} fps")

    mf.close()
    hf.close()

    fps = tot_f / tot_t if tot_t else 0
    info = {
        "method": args.method,
        "history": args.history if args.method != "vibe" else None,
        "seed": args.seed if args.method == "vibe" else None,
        "vibe_params": describe,
        "morphology": "none (ruled out 18 Aug 2026)",
        "learning_rate": "never passed; OpenCV default -1",
        "detect_shadows": False,
        "opencl_in_use": bool(cv2.ocl.useOpenCL()),
        "opencv_version": cv2.__version__,
        "videos": len(videos), "frames": tot_f,
        "seconds": round(tot_t, 1), "fps": round(fps, 2),
        "split_access": "src.splits.load_split(['train','val'])",
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    print()
    print(f"{args.method}  {tot_f} frames in {tot_t:.1f} s = {fps:.1f} fps")
    print(f"OpenCL in use: {cv2.ocl.useOpenCL()}")
    if args.method == "vibe":
        print(f"ViBe seed {args.seed} recorded in meta.json")
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
