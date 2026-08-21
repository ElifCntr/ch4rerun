#!/usr/bin/env python3
"""
1b-E extraction, stage one: proposal coordinates with labels.

Emits COORDINATES ONLY. No crops are written here; the negative draw runs
offline on this output and crop materialisation follows from the capped list.

NO RULED VALUE IS CARRIED IN THIS FILE. Every parameter comes from the ruled
block of configs/ch4.yaml through src.ruled, which has no defaults and raises
on a missing key. An earlier version hardcoded history, the area limits and
the coverage thresholds here, which contradicted the repository's own rule and
was found by review on 19 August 2026. A silent default is how the original
Chapter 4 pipeline acquired a history of 200 and a min_area of 50 that nobody
had chosen.

WHAT THE RULED VALUES MEAN, for a reader who has the config open.
  bgs.history is non-binding on every clip, since alpha is
    1/min(2*nframes, history) and the longest clip is 3,481 frames, so the
    method is alpha = 1/(2*nframes), a growing-window estimate with no
    forgetting. Declared and described, not tuned.
  bgs.learning_rate is null and is NEVER passed to apply(). Passing any rate
    makes history inert, measured 15 August by tools/validate_mog2.py.
  proposals.min_area_frac is the measured 1080p minimum annotated box area
    fraction. proposals.max_area_frac is a SINGLE frame-area fraction,
    ruled 19 August to sit between two measured anchors: above the largest
    observed best-covering blob and below the frame-0 initialisation blob.

LABELLING.
  POSITIVE  coverage >= proposals.positive_coverage, coverage being
            intersection over ground-truth area, assigned greedily one-to-one.
  NEGATIVE  coverage exactly proposals.negative_coverage, i.e. no overlap.
  IGNORED   between the two, discarded and never trained as background.

READING OF THE ONE-TO-ONE RULE, stated because two rulings meet here. Greedy
one-to-one assigns at most one proposal per ground-truth box. A proposal that
wins an assignment is POSITIVE. One that does not is labelled by its own best
coverage, so 0 gives NEGATIVE and anything above 0 gives IGNORED. A second
blob overlapping an already-matched drone is ignored, never a negative.

WINDOW VALIDITY uses the DECODED frame count, probed per video, not
cv2_frame_count, which overcounts by two on four seaside_cuts clips.

GROUND TRUTH comes from the label files via read_boxes() in
tools/measure_instances.py, the parser that produced instances_boxes.csv.

TEST requires --include-test, which passes allow_test=True and appends to
reports/test_access_log.txt.

Usage:
    python tools/extract_proposals.py --data-root <path> --include-test
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))          # tools/
sys.path.insert(0, str(HERE.parents[1]))      # repo root

from measure_instances import read_boxes  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ruled import require_all, ruled  # noqa: E402
from src.splits import load_split  # noqa: E402

NEEDED = [
    "window.T", "window.padding_fraction",
    "bgs.method", "bgs.history", "bgs.learning_rate",
    "bgs.detect_shadows", "bgs.use_opencl", "bgs.morphology",
    "bgs.warmup_frames",
    "proposals.min_area_frac", "proposals.max_area_frac",
    "proposals.positive_coverage", "proposals.negative_coverage",
    "proposals.assignment",
]


def dotted(cfg, key):
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config key '{key}' not found")
        node = node[part]
    return node


def coverage_matrix(st, gts):
    """rows = blobs, cols = ground-truth boxes. Intersection over GT area."""
    if st.shape[0] == 0 or not gts:
        return np.zeros((st.shape[0], len(gts)))
    bx = st[:, cv2.CC_STAT_LEFT].astype(np.float64)[:, None]
    by = st[:, cv2.CC_STAT_TOP].astype(np.float64)[:, None]
    bw = st[:, cv2.CC_STAT_WIDTH].astype(np.float64)[:, None]
    bh = st[:, cv2.CC_STAT_HEIGHT].astype(np.float64)[:, None]
    g = np.asarray(gts, dtype=np.float64)
    gx, gy, gw, gh = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    ix = np.clip(np.minimum(bx + bw, gx + gw) - np.maximum(bx, gx), 0, None)
    iy = np.clip(np.minimum(by + bh, gy + gh) - np.maximum(by, gy), 0, None)
    area = np.where(gw * gh > 0, gw * gh, np.inf)
    return (ix * iy) / area


def greedy_assign(cov, thr):
    """Greedy one-to-one by coverage. Returns blob index -> gt index."""
    pairs = {}
    if cov.size == 0:
        return pairs
    work = cov.copy()
    while True:
        j = int(np.argmax(work))
        r, c = divmod(j, work.shape[1])
        if work[r, c] < thr:
            break
        pairs[r] = c
        work[r, :] = -1.0
        work[:, c] = -1.0
    return pairs


def open_video(root, rel):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        p = os.path.join(root, "videos", os.path.basename(rel))
    cap = cv2.VideoCapture(p)
    return cap if cap.isOpened() else None


def decoded_length(root, rel):
    cap = open_video(root, rel)
    if cap is None:
        return None
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def make_subtractor(method, history, detect_shadows):
    if method == "mog2":
        return cv2.createBackgroundSubtractorMOG2(
            history=history, detectShadows=detect_shadows)
    if method == "knn":
        return cv2.createBackgroundSubtractorKNN(
            history=history, detectShadows=detect_shadows)
    raise ValueError(f"ruled.bgs.method '{method}' is not supported here")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="overrides data.root from the config")
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--out", default="data/proposals")
    ap.add_argument("--report", default="reports")
    ap.add_argument("--include-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    require_all(cfg, NEEDED)        # fails listing EVERY missing key

    T = ruled(cfg, "window.T")
    lead = (T - 1) // 2
    trail = T - 1 - lead
    history = ruled(cfg, "bgs.history")
    lr = ruled(cfg, "bgs.learning_rate", allow_null=True)
    morph = ruled(cfg, "bgs.morphology", allow_null=True)
    warmup = ruled(cfg, "bgs.warmup_frames")
    min_frac = ruled(cfg, "proposals.min_area_frac")
    max_frac = ruled(cfg, "proposals.max_area_frac")
    pos_cov = ruled(cfg, "proposals.positive_coverage")
    neg_cov = ruled(cfg, "proposals.negative_coverage")
    assign = ruled(cfg, "proposals.assignment")

    if lr is not None:
        print("FAIL ruled.bgs.learning_rate is not null. Passing a rate to "
              "apply() makes history inert (measured 15 Aug). Extraction "
              "will not proceed on a config that sets one.")
        sys.exit(1)
    if morph is not None:
        print(f"FAIL ruled.bgs.morphology is '{morph}'. Morphology was ruled "
              "out on 18 Aug and this script implements none.")
        sys.exit(1)
    if assign != "greedy_one_to_one":
        print(f"FAIL ruled.proposals.assignment is '{assign}', and this "
              "script implements greedy_one_to_one only.")
        sys.exit(1)

    root = args.data_root or dotted(cfg, "data.root")
    root = os.path.expanduser(root)
    ann_dir = Path(root) / dotted(cfg, "data.annotation_dir")
    ann_ext = dotted(cfg, "data.annotation_extension")
    ann_fmt = dotted(cfg, "data.annotation_format")
    splits_file = dotted(cfg, "splits.file")
    if not ann_dir.exists():
        print(f"FAIL annotation directory not found: {ann_dir}")
        sys.exit(1)

    parts = ["train", "val"] + (["test"] if args.include_test else [])
    part = load_split(splits_file, parts, allow_test=args.include_test)
    split_of = {r["video"]: p for p, rows in part.items() for r in rows}
    scene_of = {r["video"]: r.get("scene", "") for rows in part.values()
                for r in rows}

    cv2.ocl.setUseOpenCL(bool(ruled(cfg, "bgs.use_opencl")))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)

    inv = {}
    with open(os.path.join(args.report, "video_inventory.csv"),
              newline="") as fh:
        for r in csv.DictReader(fh):
            inv[r["video"]] = {
                "path": r["path"],
                "inventory_frames": int(float(r["frames"])),
                "res": f"{int(float(r['cv2_width']))}x"
                       f"{int(float(r['cv2_height']))}",
            }

    videos = sorted(v for v in split_of if v in inv)
    if not videos:
        print("FAIL no videos matched")
        sys.exit(1)

    path_csv = os.path.join(args.out, "proposals.csv")
    pf = open(path_csv, "w", newline="")
    pw = csv.writer(pf)
    pw.writerow(["video", "split", "scene", "resolution", "frame",
                 "x", "y", "w", "h", "blob_area", "area_frac",
                 "coverage", "label", "gt_index"])

    stats_rows, tot_f, tot_t, mismatches = [], 0, 0.0, []
    for vi, name in enumerate(videos, 1):
        d = inv[name]
        res = d["res"]
        fw, fh_ = (int(x) for x in res.split("x"))
        fa = float(fw * fh_)
        # Both limits are frame-area fractions.
        lo, hi = min_frac * fa, max_frac * fa

        n_dec = decoded_length(root, d["path"])
        if n_dec is None:
            print(f"FAIL cannot open {name}")
            sys.exit(1)
        if n_dec != d["inventory_frames"]:
            mismatches.append({"video": name,
                               "inventory": d["inventory_frames"],
                               "decoded": n_dec})
        first_valid = max(lead, warmup)
        last_valid = n_dec - 1 - trail

        try:
            frames_gt = read_boxes(ann_dir / f"{name}{ann_ext}", ann_fmt)
        except FileNotFoundError:
            print(f"FAIL no annotation file for {name}")
            sys.exit(1)

        cap = open_video(root, d["path"])
        sub = make_subtractor(ruled(cfg, "bgs.method"), history,
                              bool(ruled(cfg, "bgs.detect_shadows")))
        n_pos = n_neg = n_ign = n_raw = 0
        gt_seen = gt_matched = 0
        t0, fi = time.time(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = sub.apply(frame)     # no learningRate, ruled null
            nlab, _, st, _ = cv2.connectedComponentsWithStats(
                (mask > 0).astype(np.uint8), connectivity=8)
            st = st[1:] if nlab > 1 else np.empty((0, 5), dtype=np.int32)
            n_raw += st.shape[0]

            if first_valid <= fi <= last_valid:
                if st.shape[0]:
                    a = st[:, cv2.CC_STAT_AREA].astype(np.float64)
                    st = st[(a >= lo) & (a <= hi)]
                gts = frames_gt.get(fi, [])
                gt_seen += len(gts)
                cov = coverage_matrix(st, gts)
                best = cov.max(axis=1) if cov.size else np.zeros(st.shape[0])
                pairs = greedy_assign(cov, pos_cov)
                gt_matched += len(pairs)
                for k in range(st.shape[0]):
                    if k in pairs:
                        label, gi = "positive", pairs[k]
                        c = float(cov[k, pairs[k]])
                        n_pos += 1
                    elif best[k] <= neg_cov:
                        label, gi, c = "negative", -1, float(best[k])
                        n_neg += 1
                    else:
                        label, gi, c = "ignored", -1, float(best[k])
                        n_ign += 1
                    pw.writerow([
                        name, split_of[name], scene_of.get(name, ""), res, fi,
                        int(st[k, cv2.CC_STAT_LEFT]),
                        int(st[k, cv2.CC_STAT_TOP]),
                        int(st[k, cv2.CC_STAT_WIDTH]),
                        int(st[k, cv2.CC_STAT_HEIGHT]),
                        int(st[k, cv2.CC_STAT_AREA]),
                        round(int(st[k, cv2.CC_STAT_AREA]) / fa, 9),
                        round(c, 6), label, gi])
            fi += 1
        cap.release()
        dt = time.time() - t0
        tot_f += fi
        tot_t += dt

        stats_rows.append({
            "video": name, "split": split_of[name],
            "scene": scene_of.get(name, ""), "resolution": res,
            "inventory_frames": d["inventory_frames"], "decoded_frames": fi,
            "valid_centres": max(0, last_valid - first_valid + 1),
            "raw_blobs": n_raw,
            "positive": n_pos, "negative": n_neg, "ignored": n_ign,
            "gt_boxes_in_valid_window": gt_seen, "gt_matched": gt_matched,
            "gt_recall": round(gt_matched / gt_seen, 6) if gt_seen else None,
            "seconds": round(dt, 1),
        })
        print(f"[{vi}/{len(videos)}] {name:<42} {split_of[name]:<5} "
              f"{fi:>5}f  pos {n_pos:>6}  neg {n_neg:>8}  ign {n_ign:>7}  "
              f"recall {gt_matched / gt_seen if gt_seen else 0:.4f}  "
              f"{dt:>6.1f}s")
    pf.close()

    with open(os.path.join(args.report, "extraction_stats.csv"),
              "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    agg = {}
    for sp in sorted({r["split"] for r in stats_rows}):
        rs = [r for r in stats_rows if r["split"] == sp]
        gs = sum(r["gt_boxes_in_valid_window"] for r in rs)
        agg[sp] = {
            "videos": len(rs), "frames": sum(r["decoded_frames"] for r in rs),
            "positive": sum(r["positive"] for r in rs),
            "negative": sum(r["negative"] for r in rs),
            "ignored": sum(r["ignored"] for r in rs),
            "gt_boxes_in_valid_window": gs,
            "gt_matched": sum(r["gt_matched"] for r in rs),
            "gt_recall": round(sum(r["gt_matched"] for r in rs) / gs, 6)
            if gs else None,
        }

    meta = {
        "ruled_source": f"{args.config} :: ruled",
        "ruled_values": {k: ruled(cfg, k, allow_null=True) for k in NEEDED},
        "ignore_band": f"{neg_cov} < coverage < {pos_cov}",
        "lead": lead, "trail": trail,
        "ground_truth_source": "label files via "
                               "tools/measure_instances.read_boxes",
        "annotation_dir": str(ann_dir), "annotation_format": ann_fmt,
        "window_validity_source": "decoded frame count, probed per video",
        "frame_count_mismatches": mismatches,
        "opencl_in_use": bool(cv2.ocl.useOpenCL()),
        "opencv_version": cv2.__version__,
        "splits": parts, "include_test": bool(args.include_test),
        "frames": tot_f, "seconds": round(tot_t, 1),
        "fps": round(tot_f / tot_t, 2) if tot_t else None,
        "by_split": agg,
    }
    with open(os.path.join(args.out, "extraction_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if args.include_test:
        with open(os.path.join(args.report, "test_access_log.txt"), "a") as fh:
            fh.write(
                f"\n{datetime.now(timezone.utc).date()}  "
                f"tools/extract_proposals.py --include-test\n"
                f"  Authorised test access, 1b-E extraction authorised "
                f"18 Aug 2026. Decoded the test partition, read its "
                f"annotations, and emitted proposal coordinates with labels. "
                f"allow_test=True passed explicitly through "
                f"src.splits.load_split. Parameters from "
                f"{args.config} :: ruled.\n")

    print()
    if mismatches:
        print(f"FRAME COUNT MISMATCHES ({len(mismatches)}), inventory vs "
              f"decoded, window validity used DECODED:")
        for m in mismatches:
            print(f"  {m['video']:<44} {m['inventory']:>6} -> "
                  f"{m['decoded']:>6}")
        print()
    for sp, a in agg.items():
        print(f"{sp:<6} {a['videos']:>2} videos  {a['frames']:>6} frames  "
              f"pos {a['positive']:>7,}  neg {a['negative']:>9,}  "
              f"ign {a['ignored']:>8,}  gt recall {a['gt_recall']}")
    print(f"\n{tot_f} frames in {tot_t:.1f} s = "
          f"{tot_f / tot_t if tot_t else 0:.1f} fps")
    print(f"written to {path_csv} and {args.out}/extraction_meta.json")


if __name__ == "__main__":
    main()
