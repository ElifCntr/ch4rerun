#!/usr/bin/env python3
"""
Final evaluation. One decode pass, all twelve checkpoints.

DEVELOPED AND REHEARSED ON VAL, then frozen and executed ONCE on test. Val is a
full dress rehearsal of the same streaming path, exercising everything except
the data, so a defect surfaces before test is ever touched. Running with
--split test requires --i-am-executing-the-single-test-pass, and the access is
logged.

WHY ONE PASS. Decode is the expensive part and it happens once regardless of
how many models score the frames. Twelve resident models cost a few hundred
megabytes. Twelve separate passes would cost twelve decodes for no gain, and
every per-run number survives this way.

NOTHING IS COMBINED BEFORE THE METRIC. Each run is scored separately and the
arm figure is the mean and spread across its three seeds, which keeps the
seed-spread ruling intact to the last table.

THE OPERATING THRESHOLD is the point of maximum F1 on each run's UNCAPPED VAL
precision-recall curve, selected PER RUN at its selected checkpoint. The
procedure is identical across arms while its outputs are per-run: threshold
selection joins best epoch and stopping epoch as a protocol output. A shared
threshold would be the deviation, importing one arm's score calibration into
another's operating point.

THE PIXEL PATH IS SHARED WITH TRAINING AND VAL. crop_one and _to_tensor are
imported from the modules that produced the training tubelets and the val
numbers, so nothing about how a crop is built can drift between them.

Usage:
    python tools/evaluate_final.py --split val          # rehearsal
    python tools/evaluate_final.py --split test \\
        --i-am-executing-the-single-test-pass
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from materialise_tubelets import crop_one  # noqa: E402
from measure_instances import read_boxes  # noqa: E402
from src import metrics  # noqa: E402
from src.arms import build_arm  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ruled import ruled  # noqa: E402
from src.splits import load_split  # noqa: E402
from src.streaming_eval import (T, LEAD, TRAIL, PAD, CROP, _to_tensor,
                                count_ground_truth, load_proposals)  # noqa
from src.tubelets import LAYOUTS  # noqa: E402

ARMS = ["2d_single", "2d_tframe_logitavg", "tsm", "r3d_18"]


def dotted(cfg, key):
    node = cfg
    for p in key.split("."):
        node = node[p]
    return node


def max_f1_threshold(scores, labels, n_gt_boxes):
    """The point of maximum F1 on the precision-recall curve.

    Recall is against EVERY ground-truth box, including those no proposal
    covered, so the threshold is chosen under the same denominator the
    reported numbers use.
    """
    keep = labels != metrics.IGNORED
    s, lab = np.asarray(scores)[keep], np.asarray(labels)[keep]
    if s.size == 0:
        return {"threshold": float("nan"), "f1": 0.0,
                "precision": 0.0, "recall": 0.0}
    order = np.argsort(-s, kind="stable")
    hit = (lab[order] == metrics.POSITIVE).astype(np.float64)
    tp = np.cumsum(hit)
    fp = np.cumsum(1.0 - hit)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / float(n_gt_boxes)
    f1 = np.where(precision + recall > 0,
                  2 * precision * recall / np.maximum(precision + recall,
                                                      1e-12), 0.0)
    i = int(np.argmax(f1))
    return {"threshold": float(s[order][i]), "f1": float(f1[i]),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "rank": i + 1}


@torch.no_grad()
def evaluate_all(models, layouts, proposals, videos, data_root, means, stds,
                 device, batch=64):
    """Score every proposal with every model in ONE decode pass.

    models maps run tag -> model. Returns per-tag score arrays plus the
    per-proposal metadata, which is shared since every model sees the same
    proposals in the same order.
    """
    for m in models.values():
        m.eval()

    scores = {k: [] for k in models}
    labels, scenes, gt_keys, area_fracs, vids = [], [], [], [], []
    pending, meta = [], []

    def flush():
        if not pending:
            return
        clips = torch.stack(pending)          # (B, T, C, H, W) on cpu
        for tag, model in models.items():
            lay = layouts[tag]
            x = clips
            if lay == "centre":
                x = clips[:, LEAD]
            elif lay == "cthw":
                x = clips.permute(0, 2, 1, 3, 4)
            mean = means[tag].to(device)
            std = stds[tag].to(device)
            xb = x.to(device, non_blocking=True)
            # normalisation is per arm, so it is applied here rather than
            # when the clip was built
            if lay == "centre":
                xb = (xb - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
            elif lay == "cthw":
                xb = (xb - mean.view(1, 3, 1, 1, 1)) / std.view(1, 3, 1, 1, 1)
            else:
                xb = (xb - mean.view(1, 1, 3, 1, 1)) / std.view(1, 1, 3, 1, 1)
            p = torch.softmax(model(xb).float(), dim=1)[:, 1].cpu().numpy()
            scores[tag].extend(p.tolist())
        for m in meta:
            labels.append(m["label"])
            scenes.append(m["scene"])
            gt_keys.append(f'{m["video"]}:{m["centre"]}:{m["gt_index"]}'
                           if m["label"] == "positive" else "")
            area_fracs.append(m["area_frac"])
            vids.append(m["video"])
        pending.clear()
        meta.clear()

    for vi, (name, rel) in enumerate(sorted(videos.items()), 1):
        need = proposals.get(name)
        if not need:
            continue
        path = Path(data_root) / rel
        if not path.exists():
            path = Path(data_root) / "videos" / Path(rel).name
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        buf, i = deque(maxlen=T), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(frame)
            if len(buf) == T:
                centre = i - TRAIL
                for pr in need.get(centre, ()):
                    side = (1.0 + 2.0 * PAD) * max(pr["w"], pr["h"])
                    cx = pr["x"] + pr["w"] / 2.0
                    cy = pr["y"] + pr["h"] / 2.0
                    clip = np.empty((T, CROP, CROP, 3), dtype=np.uint8)
                    for t in range(T):
                        patch, _ = crop_one(buf[t], cx, cy, side)
                        clip[t] = cv2.resize(patch, (CROP, CROP),
                                             interpolation=cv2.INTER_LINEAR)
                    x = torch.from_numpy(clip).float().div_(255.0)
                    pending.append(x.permute(0, 3, 1, 2))
                    meta.append({**pr, "video": name, "centre": centre})
                    if len(pending) >= batch:
                        flush()
            i += 1
        cap.release()
        n = len(next(iter(scores.values())))
        print(f"  [{vi}/{len(videos)}] {name:<42} {i:>5} frames  "
              f"{n:>9,} scored per model")
    flush()

    return ({k: np.asarray(v) for k, v in scores.items()},
            {"labels": np.asarray(labels), "scenes": np.asarray(scenes),
             "gt_key": np.asarray(gt_keys),
             "area_frac": np.asarray(area_fracs),
             "videos": np.asarray(vids)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--i-am-executing-the-single-test-pass",
                    action="store_true", dest="confirm")
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--proposals", default="data/proposals/proposals.csv")
    ap.add_argument("--stats", default="reports/extraction_stats.csv")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm:
        print("FAIL test is evaluated once. Rehearse with --split val until "
              "the script is frozen, then pass "
              "--i-am-executing-the-single-test-pass.")
        sys.exit(1)

    cfg = load_config(args.config)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else ruled(cfg, "training.seeds"))
    Tw = ruled(cfg, "window.T")
    root = Path(dotted(cfg, "data.root")).expanduser()

    # load every checkpoint that exists, and say which do not
    models, layouts, means, stds, thresholds, missing = {}, {}, {}, {}, {}, []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for arm in ARMS:
        for s in seeds:
            tag = f"{arm}_seed{s}"
            ck = Path(args.runs) / tag / "best.pt"
            vs = Path(args.runs) / tag / "val_scores.npz"
            if not ck.exists():
                missing.append(tag)
                continue
            model, _ = build_arm(arm, Tw)
            model.load_state_dict(
                torch.load(ck, map_location="cpu")["model"])
            models[tag] = model.to(device)
            layouts[tag] = LAYOUTS[arm]
            means[tag] = torch.tensor(
                ruled(cfg, f"normalisation.{arm}.mean"))
            stds[tag] = torch.tensor(ruled(cfg, f"normalisation.{arm}.std"))
            # threshold from this run's UNCAPPED VAL scores
            if vs.exists():
                d = np.load(vs, allow_pickle=True)
                # n_gt for val comes from the run's own summary
                with open(Path(args.runs) / tag / "summary.json") as fh:
                    n_gt_val = json.load(fh)["val_gt_boxes"]
                thresholds[tag] = max_f1_threshold(
                    d["scores"], d["labels"], n_gt_val)
            else:
                missing.append(f"{tag} (val_scores.npz)")

    if not models:
        print(f"FAIL no checkpoints found under {args.runs}/")
        sys.exit(1)
    print(f"{len(models)} checkpoints loaded"
          + (f"; MISSING: {missing}" if missing else ""))

    # data
    parts = [args.split]
    part = load_split(dotted(cfg, "splits.file"), parts,
                      allow_test=args.split == "test")
    meta_rows = {r["video"]: r for r in part[args.split]}
    scene_of = {v: r.get("scene", "") for v, r in meta_rows.items()}

    inv = {}
    with open("reports/video_inventory.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            inv[r["video"]] = r["path"]
    videos = {v: inv[v] for v in meta_rows if v in inv}

    decoded = {}
    with open(args.stats, newline="") as fh:
        for r in csv.DictReader(fh):
            decoded[r["video"]] = int(r["decoded_frames"])

    ann_dir = root / dotted(cfg, "data.annotation_dir")
    ann_ext = dotted(cfg, "data.annotation_extension")
    ann_fmt = dotted(cfg, "data.annotation_format")
    frames_gt = {v: read_boxes(ann_dir / f"{v}{ann_ext}", ann_fmt)
                 for v in videos}
    n_gt, n_gt_scene = count_ground_truth(frames_gt, videos, decoded,
                                          None, scene_of)
    props = load_proposals(args.proposals, args.split)
    print(f"{args.split}: {n_gt:,} ground-truth boxes, "
          f"{len(videos)} videos")

    t0 = time.time()
    scores, shared = evaluate_all(models, layouts, props, videos, root,
                                  means, stds, device)
    elapsed = time.time() - t0

    # metrics per run
    results = {}
    for tag, sc in scores.items():
        ap_val, _, _, n_scored, n_ign = metrics.average_precision(
            sc, shared["labels"], n_gt, shared["gt_key"])
        per_scene = metrics.by_group(sc, shared["labels"], shared["scenes"],
                                     n_gt_scene, shared["gt_key"])
        row = {"ap": ap_val, "n_scored": n_scored, "n_ignored": n_ign,
               "per_scene": per_scene,
               "val_threshold": thresholds.get(tag)}
        if tag in thresholds:
            row["at_threshold"] = metrics.recall_at_score(
                sc, shared["labels"], n_gt, thresholds[tag]["threshold"])
        results[tag] = row

    ceiling = metrics.proposal_ceiling(shared["labels"], n_gt)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{args.split}_final_scores.npz",
                        **scores, **shared)
    with open(out / f"{args.split}_final_results.json", "w") as fh:
        json.dump({"split": args.split, "n_gt_boxes": n_gt,
                   "proposal_ceiling": ceiling,
                   "n_gt_by_scene": n_gt_scene,
                   "seconds": round(elapsed, 1),
                   "checkpoints": len(models), "missing": missing,
                   "runs": results}, fh, indent=2, default=float)

    if args.split == "test":
        with open(out / "test_access_log.txt", "a") as fh:
            fh.write(f"\n{datetime.now(timezone.utc).date()}  "
                     f"tools/evaluate_final.py --split test\n"
                     f"  The single test pass. {len(models)} checkpoints "
                     f"scored in one decode pass over {len(videos)} videos, "
                     f"{n_gt:,} ground-truth boxes. Thresholds selected on "
                     f"uncapped val per run, never on test.\n")

    print(f"\n{args.split.upper()}: proposal ceiling {ceiling:.4f} over "
          f"{n_gt:,} boxes, {elapsed / 60:.1f} min for "
          f"{len(models)} checkpoints in one pass")
    for arm in ARMS:
        xs = [results[f"{arm}_seed{s}"]["ap"] for s in seeds
              if f"{arm}_seed{s}" in results]
        if xs:
            m = sum(xs) / len(xs)
            sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 \
                if len(xs) > 1 else 0.0
            print(f"  {arm:<20} AP {m:.4f} +/- {sd:.4f}  over {len(xs)} seeds")
    print(f"written to {out}/{args.split}_final_results.json")


if __name__ == "__main__":
    main()
