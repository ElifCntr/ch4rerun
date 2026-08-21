#!/usr/bin/env python3
"""
Train one arm at one seed. Twelve invocations make the matrix.

EVERY PARAMETER COMES FROM THE RULED BLOCK. Nothing here has a default, and
src/ruled.py raises on a missing key. The schedule is identical for all four
arms by ruling, which guarantees an IDENTICAL PROTOCOL and not per-arm
optimality; the chapter states that distinction and scopes any result to the
matched protocol.

MODEL DEFINITIONS ARE IMPORTED from tools/epoch_timing.py, the same source the
timing coefficients were measured against, so wall-clock predictions and
actual runs describe the same models.

BATCHNORM RUNNING STATISTICS ARE FROZEN during training, following the
small-dataset protocol. Weights still train; only the running estimates are
held. With 13,347 positives against ImageNet-scale pretraining, letting the
statistics drift on a small, heavily imbalanced set is the larger risk.

EARLY STOPPING runs on a FIXED val frame subset, identical across every arm
and seed, so stopping can never differ by arm and contaminate the comparison.
The FULL UNCAPPED val pass runs once at the end, and every reported val number
comes from that pass, not from the subset.

Usage:
    python tools/train_arm.py --arm r3d_18 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from epoch_timing import build_arm  # noqa: E402
from measure_instances import read_boxes  # noqa: E402
from src import metrics, streaming_eval  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ruled import require_all, ruled  # noqa: E402
from src.seeding import seed_everything  # noqa: E402
from src.splits import load_split  # noqa: E402
from src.tubelets import LAYOUTS, TubeletDataset  # noqa: E402

NEEDED = [
    "window.T", "window.batch",
    "training.optimiser", "training.momentum", "training.weight_decay",
    "training.base_lr", "training.lr_steps", "training.lr_gamma",
    "training.epoch_ceiling", "training.patience", "training.seeds",
    "augmentation.hflip_p", "augmentation.brightness",
    "augmentation.contrast", "augmentation.per",
]


def dotted(cfg, key):
    node = cfg
    for p in key.split("."):
        node = node[p]
    return node


def freeze_bn(model):
    """Hold running statistics; weights still train."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--store", default="data/tubelets/tubelets.npy")
    ap.add_argument("--manifest", default="data/tubelets/manifest.csv")
    ap.add_argument("--proposals", default="data/proposals/proposals.csv")
    ap.add_argument("--subset", default="data/splits/val_earlystop_subset.csv")
    ap.add_argument("--stats", default="reports/extraction_stats.csv")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    require_all(cfg, NEEDED)
    if args.seed not in ruled(cfg, "training.seeds"):
        print(f"FAIL seed {args.seed} is not in ruled.training.seeds "
              f"{ruled(cfg, 'training.seeds')}")
        sys.exit(1)
    if ruled(cfg, "augmentation.per") != "tubelet":
        print("FAIL ruled.augmentation.per must be 'tubelet'; per-frame draws "
              "manufacture apparent motion")
        sys.exit(1)

    # normalisation: each backbone's own pretraining statistics, from the
    # config. No defaults here, deliberately.
    mean = ruled(cfg, f"normalisation.{args.arm}.mean")
    std = ruled(cfg, f"normalisation.{args.arm}.std")

    T = ruled(cfg, "window.T")
    batch = ruled(cfg, "window.batch")
    ceiling = ruled(cfg, "training.epoch_ceiling")
    patience = ruled(cfg, "training.patience")

    seed_everything(args.seed, True, False)
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. Stage 2 runs on the P5000.")
    device = torch.device("cuda")

    tag = f"{args.arm}_seed{args.seed}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    train_ds = TubeletDataset(
        args.store, args.manifest, args.seed, args.arm, mean, std,
        augment=True,
        hflip_p=ruled(cfg, "augmentation.hflip_p"),
        brightness=tuple(ruled(cfg, "augmentation.brightness")),
        contrast=tuple(ruled(cfg, "augmentation.contrast")))
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=False)
    print(f"{tag}: {len(train_ds):,} tubelets, "
          f"{train_ds.n_positive:,} positive")

    # val: subset for early stopping, full for the reported numbers
    subset = defaultdict(set)
    with open(args.subset, newline="") as fh:
        for r in csv.DictReader(fh):
            subset[r["video"]].add(int(r["frame"]))
    subset = dict(subset)

    part = load_split(dotted(cfg, "splits.file"), ["val"])
    val_videos_meta = {r["video"]: r for r in part["val"]}
    scene_of = {v: r.get("scene", "") for v, r in val_videos_meta.items()}

    inv = {}
    with open("reports/video_inventory.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            inv[r["video"]] = r["path"]
    videos = {v: inv[v] for v in val_videos_meta if v in inv}

    decoded = {}
    with open(args.stats, newline="") as fh:
        for r in csv.DictReader(fh):
            decoded[r["video"]] = int(r["decoded_frames"])

    root = Path(dotted(cfg, "data.root")).expanduser()
    ann_dir = root / dotted(cfg, "data.annotation_dir")
    ann_ext = dotted(cfg, "data.annotation_extension")
    ann_fmt = dotted(cfg, "data.annotation_format")
    frames_gt = {v: read_boxes(ann_dir / f"{v}{ann_ext}", ann_fmt)
                 for v in videos}

    props_sub = streaming_eval.load_proposals(args.proposals, "val", subset)
    props_all = streaming_eval.load_proposals(args.proposals, "val")
    n_gt_sub, _ = streaming_eval.count_ground_truth(
        frames_gt, videos, decoded, subset, scene_of)
    n_gt_all, n_gt_scene = streaming_eval.count_ground_truth(
        frames_gt, videos, decoded, None, scene_of)
    print(f"val subset: {n_gt_sub:,} ground-truth boxes; "
          f"full val: {n_gt_all:,}")

    model, _ = build_arm(args.arm, T)
    model = model.to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=ruled(cfg, "training.base_lr"),
        momentum=ruled(cfg, "training.momentum"),
        weight_decay=ruled(cfg, "training.weight_decay"))
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=list(ruled(cfg, "training.lr_steps")),
        gamma=ruled(cfg, "training.lr_gamma"))
    crit = nn.CrossEntropyLoss()

    log_path = out / "epochs.csv"
    lf = open(log_path, "w", newline="")
    lw = csv.writer(lf)
    lw.writerow(["epoch", "lr", "train_loss", "subset_ap", "subset_ceiling",
                 "best_ap", "epochs_since_best", "train_s", "eval_s"])

    best_ap, best_epoch, since = -1.0, -1, 0
    for epoch in range(1, ceiling + 1):
        model.train()
        freeze_bn(model)
        t0, tot, n = time.time(), 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * y.numel()
            n += y.numel()
        train_s = time.time() - t0
        sched.step()

        t1 = time.time()
        res = streaming_eval.evaluate(
            model, args.arm, LAYOUTS[args.arm], props_sub, videos,
            root, mean, std, device, progress=False)
        ap, _, _, _, _ = metrics.average_precision(
            res["scores"], res["labels"], n_gt_sub, res["gt_key"])
        ceil_sub = metrics.proposal_ceiling(res["labels"], n_gt_sub)
        eval_s = time.time() - t1

        if ap > best_ap:
            best_ap, best_epoch, since = ap, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "subset_ap": ap, "arm": args.arm, "seed": args.seed},
                       out / "best.pt")
        else:
            since += 1

        lw.writerow([epoch, opt.param_groups[0]["lr"], tot / max(n, 1),
                     ap, ceil_sub, best_ap, since,
                     round(train_s, 1), round(eval_s, 1)])
        lf.flush()
        print(f"  epoch {epoch:>2}  loss {tot / max(n, 1):.4f}  "
              f"subset AP {ap:.4f}  best {best_ap:.4f} (ep {best_epoch})  "
              f"{train_s:.0f}s + {eval_s:.0f}s")

        if since >= patience:
            print(f"  early stop: {patience} epochs without improvement")
            break
    lf.close()

    # reported numbers come from the FULL uncapped pass at the best checkpoint
    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    print(f"full uncapped val pass at epoch {ck['epoch']}")
    res = streaming_eval.evaluate(
        model, args.arm, LAYOUTS[args.arm], props_all, videos,
        root, mean, std, device)
    ap, prec, rec, n_scored, n_ign = metrics.average_precision(
        res["scores"], res["labels"], n_gt_all, res["gt_key"])
    per_scene = metrics.by_group(
        res["scores"], res["labels"], res["scenes"], n_gt_scene,
        res["gt_key"])

    np.savez_compressed(out / "val_scores.npz", **res)
    summary = {
        "arm": args.arm, "seed": args.seed,
        "best_epoch": ck["epoch"], "best_subset_ap": ck["subset_ap"],
        "epochs_run": epoch, "stopped_early": since >= patience,
        "val_ap_full": ap,
        "val_proposal_ceiling": metrics.proposal_ceiling(
            res["labels"], n_gt_all),
        "val_gt_boxes": n_gt_all, "val_scored": n_scored,
        "val_ignored": n_ign,
        "per_scene": per_scene,
        "ruled": {k: ruled(cfg, k, allow_null=True) for k in NEEDED},
        "normalisation": {"mean": mean, "std": std},
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"\n{tag}: full val AP {ap:.4f} against a proposal ceiling of "
          f"{summary['val_proposal_ceiling']:.4f} "
          f"over {n_gt_all:,} ground-truth boxes")
    for g, d in sorted(per_scene.items()):
        if d.get("ap") is not None:
            print(f"  {g:<15} AP {d['ap']:.4f}  ceiling {d['ceiling']:.4f}  "
                  f"n_gt {d['n_gt']:,}")
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
