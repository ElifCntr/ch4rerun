"""Streaming evaluation over val or test. Nothing is written to disk.

Each video is decoded ONCE, sequentially, with a rolling T-frame buffer. When
the buffer holds centre-3 to centre+4 every proposal at that centre is cropped
and scored together, so decode cost is per FRAME rather than per proposal.
That is what makes an uncapped evaluation affordable and why no frame store is
needed.

IT REUSES crop_one FROM materialise_tubelets RATHER THAN REIMPLEMENTING IT.
The pixels a model sees at evaluation must be the ones it saw in training,
including the replicate padding at frame edges. A second implementation that
agreed today and drifted tomorrow is exactly the failure the one-parser rule
exists to prevent.

THE DENOMINATOR IS COUNTED AS THE PASS RUNS. Every ground-truth box at an
evaluated centre counts, whether or not a proposal covered it, because a drone
the subtractor never proposed is a miss no classifier can recover. Counting it
here rather than reading a total from elsewhere keeps it exact when a frame
subset is used for early stopping.

BUFFER ARITHMETIC. After reading frame i the buffer holds i-7 to i, so the
centre it completes is i - 4, that is i - TRAIL. The window runs centre-3 to
centre+4 under the ruled asymmetric convention.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from materialise_tubelets import crop_one  # noqa: E402

T = 8
LEAD = (T - 1) // 2
TRAIL = T - 1 - LEAD
PAD = 0.3
CROP = 112


def load_proposals(path, split, frame_subset=None):
    """video -> centre frame -> list of proposal dicts."""
    by = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] != split:
                continue
            v, f = r["video"], int(r["frame"])
            if frame_subset is not None and f not in frame_subset.get(v, ()):
                continue
            by[v][f].append({
                "x": int(r["x"]), "y": int(r["y"]),
                "w": int(r["w"]), "h": int(r["h"]),
                "label": r["label"], "gt_index": int(r["gt_index"]),
                "scene": r["scene"], "area_frac": float(r["area_frac"]),
            })
    return by


def _to_tensor(clip, mean, std, layout):
    """clip is (T, H, W, C) uint8. Mirrors src/tubelets.py exactly, minus the
    augmentation, which never applies at evaluation."""
    x = torch.from_numpy(np.ascontiguousarray(clip)).float().div_(255.0)
    x = x.permute(0, 3, 1, 2)
    x = (x - mean) / std
    if layout == "centre":
        return x[LEAD]
    if layout == "cthw":
        return x.permute(1, 0, 2, 3)
    return x


@torch.no_grad()
def evaluate(model, arm, layout, proposals, videos, data_root, mean, std,
             device, batch=256, progress=True):
    """Score every proposal in `proposals`. videos maps name -> relative path.

    Returns a dict of parallel arrays plus the ground-truth counts the metric
    needs as denominators.
    """
    model.eval()
    mean = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    scores, labels, scenes, gt_idx, area_fracs, vids = [], [], [], [], [], []
    gt_keys = []
    n_gt_total = 0
    n_gt_by_scene = defaultdict(int)
    pending, meta = [], []

    def flush():
        if not pending:
            return
        x = torch.stack(pending).to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        scores.extend(p.tolist())
        for m in meta:
            labels.append(m["label"])
            scenes.append(m["scene"])
            gt_idx.append(m["gt_index"])
            gt_keys.append(
                f'{m["video"]}:{m["centre"]}:{m["gt_index"]}'
                if m["label"] == "positive" else "")
            area_fracs.append(m["area_frac"])
            vids.append(m["video"])
        pending.clear()
        meta.clear()

    for vi, (name, rel) in enumerate(sorted(videos.items()), 1):
        need = proposals.get(name)
        if not need:
            continue
        path = str(Path(data_root) / rel)
        if not Path(path).exists():
            path = str(Path(data_root) / "videos" / Path(rel).name)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")

        buf = deque(maxlen=T)
        i = 0
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
                    pending.append(_to_tensor(clip, mean, std, layout))
                    meta.append({**pr, "video": name, "centre": centre})
                    if len(pending) >= batch:
                        flush()
            i += 1
        cap.release()
        if progress:
            print(f"  [{vi}/{len(videos)}] {name:<42} {i:>5} frames  "
                  f"{len(scores):>9,} scored")
    flush()

    return {
        "scores": np.asarray(scores),
        "labels": np.asarray(labels),
        "scenes": np.asarray(scenes),
        "gt_index": np.asarray(gt_idx),
        "gt_key": np.asarray(gt_keys),
        "area_frac": np.asarray(area_fracs),
        "videos": np.asarray(vids),
    }


def count_ground_truth(frames_gt, videos, decoded, frame_subset=None,
                       scene_of=None):
    """Ground-truth boxes at every EVALUATED centre, which is the metric's
    denominator. frames_gt maps video -> frame -> list of boxes; decoded maps
    video -> decoded frame count."""
    total, by_scene = 0, defaultdict(int)
    for v in videos:
        n = decoded[v]
        for f in range(LEAD, n - TRAIL):
            if frame_subset is not None and f not in frame_subset.get(v, ()):
                continue
            k = len(frames_gt.get(v, {}).get(f, ()))
            total += k
            if scene_of:
                by_scene[scene_of[v]] += k
    return total, dict(by_scene)
