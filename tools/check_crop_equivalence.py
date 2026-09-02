#!/usr/bin/env python3
"""
Does batching the crops per frame reproduce the per-proposal path, and is it
faster? Measured on one video at a time, CPU only, no GPU and no checkpoint needed.

THE ANSWER WAS: exactly equivalent, and slower, on both videos measured.

  GOPR5846_005  441 centres, 22,757 proposals    4.78 s ->  15.86 s   0.30x
  gopro_005     400 centres, 109,821 proposals  23.90 s -> 105.36 s   0.23x

gopro_005 carries 418,037 of val's 712,925 proposals and is the densest case,
where grouping by side had the most to work with. It is worse there, not
better, because denser centres mean larger groups and the concatenate copy
scales with the group. The batched path is therefore NOT wired into
tools/evaluate_final.py, which keeps its original per-proposal cropping. This
script is the record of that attempt and re-runs it.

WHY IT LOSES. Batching cannot remove the work. cv2.resize at three channels
takes an SIMD path; the arbitrary-channel path it falls back to when crops are
stacked is slower, and the np.concatenate that builds the stack copies every
crop an extra time. Both costs grow with the crop side. The per-call overhead
the batching was meant to remove is a small part of a resize that writes a
112x112x3 output whatever the input side.

THE GEOMETRY IS NOT REIMPLEMENTED HERE. crop_one() in
tools/materialise_tubelets.py defines the crop and produced the training
tubelets. The batched path evaluates crop_one's own arithmetic for every
proposal of a centre at once, takes the plain slice crop_one reduces to when a
crop lies wholly inside the frame, and hands every crop that leaves the frame
to crop_one itself.

EXACT, NOT CLOSE. The two paths must agree bit for bit on uint8 pixels, so the
tolerance is zero. A resize differing by one grey level per pixel would pass a
loose np.allclose and would still mean the evaluation no longer sees the pixels
the training tubelets were built from.

Exits non-zero on any mismatch, printing where it first differs.

Usage:
    python tools/check_crop_equivalence.py --video GOPR5846_005
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from materialise_tubelets import crop_one  # noqa: E402
from src.config import load_config  # noqa: E402
from src.streaming_eval import T, TRAIL, PAD, CROP, load_proposals  # noqa

CHANNEL_CHUNK = 128  # cv2 caps an array at CV_CN_MAX = 512 channels


def dotted(cfg, key):
    node = cfg
    for p in key.split("."):
        node = node[p]
    return node


# --- the path evaluate_final.py uses, copied verbatim ----------------------

def reference_clips(buf, props):
    """One crop_one and one cv2.resize per proposal per buffer frame."""
    out = np.empty((len(props), T, CROP, CROP, 3), dtype=np.uint8)
    for j, pr in enumerate(props):
        side = (1.0 + 2.0 * PAD) * max(pr["w"], pr["h"])
        cx = pr["x"] + pr["w"] / 2.0
        cy = pr["y"] + pr["h"] / 2.0
        clip = np.empty((T, CROP, CROP, 3), dtype=np.uint8)
        for t in range(T):
            patch, _ = crop_one(buf[t], cx, cy, side)
            clip[t] = cv2.resize(patch, (CROP, CROP),
                                 interpolation=cv2.INTER_LINEAR)
        out[j] = clip
    return out


# --- the batched path under test -------------------------------------------

def crop_geometry(props):
    """Per-proposal crop square, as crop_one() computes it. The float centre
    and side are carried so edge cases can be handed to crop_one() unchanged."""
    x = np.array([p["x"] for p in props], dtype=np.float64)
    y = np.array([p["y"] for p in props], dtype=np.float64)
    w = np.array([p["w"] for p in props], dtype=np.float64)
    h = np.array([p["h"] for p in props], dtype=np.float64)
    side = (1.0 + 2.0 * PAD) * np.maximum(w, h)
    cx, cy = x + w / 2.0, y + h / 2.0
    half = side / 2.0
    # np.round and Python's round are both half-to-even, so these are the
    # integers int(round(...)) produces one at a time inside crop_one
    x0 = np.round(cx - half).astype(np.int64)
    y0 = np.round(cy - half).astype(np.int64)
    side_px = np.round(side).astype(np.int64)
    return cx, cy, side, x0, y0, side_px


def side_groups(side_px):
    """Proposal indices grouped by identical integer side, chunked so a group
    never exceeds cv2's channel limit."""
    order = np.argsort(side_px, kind="stable")
    out, start, n = [], 0, len(order)
    for k in range(1, n + 1):
        if k == n or side_px[order[k]] != side_px[order[start]]:
            idx = order[start:k]
            s = int(side_px[order[start]])
            for c in range(0, len(idx), CHANNEL_CHUNK):
                out.append((s, idx[c:c + CHANNEL_CHUNK]))
            start = k
    return out


def batched_clips(buf, props):
    """Every crop of a centre cut from each buffer frame together, and crops
    sharing an integer side resized in one call, stacked along the channel
    axis, which INTER_LINEAR treats independently."""
    out = np.empty((len(props), T, CROP, CROP, 3), dtype=np.uint8)
    cx, cy, side, x0, y0, side_px = crop_geometry(props)
    groups = side_groups(side_px)
    for t, frame in enumerate(buf):
        fh, fw = frame.shape[:2]
        for s, idx in groups:
            patches = []
            for j in idx:
                a, b = int(x0[j]), int(y0[j])
                if a >= 0 and b >= 0 and a + s <= fw and b + s <= fh:
                    # wholly inside: crop_one applies no border here
                    patches.append(frame[b:b + s, a:a + s])
                else:
                    patch, _ = crop_one(frame, cx[j], cy[j], side[j])
                    patches.append(patch)
            if len(idx) == 1:
                out[idx[0], t] = cv2.resize(
                    patches[0], (CROP, CROP),
                    interpolation=cv2.INTER_LINEAR)
                continue
            r = cv2.resize(np.concatenate(patches, axis=2), (CROP, CROP),
                           interpolation=cv2.INTER_LINEAR)
            out[idx, t] = r.reshape(CROP, CROP, len(idx), 3).transpose(
                2, 0, 1, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="GOPR5846_005")
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--proposals", default="data/proposals/proposals.csv")
    ap.add_argument("--inventory", default="reports/video_inventory.csv")
    ap.add_argument("--max-centres", type=int, default=0,
                    help="0 means every centre in the video")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(dotted(cfg, "data.root")).expanduser()

    inv = {}
    with open(args.inventory, newline="") as fh:
        for r in csv.DictReader(fh):
            inv[r["video"]] = r["path"]
    if args.video not in inv:
        print(f"FAIL {args.video} is not in {args.inventory}")
        sys.exit(1)

    props = load_proposals(args.proposals, args.split).get(args.video)
    if not props:
        print(f"FAIL no {args.split} proposals for {args.video}")
        sys.exit(1)

    path = root / inv[args.video]
    if not path.exists():
        path = root / "videos" / Path(inv[args.video]).name
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"FAIL cannot open {path}")
        sys.exit(1)

    n_prop = n_centre = 0
    t_ref = t_new = 0.0
    buf, i = deque(maxlen=T), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        if len(buf) == T:
            centre = i - TRAIL
            at_centre = props.get(centre)
            if at_centre:
                t0 = time.perf_counter()
                ref = reference_clips(buf, at_centre)
                t1 = time.perf_counter()
                new = batched_clips(buf, at_centre)
                t2 = time.perf_counter()
                t_ref += t1 - t0
                t_new += t2 - t1

                if ref.shape != new.shape or ref.dtype != new.dtype:
                    print(f"FAIL shape or dtype differ at centre {centre}: "
                          f"reference {ref.shape} {ref.dtype}, "
                          f"batched {new.shape} {new.dtype}")
                    sys.exit(1)
                if not np.allclose(ref, new, rtol=0.0, atol=0.0):
                    bad = np.argwhere(ref != new)
                    j, t, y, x, c = bad[0]
                    pr = at_centre[j]
                    diff = int(np.abs(ref.astype(np.int16)
                                      - new.astype(np.int16)).max())
                    print(f"FAIL crops differ at centre {centre}, "
                          f"{len(bad):,} of {ref.size:,} pixels")
                    print(f"  first at proposal {j} (x={pr['x']} y={pr['y']} "
                          f"w={pr['w']} h={pr['h']}), frame {t}, "
                          f"pixel ({y}, {x}), channel {c}")
                    print(f"  reference {ref[j, t, y, x, c]}  "
                          f"batched {new[j, t, y, x, c]}  "
                          f"max abs diff {diff}")
                    sys.exit(1)

                n_prop += len(at_centre)
                n_centre += 1
                if args.max_centres and n_centre >= args.max_centres:
                    break
        i += 1
    cap.release()

    if n_centre == 0:
        print(f"FAIL no completed centre carried a proposal in {args.video}")
        sys.exit(1)

    print(f"{args.video}: {n_centre:,} centres, {n_prop:,} proposals, "
          f"{n_prop * T:,} crops")
    print(f"EQUIVALENT exact match, atol 0, rtol 0, uint8, "
          f"{n_prop * T * CROP * CROP * 3:,} pixels compared")
    print(f"  per-proposal path {t_ref:8.2f} s")
    print(f"  batched path      {t_new:8.2f} s")
    print(f"  speedup           {t_ref / t_new:8.2f}x  (cropping only, "
          f"decode excluded)")
    if t_new > t_ref:
        print("  SLOWER: the batched path is not used by "
              "tools/evaluate_final.py")


if __name__ == "__main__":
    main()
