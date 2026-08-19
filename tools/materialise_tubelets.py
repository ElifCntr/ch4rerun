#!/usr/bin/env python3
"""
Crop materialisation. Training set only.

Val and test are streamed at evaluation time with a rolling buffer and store
nothing, per the 18 August ruling. This writes only what training needs: the
positives, and the union of the three seeds' negative draws. Each seed's
loader then selects its own rows by index from the shared store, so a proposal
drawn by two seeds is stored once.

RULED GEOMETRY.
  Square crop of side (1 + 2p) x max(w, h), p = 0.3, centred on the proposal
  box centre in the ORIGINAL frame, then resized to 112 x 112.
  T = 8, stride 1, window centre-3 to centre+4, the ruled asymmetric
  convention.

EDGE RULE, ruled 18 August: REPLICATE-PAD, never clamp. The crop window stays
centred everywhere; where it leaves the frame, edge pixels are replicated.
Clamping would make the target's position within the crop a function of its
position in the frame, and edge proposals cluster where the subtractor sees
jitter, so the shift would be a scene-correlated confound rather than noise.
Centring is also the invariant items D and H were measured under.

CONDITION ATTACHED TO THAT RULING: every tubelet records its SYNTHETIC-PIXEL
FRACTION in the manifest, so the replication's effect is checkable after the
fact rather than merely asserted.

DECODED-COUNT CLAMP. Window validity uses the actual decoded frame count, not
cv2_frame_count, which overcounts by two on four seaside_cuts clips. A tubelet
whose window would run past the real last frame is dropped and counted.

NO AUGMENTATION HERE. Flip and jitter are ruled per-tubelet at TRAINING time,
one draw applied identically to all eight frames. Baking them in would fix the
draw across every epoch and every arm.

Usage:
    python tools/materialise_tubelets.py --data-root <path>
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

T = 8
LEAD = (T - 1) // 2
TRAIL = T - 1 - LEAD
PAD = 0.3          # ruled 12 Aug
CROP = 112         # ruled, r3d_18's Kinetics weights were trained at 112


def crop_one(frame, cx, cy, side):
    """Square crop centred on (cx, cy), replicate-padded where it leaves the
    frame. Returns (crop, synthetic_pixel_fraction)."""
    h, w = frame.shape[:2]
    half = side / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        # wholly outside the frame; replicate the nearest pixel
        px = min(max(int(cx), 0), w - 1)
        py = min(max(int(cy), 0), h - 1)
        patch = frame[py:py + 1, px:px + 1]
        out = np.repeat(np.repeat(patch, int(round(side)), 0),
                        int(round(side)), 1)
        return out, 1.0
    patch = frame[sy0:sy1, sx0:sx1]
    top, bottom = sy0 - y0, y1 - sy1
    left, right = sx0 - x0, x1 - sx1
    if top or bottom or left or right:
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right,
                                   cv2.BORDER_REPLICATE)
    total = float(int(round(side)) ** 2)
    real = float((sx1 - sx0) * (sy1 - sy0))
    return patch, max(0.0, 1.0 - real / total) if total else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--proposals", default="data/proposals/proposals.csv")
    ap.add_argument("--negatives", default="data/negatives")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--inventory", default="reports/video_inventory.csv")
    ap.add_argument("--out", default="data/tubelets")
    ap.add_argument("--report", default="reports")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)

    inv = {}
    with open(args.inventory, newline="") as fh:
        for r in csv.DictReader(fh):
            inv[r["video"]] = r["path"]

    # wanted[(video, frame, x, y, w, h)] -> {"label":..., "seeds": set}
    wanted = {}

    def add(row, ix, label, seed=None):
        key = (row[ix["video"]], int(row[ix["frame"]]),
               int(row[ix["x"]]), int(row[ix["y"]]),
               int(row[ix["w"]]), int(row[ix["h"]]))
        e = wanted.setdefault(key, {"label": label, "seeds": set()})
        if seed is not None:
            e["seeds"].add(seed)

    with open(args.proposals, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ix = {c: i for i, c in enumerate(header)}
        for row in rd:
            if row[ix["split"]] == "train" and row[ix["label"]] == "positive":
                add(row, ix, "positive")
    n_pos = len(wanted)

    for seed in seeds:
        p = os.path.join(args.negatives, f"negatives_seed{seed}.csv")
        if not os.path.exists(p):
            print(f"FAIL missing {p}")
            sys.exit(1)
        with open(p, newline="") as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            jx = {c: i for i, c in enumerate(hdr)}
            for row in rd:
                add(row, jx, "negative", seed)

    total = len(wanted)
    print(f"{n_pos:,} positives, {total - n_pos:,} distinct negatives across "
          f"seeds {seeds}, {total:,} tubelets to build")

    by_video = defaultdict(list)
    for key, e in wanted.items():
        by_video[key[0]].append((key, e))

    store = np.lib.format.open_memmap(
        os.path.join(args.out, "tubelets.npy"), mode="w+",
        dtype=np.uint8, shape=(total, T, CROP, CROP, 3))

    man = open(os.path.join(args.out, "manifest.csv"), "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["index", "video", "frame", "x", "y", "w", "h", "label",
                 "crop_side_px", "synthetic_fraction", "seeds"])

    written, dropped, synth = 0, [], []
    for vi, video in enumerate(sorted(by_video), 1):
        items = by_video[video]
        need = defaultdict(list)
        for key, e in items:
            need[key[1]].append((key, e))

        p = os.path.join(args.data_root, inv[video])
        if not os.path.exists(p):
            p = os.path.join(args.data_root, "videos",
                             os.path.basename(inv[video]))
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"FAIL cannot open {p}")
            sys.exit(1)

        frames, fi = [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
            fi += 1
        cap.release()
        n_dec = len(frames)

        built = 0
        for centre in sorted(need):
            if centre - LEAD < 0 or centre + TRAIL > n_dec - 1:
                for key, e in need[centre]:
                    dropped.append({"video": video, "frame": centre,
                                    "reason": "window past decoded end"})
                continue
            for key, e in need[centre]:
                _, _, x, y, w, h = key
                side = (1.0 + 2.0 * PAD) * max(w, h)
                cx, cy = x + w / 2.0, y + h / 2.0
                sf = 0.0
                for t in range(T):
                    idx = centre - LEAD + t
                    patch, f_syn = crop_one(frames[idx], cx, cy, side)
                    sf = max(sf, f_syn)
                    store[written, t] = cv2.resize(
                        patch, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
                mw.writerow([written, video, centre, x, y, w, h, e["label"],
                             int(round(side)), round(sf, 6),
                             "|".join(str(s) for s in sorted(e["seeds"]))])
                synth.append(sf)
                written += 1
                built += 1
        del frames
        print(f"[{vi}/{len(by_video)}] {video:<42} {n_dec:>5}f  "
              f"built {built:>6}  total {written:>7,}")

    man.close()
    store.flush()

    sa = np.asarray(synth) if synth else np.zeros(1)
    meta = {
        "tubelets_written": written, "requested": total,
        "dropped": len(dropped), "drop_reasons": dropped[:50],
        "positives": n_pos, "distinct_negatives": total - n_pos,
        "seeds": seeds, "T": T, "lead": LEAD, "trail": TRAIL,
        "padding_fraction": PAD, "crop_px": CROP,
        "edge_rule": "replicate-pad, window always centred (ruled 18 Aug)",
        "window_validity": "decoded frame count per video",
        "augmentation": "none baked in; flip and jitter applied per tubelet "
                        "at training time",
        "synthetic_fraction": {
            "mean": float(sa.mean()), "p50": float(np.percentile(sa, 50)),
            "p90": float(np.percentile(sa, 90)),
            "p99": float(np.percentile(sa, 99)), "max": float(sa.max()),
            "fraction_with_any_padding": float((sa > 0).mean()),
        },
        "store_bytes": int(written) * T * CROP * CROP * 3,
    }
    with open(os.path.join(args.out, "tubelets_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print()
    print(f"written {written:,} of {total:,} requested, "
          f"{len(dropped):,} dropped for windows past the decoded end")
    print(f"synthetic pixels: {(sa > 0).mean():.2%} of tubelets have any, "
          f"median {np.percentile(sa, 50):.4f}, p90 {np.percentile(sa, 90):.4f}"
          f", max {sa.max():.4f}")
    print(f"store {meta['store_bytes'] / 1024**3:.1f} GB at {args.out}/")


if __name__ == "__main__":
    main()
