#!/usr/bin/env python3
"""
Decode throughput for the streaming evaluation path.

Authorised 18 Aug. The frame store was dropped on a 295.4 fps benchmark taken
on two 1080p videos, but val and test are each about 40 per cent 4K by frame,
so the figure needs re-checking where it matters.

Measures plain sequential decode, no subtractor and no matching, PER
RESOLUTION GROUP, and reports what a full streaming pass over val and test
would cost. Also times decode with a rolling 8-frame buffer held, since that
is what evaluation actually does, to confirm the buffer costs nothing.

Usage:
    python tools/check_decode_4k.py --data-root <path>
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

import cv2

T = 8   # ruled 12 Aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--inventory", default="reports/video_inventory.csv")
    ap.add_argument("--splits", default="data/splits/dvb_splits_v2.0-static.csv")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    for p in (args.inventory, args.splits):
        if not os.path.exists(p):
            print(f"FAIL missing {p}")
            sys.exit(1)

    smap = {}
    with open(args.splits, newline="") as fh:
        for r in csv.DictReader(fh):
            smap[r["video"]] = r["split"]

    inv = {}
    with open(args.inventory, newline="") as fh:
        for r in csv.DictReader(fh):
            inv[r["video"]] = {
                "frames": int(float(r["frames"])),
                "w": int(float(r["cv2_width"])),
                "h": int(float(r["cv2_height"])),
                "path": r["path"],
            }

    # pick videos per resolution group, preferring val and test since those
    # are the splits that stream
    groups = defaultdict(list)
    for v, d in sorted(inv.items()):
        if v not in smap:
            continue
        key = f"{d['w']}x{d['h']}"
        groups[key].append((0 if smap[v] in ("val", "test") else 1, v))
    picks = {k: [v for _, v in sorted(vs)][:args.per_group]
             for k, vs in groups.items()}

    def open_video(v):
        path = os.path.join(args.data_root, inv[v]["path"])
        if not os.path.exists(path):
            path = os.path.join(args.data_root, "videos",
                                os.path.basename(inv[v]["path"]))
        cap = cv2.VideoCapture(path)
        return cap if cap.isOpened() else None

    L = ["DECODE THROUGHPUT FOR THE STREAMING PATH",
         "Plain sequential decode. No subtractor, no matching.",
         f"Buffered figures hold a rolling {T}-frame window, which is what "
         "evaluation does.",
         ""]

    rates = {}
    for res, vids in sorted(picks.items()):
        L.append(f"RESOLUTION {res}")
        tot_f, tot_plain, tot_buf = 0, 0.0, 0.0
        for v in vids:
            cap = open_video(v)
            if cap is None:
                L.append(f"  could not open {v}, skipped")
                continue
            n, t0 = 0, time.time()
            while n < args.frames:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            plain = time.time() - t0
            cap.release()

            cap = open_video(v)
            buf, m, t0 = deque(maxlen=T), 0, time.time()
            while m < args.frames:
                ok, f = cap.read()
                if not ok:
                    break
                buf.append(f)
                m += 1
            buffered = time.time() - t0
            cap.release()

            tot_f += n
            tot_plain += plain
            tot_buf += buffered
            L.append(f"  {v:<44} {smap[v]:<5} {n:>4} frames   "
                     f"plain {n / plain if plain else 0:>7.1f} fps   "
                     f"buffered {m / buffered if buffered else 0:>7.1f} fps")
        if tot_plain:
            rates[res] = tot_f / tot_plain
            L.append(f"  GROUP: plain {tot_f / tot_plain:.1f} fps, "
                     f"buffered {tot_f / tot_buf:.1f} fps")
        L.append("")

    # cost of a full streaming pass, per split, using each video's own group
    L.append("FULL STREAMING PASS COST, per split")
    for sp in ("val", "test"):
        secs, missing = 0.0, 0
        for v, d in inv.items():
            if smap.get(v) != sp:
                continue
            key = f"{d['w']}x{d['h']}"
            if key in rates:
                secs += d["frames"] / rates[key]
            else:
                missing += 1
        L.append(f"  {sp:<5} {secs / 60:>7.2f} min" +
                 (f"   ({missing} videos at unmeasured resolutions)"
                  if missing else ""))
    L.append("")
    L.append("This is decode only. The forward pass is timed separately by "
             "tools/inference_timing.py and is the dominant cost.")

    txt = "\n".join(L)
    path = os.path.join(args.out, "decode_throughput.txt")
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
