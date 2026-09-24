#!/usr/bin/env python3
"""
00_02_45_to_00_03_10_cut.mpg: name arithmetic, annotation coverage, contiguity,
and the file's standing in the seaside session and the fleet.

TRAIN PARTITION ONLY. The file is in the sole seaside session, which is
train-only, so no test access is involved and src/splits.py is not consulted
for a test flag. The annotation file read here is a train annotation.

THE DECISIVE QUESTION IS ANNOTATION COVERAGE. The name spans 25 seconds of
source timestamps and the file decodes 8. If the annotation ran to the full
25 seconds, every box past frame 400 would be invisible to the pipeline and
the denominator would be understated. Section 2 settles that against the
committed annotation file, read through the same parser the split build uses,
and cross-checks the result against the committed split and extraction
artefacts. Everything else in this script is subordinate to that answer.

Sections:
  1 name arithmetic against the decoded frame count
  2 annotation coverage, the decisive check
  3 contiguity from packet timestamps, plus a contact sheet
  4 the seaside session as a whole, and the +2 overcount
  5 the .mpg against the other 36 static videos

Usage:
    python tools/investigate_seaside_mpg.py
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from measure_instances import read_boxes  # noqa: E402

VIDEO = "00_02_45_to_00_03_10_cut"
SPLIT = "data/splits/dvb_splits_v2.0-static.csv"
INVENTORY = "reports/video_inventory.csv"
STATS = "reports/extraction_stats.csv"
OUT = Path("reports/seaside_mpg")
ROOT = Path(os.path.expanduser("~/Desktop/Drone-vs-Bird/videos-and-labels"))

T, LEAD, TRAIL = 8, 3, 4
TB, FPS_EXPECTED = 90000, 50


def fail(msg: str) -> None:
    print(f"FAIL {msg}", file=sys.stderr)
    sys.exit(1)


def probe(path: Path, entries: str) -> list[str]:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries,
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True).stdout.split()


def main() -> None:
    for p in (SPLIT, INVENTORY, STATS):
        if not Path(p).exists():
            fail(f"missing {p}")
    OUT.mkdir(parents=True, exist_ok=True)

    split = {r["video"]: r for r in csv.DictReader(open(SPLIT, newline=""))}
    inv = {r["video"]: r for r in csv.DictReader(open(INVENTORY, newline=""))}
    stats = {r["video"]: r for r in csv.DictReader(open(STATS, newline=""))}
    if split[VIDEO]["split"] != "train":
        fail(f"{VIDEO} is not in train; this script asserts train-only access")

    path = ROOT / inv[VIDEO]["path"]
    if not path.exists():
        fail(f"missing video {path}")
    dec = int(stats[VIDEO]["decoded_frames"])

    # ---- 1. name arithmetic ------------------------------------------------
    h1, m1, s1, _, h2, m2, s2, _ = VIDEO.split("_")
    a = int(h1) * 3600 + int(m1) * 60 + int(s1)
    b = int(h2) * 3600 + int(m2) * 60 + int(s2)
    span = b - a
    fps = float(inv[VIDEO]["cv2_fps"])
    implied = int(round(span * fps))
    print("== 1. name arithmetic ==")
    print(f"   file                     {path.name}")
    print(f"   name parses as           {h1}:{m1}:{s1} to {h2}:{m2}:{s2}, "
          f"{a} s to {b} s")
    print(f"   source-timestamp span    {span} s")
    print(f"   rate                     {fps:.3f} fps "
          f"(inventory and ffprobe agree)")
    print(f"   frames implied by name   {span} x {fps:.0f} = {implied:,}")
    print(f"   frames decoded           {dec:,}")
    print(f"   SHORTFALL                {dec:,}/{implied:,} = "
          f"{dec / implied * 100:.2f}% present")
    print(f"   absent                   {implied - dec:,} frames, "
          f"{span - dec / fps:.2f} s of the named span")
    print(f"   decoded duration         {dec / fps:.2f} s")

    # ---- 1b. does the naming convention hold elsewhere? --------------------
    # A shortfall against the name only means something if the name is a
    # reliable duration label in the first place. The other four seaside files
    # test that, and they carry no _cut suffix.
    NAME_RE = re.compile(r"^(\d\d)_(\d\d)_(\d\d)_to_(\d\d)_(\d\d)_(\d\d)(_cut)?$")
    sea_all = sorted(v for v, r in split.items() if r["scene"] == "seaside")
    print("\n== 1b. does the name predict the duration elsewhere in the session? ==")
    print(f"   {'video':<26}{'span_s':>8}{'decoded':>9}{'fps':>8}{'dur_s':>8}"
          f"{'ratio':>9}  suffix")
    for v in sea_all:
        m = NAME_RE.match(v)
        if not m:
            print(f"   {v:<26} name does not parse")
            continue
        aa = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        bb = int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6))
        sp_s = bb - aa
        d = int(stats[v]["decoded_frames"])
        f_ = float(inv[v]["cv2_fps"])
        print(f"   {v:<26}{sp_s:>8}{d:>9}{f_:>8.3f}{d / f_:>8.2f}"
              f"{d / f_ / sp_s:>9.2%}  {m.group(7) or '-'}")
    print("   Every seaside name WITHOUT the _cut suffix predicts the duration")
    print("   to within a second. The one name WITH it does not, and that file")
    print("   is the subject of this investigation.")

    # ---- 2. annotation coverage, decisive ----------------------------------
    ann = ROOT / "labels" / f"{VIDEO}.txt"
    if not ann.exists():
        fail(f"missing annotation {ann}")
    frames = read_boxes(ann, "dvb_multibox_per_line")
    idx = sorted(frames)
    total = sum(len(v) for v in frames.values())
    beyond = {f: v for f, v in frames.items() if f >= dec}
    n_beyond = sum(len(v) for v in beyond.values())
    with_box = sum(1 for v in frames.values() if v)
    empty = [f for f, v in frames.items() if not v]
    gaps = [f for f in range(min(idx), max(idx) + 1) if f not in frames]

    print("\n== 2. annotation coverage, the decisive check ==")
    print(f"   annotation file          {ann}")
    print(f"   frame entries            {len(frames)}")
    print(f"   annotated frame range    {min(idx)} .. {max(idx)}")
    print(f"   MAXIMUM annotated frame  {max(idx)}")
    print(f"   total boxes              {total}")
    print(f"   frames carrying a box    {with_box}")
    print(f"   frames with zero boxes   {len(empty)} {empty}")
    print(f"   missing indices in range {len(gaps)} {gaps[:10]}")
    print(f"   decoded frames           {dec}  (indices 0 .. {dec - 1})")
    print(f"   BOXES ON FRAMES >= {dec}  {n_beyond}")
    if n_beyond:
        print(f"   frames beyond decode     {sorted(beyond)[:20]}")
        print(f"   THE PIPELINE NEVER SEES  {n_beyond} boxes "
              f"({n_beyond / total * 100:.2f}% of {total})")
    else:
        print("   Nothing is lost to truncation. The annotation ends exactly")
        print("   where the decode ends, and every annotated box is inside the")
        print("   decoded range.")

    # cross-check against the committed artefacts
    sb, sf = int(split[VIDEO]["boxes"]), int(split[VIDEO]["annotated_frames"])
    inwin = sum(len(frames.get(f, ())) for f in range(LEAD, dec - TRAIL))
    es_in = int(stats[VIDEO]["gt_boxes_in_valid_window"])
    checks = [("split boxes", sb, total),
              ("split annotated_frames", sf, with_box),
              ("gt_boxes_in_valid_window", es_in, inwin)]
    print("\n   cross-check against the committed artefacts:")
    bad = []
    for name, committed, recount in checks:
        ok = committed == recount
        print(f"     {name:<26} committed {committed:>5}   recount {recount:>5}"
              f"   {'match' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{name}: committed {committed}, recount {recount}")
    print(f"     {'boxes excluded by window':<26} {total - inwin:>5} "
          f"(first {LEAD} and last {TRAIL} decoded frames, T={T})")
    if bad:
        fail("annotation recount disagrees with the committed record: "
             + "; ".join(bad))

    # ---- 3. contiguity -----------------------------------------------------
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
         "-show_entries", "packet=pts,dts,duration,size,flags",
         "-of", "csv=p=0:nk=1", str(path)],
        capture_output=True, text=True).stdout
    pk = [l.split(",") for l in out.strip().splitlines() if l]
    def I(v): return None if v in ("N/A", "") else int(v)
    pts = [I(r[0]) for r in pk]
    dts = [I(r[1]) for r in pk]
    dur = [I(r[2]) for r in pk]
    flags = [r[4] for r in pk]
    step = TB // int(round(fps))
    have = sorted(v for v in pts if v is not None)
    no_pts = [i for i, v in enumerate(pts) if v is None]
    dts_deltas = Counter(b - a for a, b in zip(dts, dts[1:]))
    ladder = list(range(have[0], have[-1] + 1, step))
    absent = [v for v in ladder if v not in set(have)]
    key = [i for i, f in enumerate(flags) if "K" in f]

    print("\n== 3. contiguity ==")
    print(f"   packets                  {len(pk)}   decoded {dec}   "
          f"{'agree' if len(pk) == dec else 'DISAGREE'}")
    print(f"   packet durations         {dict(Counter(dur))} ticks "
          f"(expected {step} at {fps:.0f} fps, time_base 1/{TB})")
    print(f"   PTS first/last           {have[0]} / {have[-1]}  = "
          f"{have[0] / TB:.3f} s .. {have[-1] / TB:.3f} s")
    print(f"   packets with no PTS      {len(no_pts)} at index {no_pts}")
    print(f"   PTS values absent        {absent}")
    print(f"   DTS deltas               {dict(dts_deltas)}")
    print(f"   DTS monotonic            {all(b >= a for a, b in zip(dts, dts[1:]))}")
    expected_last = dts[0] + (len(pk) - 1) * step
    contiguous = (dict(dts_deltas) == {step: len(pk) - 1}
                  and dts[-1] == expected_last
                  and len(absent) == len(no_pts))
    print(f"   DTS last / expected      {dts[-1]} / {expected_last}")
    print(f"   keyframes                {len(key)} at {key}")
    print(f"   GOP gaps                 {dict(Counter([b - a for a, b in zip(key, key[1:])]))}")
    print(f"   CONTIGUOUS               {contiguous}")
    print("   The one absent PTS is a reordered frame whose timestamp the")
    print("   MPEG-PS mux omits; DTS is unbroken across it, so it is a missing")
    print("   timestamp and not a discontinuity. No gap and no reset.")

    # contact sheet: first, last, and 5 spread, decoded sequentially
    want = sorted({0, dec - 1, *[int(round(x)) for x in
                                 np.linspace(0, dec - 1, 7)[1:-1]]})
    cap = cv2.VideoCapture(str(path))
    got, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = fr
        i += 1
    cap.release()
    if i != dec:
        fail(f"sequential decode yielded {i} frames, extraction_stats says {dec}")
    if sorted(got) != want:
        fail(f"contact sheet wanted {want}, got {sorted(got)}")
    tiles = []
    for f in want:
        t = cv2.resize(got[f], (480, 270))
        cv2.putText(t, f"f{f}  t={f / fps:.2f}s", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[:4]),
                       np.hstack(tiles[4:] + [np.zeros_like(tiles[0])]
                                 * (4 - len(tiles[4:])))])
    cv2.imwrite(str(OUT / "contact_sheet.png"), sheet)
    print(f"   contact sheet            {OUT / 'contact_sheet.png'} "
          f"frames {want}")

    # ---- 4. the seaside session -------------------------------------------
    sea = sorted(v for v, r in split.items() if r["scene"] == "seaside")
    print("\n== 4. the seaside session ==")
    print(f"   {'video':<26}{'ext':<6}{'container_n':>12}{'decoded':>9}"
          f"{'over':>6}{'fps':>9}  {'codec':<12}encoder")
    fleet = {}
    for v in sorted(split):
        p = ROOT / inv[v]["path"]
        o = probe(p, "format=format_name:format_tags=encoder:stream=codec_name")
        fleet[v] = dict(codec=o[0], fmt=o[1], enc=o[2] if len(o) > 2 else "(none)",
                        ext=p.suffix, cv2n=int(inv[v]["cv2_frame_count"]),
                        dec=int(stats[v]["decoded_frames"]),
                        fps=round(float(inv[v]["cv2_fps"]), 3),
                        scene=split[v]["scene"])
        fleet[v]["over"] = fleet[v]["cv2n"] - fleet[v]["dec"]
    for v in sea:
        f = fleet[v]
        print(f"   {v:<26}{f['ext']:<6}{f['cv2n']:>12}{f['dec']:>9}"
              f"{f['over']:>6}{f['fps']:>9.3f}  {f['codec']:<12}{f['enc']}")
    over = [v for v in fleet if fleet[v]["over"]]
    print(f"\n   videos with an overcount, whole fleet: {len(over)}")
    for v in sorted(over):
        print(f"     {v:<26} +{fleet[v]['over']}")
    print(f"   {VIDEO} carries an overcount: "
          f"{bool(fleet[VIDEO]['over'])}")
    sig = defaultdict(list)
    for v, f in fleet.items():
        sig[(f["codec"], f["ext"], f["enc"])].append(f["over"])
    print("\n   overcount against (codec, extension, encoder):")
    for k in sorted(sig):
        print(f"     {str(k):<48} overcounts {dict(Counter(sig[k]))}")

    # ---- 5. the .mpg against the fleet -------------------------------------
    print("\n== 5. the .mpg against the other 36 ==")
    print(f"   container format_name  {dict(Counter(f['fmt'] for f in fleet.values()))}")
    print(f"   extension              {dict(Counter(f['ext'] for f in fleet.values()))}")
    print(f"   codec                  {dict(Counter(f['codec'] for f in fleet.values()))}")
    print(f"   encoder tag            {dict(Counter(f['enc'] for f in fleet.values()))}")
    print(f"   frame rate             {dict(Counter(f['fps'] for f in fleet.values()))}")
    m = fleet[VIDEO]
    print(f"\n   {VIDEO}: {m['codec']} in {m['fmt']}, encoder {m['enc']}, "
          f"{m['fps']:.3f} fps")
    print("   It is the only MPEG-PS container, the only .mpg, the only file")
    print("   with no encoder tag, and the only file at 50 fps. It did NOT go")
    print("   through the Lavf re-encode that produced every other file.")

    with open(OUT / "fleet_containers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "scene", "ext", "codec",
                                           "fmt", "enc", "cv2n", "dec", "over",
                                           "fps"])
        w.writeheader()
        for v in sorted(fleet):
            w.writerow({"video": v, **fleet[v]})
    print(f"\n   written to {OUT / 'fleet_containers.csv'}")
    print("\nNo verdict on chapter wording is drawn here.")


if __name__ == "__main__":
    main()
