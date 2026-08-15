"""Stage 1a item F. Motion separability, drone against background.

Ruled 12 August 2026: T = 8, p = 0.3, both now fixed, so item F carries no T
or padding dimension. Drone tubelets are set against random background crops
drawn from the 1,813 zero-drone frames, size-matched to the measured drone
distribution.

WHAT F ASKS. Shown a strip with no label, can the rater tell a drone from
background clutter, and WHAT told them. The second half is the point. A
forced choice alone would be answered by appearance, and F is commissioned as
a MOTION question, so every strip also records the cue: appearance, motion, or
both. That does not isolate motion, which no design on real footage can, but
it reports how often motion was the deciding cue rather than assuming it.

DESIGN DECISIONS THAT ARE MINE AND BELONG IN THE METHOD SECTION.

1. Drone tubelets are sampled FRESH, not reused from item E. Reuse would let
   the rater recognise strips already scored and answer from memory rather
   than from the image. The cost is that F cannot be related strip-by-strip to
   E's usability scores.

2. A background tubelet requires ALL EIGHT of its frames to be zero-drone in
   the annotations, not just its centre. Otherwise a drone could wander into a
   strip labelled background and the ground truth would be wrong.

3. Background crop size is drawn from the EMPIRICAL drone box distribution of
   the same resolution group, not from a fitted one. Matching per resolution
   matters because the 4K clips carry 2.5x the area fraction of the 1080p
   ones, so a pooled draw would give 4K backgrounds the wrong scale.

4. Background crop placement is uniform over positions where the whole crop
   fits inside the frame. No attempt is made to place crops on "interesting"
   background: that would be the rater's job to judge, not the sampler's to
   pre-select, and choosing what counts as interesting would be a parameter.

Same blind protocol as item E: randomised order, opaque strip ids, no label
on the image, and about a tenth of strips repeated for intra-rater
consistency. The manifest records what each strip really is; the scoring sheet
does not.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_root, load_config, repo_path, require  # noqa: E402
from src.seeding import seed_everything  # noqa: E402
from tools.make_strips import (contact_sheet, crop_box, extract,  # noqa: E402
                               window_frames)


def read_csv(path: Path) -> list[dict]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"],
                           cfg["seed"]["cudnn_benchmark"])
    # Offset so F draws a different sample from E even at the same base seed.
    rng = random.Random(seed + 1000)

    f = require(cfg, "item_f")
    T = f["t"]
    P = f["padding_fraction"]
    N_DRONE = f["drone_tubelets"]
    N_BG = f["background_tubelets"]
    DUP_FRACTION = f["duplicate_fraction"]
    STRIDE = 1

    root = data_root(cfg)
    video_dir = root / require(cfg, "data.video_dir")
    exts = {e.lower() for e in require(cfg, "data.video_extensions")}
    out_dir = repo_path(cfg, cfg["reports"]["dir"])
    strips_dir = repo_path(cfg, cfg["reports"]["strips_dir"])
    man_dir = repo_path(cfg, cfg["reports"]["strip_manifest_dir"])
    sheet_dir = repo_path(cfg, cfg["reports"]["scoring_sheet_dir"])
    for d in (strips_dir, man_dir, sheet_dir):
        d.mkdir(parents=True, exist_ok=True)

    boxes = read_csv(out_dir / "instances_boxes.csv")
    tracks: dict[tuple, list[dict]] = defaultdict(list)
    for b in boxes:
        tracks[(b["video"], b["track_id"])].append(b)
    for rows in tracks.values():
        rows.sort(key=lambda r: int(r["frame"]))

    meta = {b["video"]: b for b in boxes}          # scene, resolution, fps
    sizes_by_res: dict[str, list[tuple]] = defaultdict(list)
    for b in boxes:
        sizes_by_res[b["resolution"]].append((float(b["w"]), float(b["h"])))

    # ------------------------------------------------- drone tubelets ------
    lead = (T - 1) // 2
    tail = T - 1 - lead
    cands = []
    for key, rows in tracks.items():
        if int(rows[-1]["frame"]) - int(rows[0]["frame"]) != len(rows) - 1:
            continue
        lo = int(rows[0]["frame"]) + lead
        hi = int(rows[-1]["frame"]) - tail
        if hi >= lo:
            cands.append((rows[0]["scene"], key, lo, hi))
    by_scene: dict[str, list] = defaultdict(list)
    for scene, key, lo, hi in cands:
        by_scene[scene].append((key, lo, hi))

    drone = []
    scenes = sorted(by_scene)
    per_scene = [N_DRONE // len(scenes)] * len(scenes)
    for i in range(N_DRONE - sum(per_scene)):
        per_scene[i] += 1
    for scene, want in zip(scenes, per_scene):
        pool = by_scene[scene][:]
        rng.shuffle(pool)
        picked, used = [], set()
        for round_ in (0, 1):
            for key, lo, hi in pool:
                if len(picked) >= want:
                    break
                if round_ == 0 and key in used:
                    continue
                picked.append((key, rng.randint(lo, hi)))
                used.add(key)
        drone += [(scene, k, c) for k, c in picked[:want]]

    # -------------------------------------------- background tubelets ------
    # A zero-drone frame is one the annotations list with no box. Every frame
    # of the window must be zero-drone, not merely the centre.
    annotated: dict[str, set[int]] = defaultdict(set)
    all_frames: dict[str, set[int]] = defaultdict(set)
    for b in boxes:
        annotated[b["video"]].add(int(b["frame"]))
    ann_dir = root / require(cfg, "data.annotation_dir")
    ann_ext = require(cfg, "data.annotation_extension")
    zero_by_video: dict[str, list[int]] = defaultdict(list)
    for video in sorted(annotated):
        path = ann_dir / f"{video}{ann_ext}"
        for line in path.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) < 2:
                continue
            fr, n_obj = int(float(t[0])), int(float(t[1]))
            all_frames[video].add(fr)
            if n_obj == 0:
                zero_by_video[video].append(fr)

    bg_cands = []
    for video, zeros in zero_by_video.items():
        z = set(zeros)
        for c in sorted(z):
            frs = window_frames(c, T, STRIDE)
            if all(fr in z for fr in frs):
                bg_cands.append((video, c))
    if len(bg_cands) < N_BG:
        print(f"WARNING: only {len(bg_cands)} background windows available, "
              f"wanted {N_BG}")
    rng.shuffle(bg_cands)
    # Spread over videos rather than taking many from one.
    per_video: dict[str, int] = Counter()
    background = []
    for video, c in bg_cands:
        if len(background) >= N_BG:
            break
        if per_video[video] >= max(1, N_BG // max(1, len(zero_by_video))) + 1:
            continue
        m = meta.get(video)
        if m is None:
            continue
        W, H = (int(v) for v in m["resolution"].split("x"))
        w, h = rng.choice(sizes_by_res[m["resolution"]])
        side = max(w, h) * (1.0 + 2.0 * P)
        if side >= min(W, H):
            continue
        cx = rng.uniform(side / 2, W - side / 2)
        cy = rng.uniform(side / 2, H - side / 2)
        background.append((video, c, cx, cy, w, h))
        per_video[video] += 1

    # ------------------------------------------------------- strip list ----
    strips = []
    for scene, key, centre in drone:
        video, tid = key
        rows = {int(r["frame"]): r for r in tracks[key]}
        c = rows[centre]
        strips.append({
            "kind": "drone", "video": video, "track_id": tid, "scene": scene,
            "resolution": c["resolution"], "fps": c["fps"],
            "centre_frame": centre, "T": T, "stride": STRIDE,
            "padding_fraction": P,
            "frame_indices": ";".join(str(x) for x in
                                      window_frames(centre, T, STRIDE)),
            "cx": c["cx"], "cy": c["cy"], "w": c["w"], "h": c["h"],
            "equiv_side": c["equiv_side"], "seed": seed,
        })
    for video, centre, cx, cy, w, h in background:
        m = meta[video]
        strips.append({
            "kind": "background", "video": video, "track_id": "",
            "scene": m["scene"], "resolution": m["resolution"],
            "fps": m["fps"], "centre_frame": centre, "T": T, "stride": STRIDE,
            "padding_fraction": P,
            "frame_indices": ";".join(str(x) for x in
                                      window_frames(centre, T, STRIDE)),
            "cx": round(cx, 2), "cy": round(cy, 2),
            "w": round(w, 2), "h": round(h, 2),
            "equiv_side": round((w * h) ** 0.5, 3), "seed": seed,
        })
    rng.shuffle(strips)
    for i, s in enumerate(strips):
        s["strip_id"] = f"F{i:04d}"

    # ---------------------------------------------------------- decode -----
    needed: dict[str, set[int]] = defaultdict(set)
    for s in strips:
        needed[s["video"]].update(int(x) for x in s["frame_indices"].split(";"))
    cache: dict[tuple, np.ndarray] = {}
    for video, want in sorted(needed.items()):
        matches = [p for p in video_dir.rglob("*")
                   if p.stem == video and p.suffix.lower() in exts]
        if not matches:
            sys.exit(f"video not found: {video}")
        cap = cv2.VideoCapture(str(matches[0]))
        target, idx, ptr = sorted(want), 0, 0
        while ptr < len(target):
            ok, frame = cap.read()
            if not ok:
                print(f"WARNING: {video} ended at {idx}")
                break
            while ptr < len(target) and target[ptr] == idx:
                cache[(video, idx)] = frame.copy()
                ptr += 1
            idx += 1
        cap.release()
        print(f"decoded {video}: {len(want)} frames")

    written = 0
    for s in strips:
        x0, y0, side = crop_box(float(s["cx"]), float(s["cy"]),
                                float(s["w"]), float(s["h"]), P)
        crops = []
        for fr in (int(x) for x in s["frame_indices"].split(";")):
            fm = cache.get((s["video"], fr))
            if fm is None:
                break
            crops.append(extract(fm, x0, y0, side))
        if len(crops) != T:
            print(f"WARNING: {s['strip_id']} incomplete, skipped")
            continue
        cv2.imwrite(str(strips_dir / f"{s['strip_id']}.png"),
                    contact_sheet(crops, s["strip_id"]))
        written += 1

    # -------------------------------------------------------- artefacts ----
    with open(man_dir / "item_f_sampling_manifest.csv", "w", newline="",
              encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(strips[0].keys()))
        w_.writeheader()
        w_.writerows(strips)

    n_dup = max(1, int(round(len(strips) * DUP_FRACTION)))
    dups = rng.sample(strips, n_dup)
    sheet = [{"sheet_row": None, "strip_id": s["strip_id"],
              "call": "", "cue": "", "note": ""} for s in strips]
    dup_map = []
    for i, s in enumerate(dups):
        rid = f"F9{i:03d}"
        sheet.append({"sheet_row": None, "strip_id": rid, "call": "",
                      "cue": "", "note": ""})
        dup_map.append({"repeat_id": rid, "original_id": s["strip_id"]})
        x0, y0, side = crop_box(float(s["cx"]), float(s["cy"]),
                                float(s["w"]), float(s["h"]), P)
        crops = [extract(cache[(s["video"], fr)], x0, y0, side)
                 for fr in (int(x) for x in s["frame_indices"].split(";"))
                 if (s["video"], fr) in cache]
        if len(crops) == T:
            cv2.imwrite(str(strips_dir / f"{rid}.png"),
                        contact_sheet(crops, rid))
    rng.shuffle(sheet)
    for i, r in enumerate(sheet, 1):
        r["sheet_row"] = i

    with open(sheet_dir / "item_f_scoring_sheet_BLANK.csv", "w", newline="",
              encoding="utf-8") as fh:
        fh.write("# Item F motion-separability sheet. Randomised order, no "
                 "label on any image.\n")
        fh.write("# call: drone / background / cannot_tell\n")
        fh.write("# cue:  appearance / motion / both  -- what decided it. "
                 "Leave blank for cannot_tell.\n")
        fh.write("#   appearance  a single frame would have told you\n")
        fh.write("#   motion      only the change across frames told you\n")
        fh.write("#   both        either alone would have sufficed\n")
        fh.write("# note: optional free text.\n")
        w_ = csv.DictWriter(fh, fieldnames=["sheet_row", "strip_id", "call",
                                            "cue", "note"])
        w_.writeheader()
        w_.writerows(sheet)

    with open(man_dir / "item_f_duplicate_map.csv", "w", newline="",
              encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["repeat_id", "original_id"])
        w_.writeheader()
        w_.writerows(dup_map)

    # Per-video distribution of zero-drone frames, ruled 12 Aug.
    with open(out_dir / "zero_drone_frame_distribution.csv", "w", newline="",
              encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=[
            "video", "scene", "zero_drone_frames", "windows_of_T",
            "sampled_backgrounds"])
        w_.writeheader()
        wins = Counter(v for v, _ in bg_cands)
        used = Counter(s["video"] for s in strips if s["kind"] == "background")
        for video in sorted(zero_by_video):
            w_.writerow({
                "video": video, "scene": meta.get(video, {}).get("scene", ""),
                "zero_drone_frames": len(zero_by_video[video]),
                "windows_of_T": wins.get(video, 0),
                "sampled_backgrounds": used.get(video, 0)})

    print(f"\nseed                 {seed} (+1000 offset so F != E sample)")
    print(f"drone tubelets       {len(drone)}")
    print(f"background tubelets  {len(background)} "
          f"from {len(set(v for v, _, _, _, _, _ in background))} videos")
    print(f"strips written       {written} of {len(strips)}")
    print(f"sheet rows           {len(sheet)} ({n_dup} repeated)")
    print(f"\nzero-drone windows of T={T} available per video written to "
          f"{out_dir / 'zero_drone_frame_distribution.csv'}")


if __name__ == "__main__":
    main()
