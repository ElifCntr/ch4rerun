"""Stage 1a item E. Crop-retention strips, generation and scoring sheet.

Ruled 12 August 2026. Reads reports/instances_boxes.csv, decodes the sampled
frames, writes contact sheets to the ignored reports/strips/, and writes two
committed artefacts: the sampling manifest and a blank scoring sheet.

WHAT E ASKS THAT H CANNOT. Item H measured whether the box stays inside the
crop. E asks whether the target is USABLE inside it: identifiable, not smeared,
not lost against background. Only a human can answer that, so E's output is a
scoring sheet, not a number.

TWO PADDING CONDITIONS, ruled rather than one or a grid.
  A. p = 0.3 at every T. Inherited, resolution-preserving; retention falls as
     T rises (item H: 0.905, 0.799, 0.684, 0.508).
  B. p set per T from item H to hold retention near 0.9 where reachable:
     T=3 0.3, T=5 0.5, T=8 0.75, T=16 1.0 (item H: 0.905, 0.909, 0.889,
     0.807). At T=16 0.9 is NOT reachable and 0.807 is the best available;
     the report must say so rather than implying the condition held.
  At T=3 the two conditions coincide, so that strip is generated once.

Either condition alone confounds one dimension. A alone re-measures
containment, which H already did. B alone shrinks the target as T rises. The
pair is what the chapter needs to show.

STRIDE. Stride 1 only in the scoring pass, because stride changes temporal
span and that is already understood arithmetically, whereas padding changes
what can be seen. Stride 2 images are generated in the same run so the
confirmation pass costs no further decoding.

SAMPLING. 15 tubelets, 3 per scene across all five scenes, seeded. The
sampling unit is a TUBELET, a (track, centre frame) pair, not a track, because
lake has only two tracks in train and val and could not otherwise supply
three. Where a scene has three or more tracks, the three tubelets come from
three different tracks.

WHAT THE PLAN CANNOT DO, and does not claim to. With 15 tubelets over five
scenes, rate-group separation and scene separation cannot both hold. Scene is
the stronger stratifier and correlates with rate anyway, since industrial is
the 25 fps group. So: sample by scene, record rate per strip, analyse by
scene.

TWO DELIBERATE OMISSIONS FROM THE CONTACT SHEETS.
  No bounding box is drawn. A drawn box tells the scorer where to look, which
  is exactly what "identifiable" is testing.
  No condition or padding label appears on the image. Only an opaque strip id.
  Order is randomised and about a tenth of strips are duplicated, so neither
  condition nor repetition is visible to the scorer.

Frames are upscaled for display by NEAREST NEIGHBOUR only, so nothing in the
strip is detail the network would not receive.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_root, load_config, repo_path, require  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

DISPLAY_SCALE = 3          # nearest-neighbour, display only
CROP_PX = 112              # the network input size, ruled
GUTTER = 6


def window_frames(centre: int, T: int, stride: int) -> list[int]:
    """Frame indices of a tubelet. For even T there is no exact centre; the
    convention here is offset -(T-1)//2, so T=8 runs centre-3 to centre+4."""
    lead = (T - 1) // 2
    return [centre + stride * (i - lead) for i in range(T)]


def crop_box(cx: float, cy: float, w: float, h: float, p: float):
    """Square crop of side (1+2p)*max(w,h) on the centre-frame box centre.
    The ruled square-pad-in-original-frame geometry."""
    side = max(w, h) * (1.0 + 2.0 * p)
    return cx - side / 2.0, cy - side / 2.0, side


def extract(frame: np.ndarray, x0: float, y0: float, side: float) -> np.ndarray:
    """Crop with real image context where available, zero padding past the
    frame edge, then resize to the network input size."""
    H, W = frame.shape[:2]
    xi, yi, si = int(round(x0)), int(round(y0)), max(1, int(round(side)))
    out = np.zeros((si, si, 3), np.uint8)
    sx0, sy0 = max(0, xi), max(0, yi)
    sx1, sy1 = min(W, xi + si), min(H, yi + si)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - yi:sy1 - yi, sx0 - xi:sx1 - xi] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA)


def contact_sheet(crops: list[np.ndarray], strip_id: str) -> np.ndarray:
    s = CROP_PX * DISPLAY_SCALE
    tiles = [cv2.resize(c, (s, s), interpolation=cv2.INTER_NEAREST)
             for c in crops]
    header = 26
    W = len(tiles) * s + (len(tiles) - 1) * GUTTER
    sheet = np.full((header + s, W, 3), 32, np.uint8)
    for i, t in enumerate(tiles):
        x = i * (s + GUTTER)
        sheet[header:header + s, x:x + s] = t
    cv2.putText(sheet, strip_id, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"],
                           cfg["seed"]["cudnn_benchmark"])
    rng = random.Random(seed)

    root = data_root(cfg)
    video_dir = root / require(cfg, "data.video_dir")
    exts = {e.lower() for e in require(cfg, "data.video_extensions")}
    out_dir = repo_path(cfg, cfg["reports"]["dir"])
    strips_dir = repo_path(cfg, cfg["reports"]["strips_dir"])
    man_dir = repo_path(cfg, cfg["reports"]["strip_manifest_dir"])
    sheet_dir = repo_path(cfg, cfg["reports"]["scoring_sheet_dir"])
    for d in (strips_dir, man_dir, sheet_dir):
        d.mkdir(parents=True, exist_ok=True)

    e = require(cfg, "item_e")
    T_VALUES = e["t_values"]
    STRIDES = e["strides"]
    COND_A = e["condition_a_padding"]
    COND_B = e["condition_b_padding"]          # {T: p}
    N_PER_SCENE = e["tubelets_per_scene"]
    DUP_FRACTION = e["duplicate_fraction"]

    boxes = list(csv.DictReader(
        open(out_dir / "instances_boxes.csv", encoding="utf-8")))
    tracks: dict[tuple, list[dict]] = defaultdict(list)
    for b in boxes:
        tracks[(b["video"], b["track_id"])].append(b)
    for rows in tracks.values():
        rows.sort(key=lambda r: int(r["frame"]))

    # A tubelet must fit the widest window: T=16 at stride 2 spans 31 frames.
    max_lead = max((T - 1) // 2 * s for T in T_VALUES for s in STRIDES)
    max_tail = max((T - 1 - (T - 1) // 2) * s
                   for T in T_VALUES for s in STRIDES)

    by_scene: dict[str, list[tuple]] = defaultdict(list)
    for key, rows in tracks.items():
        # Contiguous runs only; a gap already ended the track upstream.
        if int(rows[-1]["frame"]) - int(rows[0]["frame"]) != len(rows) - 1:
            continue
        lo = int(rows[0]["frame"]) + max_lead
        hi = int(rows[-1]["frame"]) - max_tail
        if hi < lo:
            continue
        by_scene[rows[0]["scene"]].append((key, lo, hi))

    chosen = []
    for scene in sorted(by_scene):
        cands = sorted(by_scene[scene])
        rng.shuffle(cands)
        picked, used_tracks = [], set()
        # Prefer distinct tracks; fall back to more positions in the same one.
        for pool in (0, 1):
            for key, lo, hi in cands:
                if len(picked) >= N_PER_SCENE:
                    break
                if pool == 0 and key in used_tracks:
                    continue
                picked.append((key, rng.randint(lo, hi)))
                used_tracks.add(key)
        if len(picked) < N_PER_SCENE:
            print(f"WARNING: scene {scene} yielded only {len(picked)} "
                  f"tubelets, wanted {N_PER_SCENE}")
        chosen += [(scene, k, c) for k, c in picked[:N_PER_SCENE]]

    # Build the strip list. Condition A and B coincide wherever their p match.
    strips = []
    for scene, key, centre in chosen:
        video, tid = key
        rows = {int(r["frame"]): r for r in tracks[key]}
        c = rows[centre]
        for T in T_VALUES:
            conds = {"A": COND_A}
            if abs(COND_B[str(T)] - COND_A) > 1e-9:
                conds["B"] = COND_B[str(T)]
            else:
                conds = {"AB": COND_A}
            for cond, p in conds.items():
                for stride in STRIDES:
                    frames = window_frames(centre, T, stride)
                    strips.append({
                        "video": video, "track_id": tid, "scene": scene,
                        "resolution": c["resolution"], "fps": c["fps"],
                        "centre_frame": centre, "T": T, "stride": stride,
                        "condition": cond, "padding_fraction": p,
                        "frame_indices": ";".join(str(f) for f in frames),
                        "centre_box_w": c["w"], "centre_box_h": c["h"],
                        "centre_equiv_side": c["equiv_side"],
                        "seed": seed,
                    })
    for i, s in enumerate(strips):
        s["strip_id"] = f"E{i:04d}"

    # ---------------------------------------------------------- decoding ---
    needed: dict[str, set[int]] = defaultdict(set)
    for s in strips:
        needed[s["video"]].update(int(f) for f in s["frame_indices"].split(";"))

    frames_cache: dict[tuple, np.ndarray] = {}
    for video, want in sorted(needed.items()):
        matches = [pth for pth in video_dir.rglob("*")
                   if pth.stem == video and pth.suffix.lower() in exts]
        if not matches:
            sys.exit(f"video not found for {video}")
        cap = cv2.VideoCapture(str(matches[0]))
        if not cap.isOpened():
            sys.exit(f"could not open {matches[0]}")
        # Sequential read. Seeking is unreliable on some of these containers,
        # notably the single .mpg, and correctness matters more than speed.
        target = sorted(want)
        idx, ptr = 0, 0
        while ptr < len(target):
            ok, frame = cap.read()
            if not ok:
                print(f"WARNING: {video} ended at frame {idx}, "
                      f"{len(target) - ptr} frames unread")
                break
            while ptr < len(target) and target[ptr] == idx:
                frames_cache[(video, idx)] = frame.copy()
                ptr += 1
            idx += 1
        cap.release()
        print(f"decoded {video}: {len(want)} frames")

    # ------------------------------------------------------------ strips ---
    written = 0
    for s in strips:
        rows = {int(r["frame"]): r for r in tracks[(s["video"], s["track_id"])]}
        c = rows[s["centre_frame"]]
        x0, y0, side = crop_box(float(c["cx"]), float(c["cy"]),
                                float(c["w"]), float(c["h"]),
                                s["padding_fraction"])
        crops = []
        missing = False
        for f in (int(v) for v in s["frame_indices"].split(";")):
            frame = frames_cache.get((s["video"], f))
            if frame is None:
                missing = True
                break
            crops.append(extract(frame, x0, y0, side))
        if missing:
            print(f"WARNING: {s['strip_id']} skipped, frames missing")
            continue
        cv2.imwrite(str(strips_dir / f"{s['strip_id']}.png"),
                    contact_sheet(crops, s["strip_id"]))
        written += 1

    # --------------------------------------------------------- artefacts ---
    with open(man_dir / "item_e_sampling_manifest.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(strips[0].keys()))
        w.writeheader()
        w.writerows(strips)

    # Scoring sheet: stride 1 only, randomised, with duplicates for
    # intra-rater consistency. The duplicate map is committed separately so
    # the sheet itself gives nothing away.
    pass1 = [s for s in strips if s["stride"] == 1]
    n_dup = max(1, int(round(len(pass1) * DUP_FRACTION)))
    dups = rng.sample(pass1, n_dup)
    sheet = [{"sheet_row": None, "strip_id": s["strip_id"],
              "score": "", "usable_frames": "", "note": ""}
             for s in pass1]
    dup_map = []
    for i, s in enumerate(dups):
        rid = f"E9{i:03d}"
        sheet.append({"sheet_row": None, "strip_id": rid, "score": "",
                      "usable_frames": "", "note": ""})
        dup_map.append({"repeat_id": rid, "original_id": s["strip_id"]})
        # Regenerate rather than copy, so the repeat carries its own id in
        # the header and is not recognisable as a duplicate.
        rows = {int(r["frame"]): r for r in tracks[(s["video"], s["track_id"])]}
        c = rows[s["centre_frame"]]
        x0, y0, side = crop_box(float(c["cx"]), float(c["cy"]),
                                float(c["w"]), float(c["h"]),
                                s["padding_fraction"])
        crops = [extract(frames_cache[(s["video"], f)], x0, y0, side)
                 for f in (int(v) for v in s["frame_indices"].split(";"))
                 if (s["video"], f) in frames_cache]
        if len(crops) == s["T"]:
            cv2.imwrite(str(strips_dir / f"{rid}.png"),
                        contact_sheet(crops, rid))
    rng.shuffle(sheet)
    for i, row in enumerate(sheet, 1):
        row["sheet_row"] = i

    with open(sheet_dir / "item_e_scoring_sheet_BLANK.csv", "w", newline="",
              encoding="utf-8") as fh:
        fh.write("# Item E scoring sheet. One row per strip, presented in "
                 "randomised order.\n")
        fh.write("# score: one of  usable / degrades / not_usable / cannot_tell\n")
        fh.write("#   usable      target identifiable in every frame\n")
        fh.write("#   degrades    identifiable at the centre frame, lost, "
                 "clipped or unrecognisable in at least one other\n")
        fh.write("#   not_usable  not identifiable even at the centre frame\n")
        fh.write("#   cannot_tell too few pixels or too ambiguous to judge\n")
        fh.write("# usable_frames: how many of the strip's frames show an "
                 "identifiable drone. Required for 'degrades', optional "
                 "otherwise (usable = all, not_usable = 0).\n")
        fh.write("#   Carries the severity gradient the four-way score "
                 "cannot: losing 1 frame of 16 and losing 7 of 16 are both "
                 "'degrades' but are not the same result.\n")
        fh.write("# note: optional free text. Use it to say WHERE the "
                 "failures fell if they were scattered rather than at the "
                 "ends of the window.\n")
        w = csv.DictWriter(fh, fieldnames=["sheet_row", "strip_id", "score",
                                           "usable_frames", "note"])
        w.writeheader()
        w.writerows(sheet)

    with open(man_dir / "item_e_duplicate_map.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["repeat_id", "original_id"])
        w.writeheader()
        w.writerows(dup_map)

    print(f"\nseed                {seed}")
    print(f"tubelets            {len(chosen)}  "
          f"({N_PER_SCENE} per scene x {len(by_scene)} scenes)")
    print(f"strips generated    {written} of {len(strips)}")
    print(f"scoring sheet rows  {len(sheet)}  "
          f"(stride 1 only, {n_dup} duplicated for consistency)")
    print(f"\nimages   {strips_dir}   (NOT committed)")
    print(f"manifest {man_dir}   (committed)")
    print(f"sheet    {sheet_dir}   (committed once filled in)")


if __name__ == "__main__":
    main()
