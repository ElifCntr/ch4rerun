"""Stage 1a per-instance measurement. Train and val only, ground truth only.

Covers item B first half (size distribution of annotated instances), item C
(track length) and item D (drift). Items G1 and H read the CSVs this writes,
so they are separate scripts and are not run from here.

NOTHING IN THIS FILE CHOOSES A PARAMETER. The only numeric inputs are the
manager's candidate window lengths for cumulative drift, which are a reporting
grid rather than a choice, and they live in the config.

TRACK ASSOCIATION, ruled 12 August 2026.
The DvB annotations carry no track identifiers. Nothing links a box in one
frame to the same drone in the next. Measured over train and val: 21,335
frames with one drone, 1,274 with two, 1,813 with none, never more than two.
The multi-drone frames are almost entirely gopro_000, gopro_001 and gopro_002.

So association is needed only where a frame carries more than one box, and it
is INFERENCE, which Stage 1a otherwise forbids. The ruling is therefore:

  - nearest-centre association, applied only in the frames that need it
  - the ratio of second-best to best assignment cost is recorded per decision
    and reported as a DISTRIBUTION, not as a rate against a threshold, since a
    rate would need a cut-off and none is being supplied
  - every row carries `associated`, true where a track identifier came from
    the rule and false where it came from the data
  - items C and D are reported both with and without the three gopro videos,
    so the rule's effect on the distributions is visible rather than argued
  - the method section labels the association as inference

BOXES PAST THE FRAME EDGE. Verified on the P5000 copy: some annotations have
negative coordinates or extend beyond the frame. Both the raw box area and the
area clipped to the frame are recorded, with a `clipped` flag, because a
partly visible drone has a smaller apparent size than its box implies and item
B would otherwise overstate it.

FRAME RATES. The static subset carries three rates. Every frame-counted
quantity gets a companion column in seconds, and nothing is pooled across
rate groups without that column present.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_root, load_config, repo_path, require  # noqa: E402
from src.seeding import seed_everything  # noqa: E402
from src.splits import load_split  # noqa: E402


# ------------------------------------------------------------- annotations --

class ParseError(RuntimeError):
    pass


def read_boxes(path: Path, fmt: str) -> dict[int, list[tuple[float, ...]]]:
    """Return {frame_index: [(x, y, w, h), ...]}. Frames with no box are kept
    with an empty list, because item F needs to know which frames are empty."""
    if fmt != "dvb_multibox_per_line":
        raise ParseError(f"Unsupported annotation format '{fmt}'")

    frames: dict[int, list[tuple[float, ...]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            t = line.split()
            frame, n_obj = int(float(t[0])), int(float(t[1]))
            # Five fields per object: x y w h class. Verified 12 Aug 2026.
            if len(t) != 2 + 5 * n_obj:
                raise ParseError(
                    f"{path.name}:{lineno} declares {n_obj} objects but "
                    f"carries {len(t) - 2} fields after the count, expected "
                    f"{5 * n_obj}. Stop and re-sniff the format.")
            boxes = []
            for i in range(n_obj):
                x, y, w, h = (float(v) for v in t[2 + 5 * i: 6 + 5 * i])
                boxes.append((x, y, w, h))
            frames[frame] = boxes
    return frames


# ------------------------------------------------------------- association --

def centre(b: tuple[float, ...]) -> tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def associate(prev: list[tuple[float, ...]],
              curr: list[tuple[float, ...]]) -> tuple[list[int], float | None]:
    """Map each box in curr to an index into prev by nearest centre.

    Returns (mapping, cost_ratio). mapping[i] is the index in prev that curr[i]
    continues, or -1 for a new track. cost_ratio is second-best total cost over
    best total cost, or None when no choice was made (one box, or differing
    counts). Ratio near 1 means the two assignments were nearly as good as each
    other, so the association was close to arbitrary.

    No maximum-displacement gate is applied, because that would be a threshold
    and Stage 1a sets none. Where the counts differ the extra boxes start new
    tracks, which is a definition rather than a parameter.
    """
    if not prev:
        return [-1] * len(curr), None
    if len(curr) == 1 and len(prev) == 1:
        return [0], None
    if len(curr) != len(prev):
        # Unequal counts. Assign greedily by nearest centre, remainder is new.
        taken: set[int] = set()
        mapping = []
        for c in curr:
            best, best_d = -1, None
            for j, p in enumerate(prev):
                if j in taken:
                    continue
                d = dist(centre(c), centre(p))
                if best_d is None or d < best_d:
                    best, best_d = j, d
            if best >= 0:
                taken.add(best)
            mapping.append(best)
        return mapping, None

    # Equal counts, more than one box: score every permutation.
    costs = []
    for perm in permutations(range(len(prev))):
        total = sum(dist(centre(curr[i]), centre(prev[perm[i]]))
                    for i in range(len(curr)))
        costs.append((total, perm))
    costs.sort()
    best_cost, best_perm = costs[0]
    ratio = (costs[1][0] / best_cost) if len(costs) > 1 and best_cost > 0 else None
    return list(best_perm), ratio


# ----------------------------------------------------------------- measure --

def measure_video(video: str, meta: dict, frames: dict[int, list],
                  windows: list[int], needs_assoc: bool):
    """Return (box_rows, track_rows, drift_rows, ratio_rows, empty_frames)."""
    width, height = int(meta["width"]), int(meta["height"])
    fps = float(meta["fps"])
    frame_area = float(width * height)

    box_rows, drift_rows, ratio_rows = [], [], []
    empty_frames = 0

    # Walk frames in order, carrying track identifiers forward.
    ordered = sorted(frames)
    prev_boxes: list[tuple[float, ...]] = []
    prev_ids: list[int] = []
    next_id = 0
    prev_frame: int | None = None
    tracks: dict[int, list[dict]] = defaultdict(list)

    for f in ordered:
        boxes = frames[f]
        if not boxes:
            empty_frames += 1
            prev_boxes, prev_ids, prev_frame = [], [], None
            continue

        contiguous = prev_frame is not None and f == prev_frame + 1
        if contiguous:
            mapping, ratio = associate(prev_boxes, boxes)
            if ratio is not None:
                ratio_rows.append({"video": video, "frame": f,
                                   "cost_ratio": round(ratio, 6),
                                   "n_boxes": len(boxes)})
        else:
            mapping, ratio = [-1] * len(boxes), None

        ids = []
        for i, b in enumerate(boxes):
            j = mapping[i] if i < len(mapping) else -1
            if j >= 0 and j < len(prev_ids):
                ids.append(prev_ids[j])
            else:
                ids.append(next_id)
                next_id += 1

        for b, tid in zip(boxes, ids):
            x, y, w, h = b
            cx, cy = centre(b)
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(width), x + w), min(float(height), y + h)
            vis_w, vis_h = max(0.0, x1 - x0), max(0.0, y1 - y0)
            row = {
                "video": video, "split": meta["split"], "scene": meta["scene"],
                "session": meta["session"], "resolution": f"{width}x{height}",
                "fps": fps, "frame": f, "track_id": tid,
                "associated": bool(needs_assoc and len(boxes) > 1),
                "x": x, "y": y, "w": w, "h": h,
                "cx": round(cx, 3), "cy": round(cy, 3),
                "box_area": round(w * h, 3),
                "visible_area": round(vis_w * vis_h, 3),
                "clipped": bool(vis_w * vis_h < w * h - 1e-6),
                "area_frac": round(w * h / frame_area, 9),
                "equiv_side": round(math.sqrt(max(w * h, 0.0)), 4),
                "equiv_side_frac": round(
                    math.sqrt(max(w * h, 0.0) / frame_area), 9),
            }
            box_rows.append(row)
            tracks[tid].append(row)

        prev_boxes, prev_ids, prev_frame = boxes, ids, f

    # Item C, track length. Item D, drift.
    track_rows = []
    for tid, rows in tracks.items():
        rows.sort(key=lambda r: r["frame"])
        n = len(rows)
        assoc = any(r["associated"] for r in rows)
        steps = []
        for a, b in zip(rows, rows[1:]):
            if b["frame"] != a["frame"] + 1:
                continue
            d = math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])
            size = math.sqrt(max(a["box_area"], 1e-9))
            steps.append(d)
            drift_rows.append({
                "video": video, "split": meta["split"], "track_id": tid,
                "frame": a["frame"], "fps": fps, "associated": assoc,
                "dx": round(b["cx"] - a["cx"], 4),
                "dy": round(b["cy"] - a["cy"], 4),
                "displacement_px": round(d, 4),
                "displacement_per_s": round(d * fps, 4),
                "displacement_over_size": round(d / size, 6),
            })

        cum = {}
        for W in windows:
            # Cumulative centre displacement over a window of W frames,
            # maximum over all positions of that window within the track.
            best = None
            for s in range(0, max(0, n - W + 1)):
                a, b = rows[s], rows[s + W - 1]
                if b["frame"] - a["frame"] != W - 1:
                    continue
                d = math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])
                best = d if best is None else max(best, d)
            cum[f"cum_drift_T{W}_px"] = None if best is None else round(best, 4)
            if best is not None:
                size = math.sqrt(max(rows[0]["box_area"], 1e-9))
                cum[f"cum_drift_T{W}_over_size"] = round(best / size, 6)
            else:
                cum[f"cum_drift_T{W}_over_size"] = None

        row = {
            "video": video, "split": meta["split"], "scene": meta["scene"],
            "resolution": f"{width}x{height}", "fps": fps, "track_id": tid,
            "associated": assoc,
            "length_frames": n,
            "length_seconds": round(n / fps, 4),
            "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
            "mean_step_px": round(sum(steps) / len(steps), 4) if steps else None,
            "max_step_px": round(max(steps), 4) if steps else None,
        }
        row.update(cum)
        track_rows.append(row)

    return box_rows, track_rows, drift_rows, ratio_rows, empty_frames


# -------------------------------------------------------------------- main --

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"],
                           cfg["seed"]["cudnn_benchmark"])

    root = data_root(cfg)
    ann_dir = root / require(cfg, "data.annotation_dir")
    ann_ext = require(cfg, "data.annotation_extension")
    fmt = require(cfg, "data.annotation_format")
    windows = require(cfg, "measurement.drift_windows")

    # Train and val only. allow_test is not passed and must never be.
    split = load_split(repo_path(cfg, cfg["splits"]["file"]), ["train", "val"])
    inv = {r["video"]: r for r in csv.DictReader(
        open(repo_path(cfg, cfg["reports"]["dir"]) / "video_inventory.csv",
             encoding="utf-8"))}

    all_boxes, all_tracks, all_drift, all_ratios = [], [], [], []
    empty_rows = []

    for part in ("train", "val"):
        for row in split[part]:
            video = row["video"]
            if video not in inv:
                raise KeyError(f"{video} missing from video_inventory.csv; "
                               f"run tools/inventory.py --stage videos first")
            iv = inv[video]
            frames = read_boxes(ann_dir / f"{video}{ann_ext}", fmt)
            needs = any(len(b) > 1 for b in frames.values())
            meta = {
                "split": part, "scene": row["scene"], "session": row["session"],
                "width": iv["ffprobe_width"] or iv["cv2_width"],
                "height": iv["ffprobe_height"] or iv["cv2_height"],
                "fps": iv["ffprobe_avg_frame_rate"] or iv["cv2_fps"],
            }
            b, t, d, r, empty = measure_video(video, meta, frames, windows, needs)
            all_boxes += b
            all_tracks += t
            all_drift += d
            all_ratios += r
            empty_rows.append({
                "video": video, "split": part, "scene": row["scene"],
                "zero_drone_frames": empty,
                "annotated_frames": sum(1 for v in frames.values() if v),
                "needs_association": needs,
            })

    out = repo_path(cfg, cfg["reports"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "instances_boxes.csv", all_boxes)
    write_csv(out / "instances_tracks.csv", all_tracks)
    write_csv(out / "instances_drift.csv", all_drift)
    write_csv(out / "association_cost_ratios.csv", all_ratios)
    write_csv(out / "zero_drone_frames.csv", empty_rows)

    assoc_tracks = sum(1 for t in all_tracks if t["associated"])
    clipped = sum(1 for b in all_boxes if b["clipped"])
    print(f"seed                     {seed}")
    print(f"boxes                    {len(all_boxes)}")
    print(f"  clipped by frame edge  {clipped}")
    print(f"tracks                   {len(all_tracks)}")
    print(f"  from association       {assoc_tracks}")
    print(f"drift steps              {len(all_drift)}")
    print(f"association decisions    {len(all_ratios)}")
    print(f"zero-drone frames        {sum(r['zero_drone_frames'] for r in empty_rows)}")
    print(f"videos needing assoc     "
          f"{[r['video'] for r in empty_rows if r['needs_association']]}")
    print(f"\nwritten to {out}")
    print("Items C and D are reported both ways downstream by filtering on "
          "the `associated` column. No threshold is applied here.")


if __name__ == "__main__":
    main()
