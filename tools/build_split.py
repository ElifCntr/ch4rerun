"""Build and verify dvb_splits_v2.0-static.csv.

Nothing here is transcribed. The static subset is derived from
data/dvb_camera_motion.csv against the v1.0 manifest, session and scene come
from the manifest, and box counts are recomputed two independent ways: from the
manifest's own boxes column and by counting the label files.

Every figure below is an ASSERTION. Any mismatch stops the run with a non-zero
exit and a diff, and no split file is written. A recomputed figure that differs
means the inputs differ from what was measured. It is reported, never
reconciled, and the split is never adjusted to fit.

Asserted (from data/splits/expected_v2.0.json):
  77 videos labelled, 40 dynamic, 37 static
  static subset 37 videos, 30,788 boxes
  train 24 / 18,624, val 6 / 5,259, test 7 / 6,905
  zero session overlap between any pair of partitions
  all five surviving scenes present in train
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_root, load_config, repo_path, require  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

PARTITIONS = ("train", "val", "test")


class ParseError(RuntimeError):
    pass


# ----------------------------------------------------------------- inputs ---

def read_commented_csv(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def read_partitions(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("video\t"):
            continue
        video, part = line.split("\t")
        out.append((video, part.strip().lower()))
    return out


def load_manifest(path: Path) -> dict[str, dict]:
    rows = read_commented_csv(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    for needed in ("video", "session", "scene", "boxes"):
        if needed not in rows[0]:
            raise ValueError(
                f"{path} has no '{needed}' column. The split must not be "
                f"written without it.")
    return {r["video"]: r for r in rows}


def count_boxes(path: Path, fmt: str) -> tuple[int, int]:
    """Return (boxes, annotated_frames). Raises if the file does not validate."""
    boxes, frames = 0, set()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if fmt == "dvb_multibox_per_line":
                if len(toks) < 2:
                    raise ParseError(f"{path.name}:{lineno} too few fields")
                frame, n_obj = int(float(toks[0])), int(float(toks[1]))
                # Verified 12 Aug 2026 against the P5000 copy: each object
                # carries five fields, x y w h and a class name, not four.
                # 119,259 class tokens across all 77 files, all 'drone',
                # matching the v1.0 manifest box total exactly.
                if len(toks) != 2 + 5 * n_obj:
                    raise ParseError(
                        f"{path.name}:{lineno} declares {n_obj} objects but "
                        f"carries {len(toks) - 2} fields after the count, "
                        f"expected {5 * n_obj}. The format assumption is "
                        f"wrong; stop and re-sniff.")
                if n_obj:
                    frames.add(frame)
                boxes += n_obj
            elif fmt == "one_box_per_line":
                if len(toks) < 5:
                    raise ParseError(f"{path.name}:{lineno} too few fields")
                frames.add(int(float(toks[0])))
                boxes += 1
            else:
                raise ParseError(f"Unknown annotation format '{fmt}'")
    return boxes, len(frames)


def diff(label: str, measured, expected) -> str:
    delta = ""
    if isinstance(measured, int) and isinstance(expected, int):
        delta = f", difference {measured - expected:+d}"
    return f"{label}: measured {measured}, expected {expected}{delta}"


# ------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"],
                    cfg["seed"]["cudnn_benchmark"])

    root = data_root(cfg)
    ann_dir = root / require(cfg, "data.annotation_dir")
    ann_ext = require(cfg, "data.annotation_extension")
    fmt = require(cfg, "data.annotation_format")

    manifest = load_manifest(repo_path(cfg, require(cfg, "splits.v1_manifest")))
    expected = json.loads(
        repo_path(cfg, cfg["splits"]["expected"]).read_text(encoding="utf-8"))
    camera = read_commented_csv(repo_path(cfg, cfg["data"]["camera_motion"]))
    assignment = read_partitions(repo_path(cfg, cfg["splits"]["partitions"]))

    fail: list[str] = []
    note: list[str] = []

    # 1. Camera-motion labels against the manifest.
    labelled = {r["video"]: r["camera"].strip().lower() for r in camera}
    exp_cam = expected["camera_motion"]
    if len(labelled) != exp_cam["total"]:
        fail.append(diff("camera-motion rows", len(labelled), exp_cam["total"]))
    for state in ("static", "dynamic"):
        got = sum(1 for v in labelled.values() if v == state)
        if got != exp_cam[state]:
            fail.append(diff(f"camera-motion {state}", got, exp_cam[state]))

    not_in_manifest = sorted(set(labelled) - set(manifest))
    not_labelled = sorted(set(manifest) - set(labelled))
    if not_in_manifest:
        fail.append(f"labelled but absent from the v1.0 manifest: "
                    f"{not_in_manifest}")
    if not_labelled:
        fail.append(f"in the v1.0 manifest but unlabelled: {not_labelled}")

    # 2. Static subset derived, not assumed.
    static = sorted(v for v, c in labelled.items() if c == "static")
    exp_static = expected["static_subset"]
    if len(static) != exp_static["videos"]:
        fail.append(diff("static subset videos", len(static),
                         exp_static["videos"]))
    manifest_boxes = sum(int(manifest[v]["boxes"]) for v in static
                         if v in manifest)
    if manifest_boxes != exp_static["boxes"]:
        fail.append(diff("static subset boxes (from manifest column)",
                         manifest_boxes, exp_static["boxes"]))

    # 3. Partition assignment must be exactly the static subset.
    assigned = {v for v, _ in assignment}
    if assigned != set(static):
        extra = sorted(assigned - set(static))
        missing = sorted(set(static) - assigned)
        if extra:
            fail.append(f"assigned to a partition but not static: {extra}")
        if missing:
            fail.append(f"static but assigned to no partition: {missing}")

    # 4. Count boxes from the label files. Independent of the manifest column.
    rows: list[dict] = []
    counted_boxes: dict[str, int] = defaultdict(int)
    manifest_part_boxes: dict[str, int] = defaultdict(int)
    sessions: dict[str, set[str]] = defaultdict(set)

    for video, part in assignment:
        ann = ann_dir / f"{video}{ann_ext}"
        if not ann.exists():
            fail.append(f"missing label file: {ann}")
            continue
        if video not in manifest:
            continue  # already reported above
        try:
            boxes, frames = count_boxes(ann, fmt)
        except ParseError as exc:
            fail.append(f"parse: {exc}")
            continue

        meta = manifest[video]
        m_boxes = int(meta["boxes"])
        if boxes != m_boxes:
            note.append(diff(f"boxes for {video} (file vs manifest column)",
                             boxes, m_boxes))
        sessions[meta["session"]].add(part)
        counted_boxes[part] += boxes
        manifest_part_boxes[part] += m_boxes
        rows.append({"video": video, "session": meta["session"],
                     "scene": meta.get("scene", ""), "split": part,
                     "boxes": boxes, "annotated_frames": frames})

    # 5. Per-partition counts, both ways.
    for part in PARTITIONS:
        n_videos = sum(1 for r in rows if r["split"] == part)
        if n_videos != expected["videos"][part]:
            fail.append(diff(f"{part} videos", n_videos,
                             expected["videos"][part]))
        if counted_boxes[part] != expected["boxes"][part]:
            fail.append(diff(f"{part} boxes (counted from label files)",
                             counted_boxes[part], expected["boxes"][part]))
        if manifest_part_boxes[part] != expected["boxes"][part]:
            fail.append(diff(f"{part} boxes (from manifest column)",
                             manifest_part_boxes[part],
                             expected["boxes"][part]))

    # 6. Session overlap. This is the check that voided the original chapter.
    for session, parts in sorted(sessions.items()):
        if len(parts) > 1:
            members = sorted(r["video"] for r in rows if r["session"] == session)
            fail.append(f"SESSION LEAK: session '{session}' spans "
                        f"{sorted(parts)}. Videos: {', '.join(members)}")

    # 7. Scene coverage in train.
    train_scenes = {r["scene"] for r in rows if r["split"] == "train"}
    for scene in expected["scenes_required_in_train"]:
        if scene not in train_scenes:
            fail.append(f"scene '{scene}' expected in train but absent. "
                        f"Train scenes: {sorted(train_scenes)}")
    unexpected = train_scenes - set(expected["scenes_required_in_train"])
    if unexpected:
        fail.append(f"train carries unexpected scenes: {sorted(unexpected)}")

    # ------------------------------------------------------------- report ---
    out_dir = repo_path(cfg, cfg["reports"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "split_verification.txt"

    body = [
        "Chapter 4 split verification. Every figure is asserted.",
        "",
        f"camera-motion labels: {len(labelled)} rows, "
        f"{sum(1 for v in labelled.values() if v == 'static')} static, "
        f"{sum(1 for v in labelled.values() if v == 'dynamic')} dynamic",
        f"static subset boxes, manifest column: {manifest_boxes}",
        f"boxes counted from label files: {dict(counted_boxes)}",
        f"boxes from manifest column:     {dict(manifest_part_boxes)}",
        f"expected:                       {expected['boxes']}",
        "",
        "sessions per partition:",
    ]
    for part in PARTITIONS:
        body.append(f"  {part}: "
                    f"{sorted({r['session'] for r in rows if r['split'] == part})}")
    body += ["", "scenes per partition:"]
    for part in PARTITIONS:
        body.append(f"  {part}: "
                    f"{sorted({r['scene'] for r in rows if r['split'] == part})}")
    if note:
        body += ["", "NOTES (not fatal, but reported):"] + [f"  {n}" for n in note]
    body += ["", "DISCREPANCIES:"] + ([f"  {f}" for f in fail] or
                                      ["  none, all assertions hold"])
    report.write_text("\n".join(body), encoding="utf-8")

    if fail:
        print("SPLIT NOT WRITTEN. Assertions failed:\n")
        for f in fail:
            print("  " + f)
        if note:
            print("\nAlso noted:")
            for n in note:
                print("  " + n)
        print(f"\nFull report: {report}")
        print("Report the discrepancy. Do not reconcile it or adjust the split.")
        sys.exit(1)

    split_path = repo_path(cfg, cfg["splits"]["file"])
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video", "session", "scene",
                                                "split", "boxes",
                                                "annotated_frames"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["split"], r["video"])))

    print(f"All assertions hold. Written: {split_path}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
