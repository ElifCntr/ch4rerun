"""Stage 1a, Part 2 and item A.

--stage layout      reports what is actually at the data path. Assumes nothing.
--stage videos      per-video inventory. Frame rate is read from the file itself.
--stage continuity  the gopro_000-008 continuity check ordered on 11 Aug.

No stage writes into the dataset.

TEST-SET NOTE. The continuity check names gopro_004, which is in the test
partition. It therefore requires --inspect-test-metadata, and it appends to a
committed access log. Container metadata only: duration, resolution, frame
rate, creation timestamp and file size. No frames are decoded and no
annotation file is read. Anything beyond container metadata, a frame
comparison across the join or a check on annotation continuity, is a real
test-set access and needs its own ruling before it is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_root, load_config, repo_path, require  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}
TEXTLIKE_EXT = {".txt", ".csv", ".json", ".xml", ".ann", ".label", ".labels"}


# ---------------------------------------------------------------- layout ----

def walk_layout(root: Path, max_depth: int = 4) -> dict:
    dirs: dict[str, Counter] = defaultdict(Counter)
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        parent = str(rel.parent) if str(rel.parent) != "." else "."
        if path.is_dir():
            dirs[parent]["<dir> " + path.name] += 1
        else:
            dirs[parent][path.suffix.lower() or "<no extension>"] += 1
    return {k: dict(v) for k, v in sorted(dirs.items())}


def sniff_annotation(path: Path, n_lines: int = 5) -> dict:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                break
            lines.append(line.rstrip("\n"))

    verdict = "unknown"
    if lines:
        toks = lines[0].split()
        if len(toks) >= 2 and all(t.lstrip("-").replace(".", "", 1).isdigit()
                                  for t in toks):
            try:
                n_obj = int(float(toks[1]))
                if len(toks) == 2 + 4 * n_obj:
                    verdict = "dvb_multibox_per_line"
                elif len(toks) in (5, 6):
                    verdict = "one_box_per_line"
            except ValueError:
                pass
    return {"file": path.name, "sample_lines": lines, "sniffed_format": verdict}


def stage_layout(cfg: dict, out_dir: Path) -> None:
    root = data_root(cfg)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. This script is meant to run on the machine "
            f"holding the dataset."
        )

    layout = walk_layout(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    textish = [p for p in files if p.suffix.lower() in TEXTLIKE_EXT]

    static_dirs = [str(p.relative_to(root)) for p in root.rglob("*")
                   if p.is_dir() and "static" in p.name.lower()]

    video_dirs = Counter(str(p.parent.relative_to(root)) for p in videos)
    ann_dirs = Counter(str(p.parent.relative_to(root)) for p in textish)

    stems_v = {p.stem for p in videos}
    stems_a = {p.stem for p in textish}
    paired = sorted(stems_v & stems_a)
    video_only = sorted(stems_v - stems_a)
    ann_only = sorted(stems_a - stems_v)

    samples = [sniff_annotation(p) for p in textish[:5]]
    formats = Counter(s["sniffed_format"] for s in samples)

    report = {
        "root": str(root),
        "counts": {
            "files_total": len(files),
            "videos": len(videos),
            "text_like_files": len(textish),
        },
        "directory_contents": layout,
        "video_directories": dict(video_dirs),
        "annotation_directories": dict(ann_dirs),
        "video_extensions": dict(Counter(p.suffix.lower() for p in videos)),
        "annotation_extensions": dict(Counter(p.suffix.lower() for p in textish)),
        "static_camera_folders_found": static_dirs,
        "pairing_by_stem": {
            "paired": len(paired),
            "video_without_annotation": video_only,
            "annotation_without_video": ann_only,
        },
        "annotation_samples": samples,
        "sniffed_format_agreement": dict(formats),
    }

    out = out_dir / "layout_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"root                     {root}")
    print(f"videos                   {len(videos)}")
    print(f"text-like files          {len(textish)}")
    print(f"paired by stem           {len(paired)}")
    print(f"video without annotation {len(video_only)}")
    print(f"annotation without video {len(ann_only)}")
    print(f"static_camera folders    {static_dirs or 'NONE FOUND'}")
    print(f"sniffed format           {dict(formats)}")
    print(f"\nwritten to {out}")
    print("Fill data.video_dir, data.annotation_dir, data.video_extensions, "
          "data.annotation_extension and data.annotation_format in the config "
          "from this report, commit, then rerun with --stage videos.")


# ---------------------------------------------------------------- videos ----

def ffprobe_video(path: Path, with_tags: bool = False) -> dict:
    if shutil.which("ffprobe") is None:
        return {"ffprobe": "not available"}
    entries = ("stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,"
               "duration")
    if with_tags:
        entries += " : format=duration,size : format_tags=creation_time"
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", entries.replace(" ", ""),
        "-of", "json", str(path),
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True,
                             check=True).stdout
        parsed = json.loads(raw)
        stream = parsed["streams"][0]
        fmt = parsed.get("format", {})
    except Exception as exc:  # noqa: BLE001
        return {"ffprobe_error": str(exc)}

    def as_fps(value):
        if not value or "/" not in str(value):
            return None
        num, den = str(value).split("/")
        return float(num) / float(den) if float(den) else None

    return {
        "ffprobe_width": stream.get("width"),
        "ffprobe_height": stream.get("height"),
        "ffprobe_r_frame_rate": as_fps(stream.get("r_frame_rate")),
        "ffprobe_avg_frame_rate": as_fps(stream.get("avg_frame_rate")),
        "ffprobe_nb_frames": (int(stream["nb_frames"])
                              if stream.get("nb_frames") else None),
        "ffprobe_duration_s": (float(stream["duration"])
                               if stream.get("duration") else None),
        "container_duration_s": (float(fmt["duration"])
                                 if fmt.get("duration") else None),
        "file_size_bytes": int(fmt["size"]) if fmt.get("size") else None,
        "creation_time": fmt.get("tags", {}).get("creation_time"),
    }


def cv2_video(path: Path) -> dict:
    try:
        import cv2
    except ImportError:
        return {"cv2": "not available"}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"cv2_error": "could not open"}
    info = {
        "cv2_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "cv2_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "cv2_fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "cv2_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def stage_videos(cfg: dict, out_dir: Path) -> None:
    root = data_root(cfg)
    video_dir = root / require(cfg, "data.video_dir")
    exts = {e.lower() for e in require(cfg, "data.video_extensions")}

    videos = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in exts)
    if not videos:
        raise FileNotFoundError(f"No videos with extensions {exts} under "
                                f"{video_dir}")

    rows, disagreements = [], []
    for path in videos:
        row = {"video": path.stem, "path": str(path.relative_to(root))}
        row.update(ffprobe_video(path, with_tags=True))
        row.update(cv2_video(path))

        fps_probe = row.get("ffprobe_avg_frame_rate")
        fps_cv2 = row.get("cv2_fps")
        if fps_probe and fps_cv2 and abs(fps_probe - fps_cv2) > 0.01:
            disagreements.append(
                f"{path.stem}: ffprobe avg_frame_rate {fps_probe:.4f} vs "
                f"cv2 CAP_PROP_FPS {fps_cv2:.4f}")

        r_fps = row.get("ffprobe_r_frame_rate")
        if fps_probe and r_fps and abs(fps_probe - r_fps) > 0.01:
            disagreements.append(
                f"{path.stem}: variable frame rate, r_frame_rate {r_fps:.4f} "
                f"vs avg_frame_rate {fps_probe:.4f}")

        n_frames = row.get("ffprobe_nb_frames") or row.get("cv2_frame_count")
        fps = fps_probe or fps_cv2
        row["frames"] = n_frames
        row["duration_s"] = (round(n_frames / fps, 3)
                             if n_frames and fps else None)
        rows.append(row)

    fields = sorted({k for r in rows for k in r})
    csv_path = out_dir / "video_inventory.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "video_inventory_disagreements.txt").write_text(
        "\n".join(disagreements) or "none", encoding="utf-8")

    print(f"videos inventoried  {len(rows)}")
    print(f"disagreements       {len(disagreements)}")
    print(f"written to          {csv_path}")
    print("Split column is added by tools/build_split.py, not here, so that "
          "the inventory cannot depend on an unverified split.")


# ------------------------------------------------------------ continuity ---

GOPRO_GROUP = [f"gopro_{i:03d}" for i in range(9)]


def stage_continuity(cfg: dict, out_dir: Path, allow_test_metadata: bool,
                     names: list[str]) -> None:
    """Is gopro_004 (test) a continuation of gopro_005 (val), or of the rest?

    Reports all nine clips side by side rather than only the two named ones.
    Two clips from one session share resolution and frame rate by
    construction, so agreement on those is not evidence of continuity, only
    absence of contradiction. The pattern across the whole group carries the
    argument: if the seven training clips show a regular cadence, with
    consecutive creation times separated by about one clip duration, then
    004 and 005 sitting on that same cadence is evidence they belong to one
    continuous recording that was cut into numbered clips.
    """
    test_named = [n for n in names if n == "gopro_004"]
    if test_named and not allow_test_metadata:
        raise SystemExit(
            "gopro_004 is in the test partition. This check reads container "
            "metadata only, but it still inspects a held-out video, so it "
            "requires --inspect-test-metadata. The access is appended to "
            "reports/test_access_log.txt, which is committed.")

    root = data_root(cfg)
    video_dir = root / require(cfg, "data.video_dir")
    exts = {e.lower() for e in require(cfg, "data.video_extensions")}

    rows = []
    for name in names:
        matches = [p for p in video_dir.rglob("*")
                   if p.stem == name and p.suffix.lower() in exts]
        if not matches:
            rows.append({"video": name, "error": "not found"})
            continue
        info = ffprobe_video(matches[0], with_tags=True)
        info.update(cv2_video(matches[0]))
        info["video"] = name
        rows.append(info)

    # Cadence, only where creation timestamps survived the dataset's packaging.
    stamps = [(r["video"], r.get("creation_time")) for r in rows]
    have_stamps = [s for _, s in stamps if s]
    cadence: list[str] = []
    if len(have_stamps) < 2:
        cadence.append(
            "No usable creation timestamps. The dataset copy did not preserve "
            "them, so this check cannot establish continuity either way. It "
            "can still falsify it, by a mismatch in resolution or frame rate. "
            "Report that and stop rather than inferring from duration alone.")
    else:
        from datetime import datetime

        def parse(ts):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))

        ordered = [r for r in rows if r.get("creation_time")]
        ordered.sort(key=lambda r: parse(r["creation_time"]))
        for prev, nxt in zip(ordered, ordered[1:]):
            gap = (parse(nxt["creation_time"])
                   - parse(prev["creation_time"])).total_seconds()
            dur = prev.get("container_duration_s")
            slack = (round(gap - dur, 2) if dur else None)
            cadence.append(
                f"{prev['video']} -> {nxt['video']}: gap {gap:.2f}s, "
                f"previous clip duration {dur}, slack {slack}")

    fields = sorted({k for r in rows for k in r})
    with open(out_dir / "gopro_continuity.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "gopro_continuity_cadence.txt").write_text(
        "\n".join(cadence), encoding="utf-8")

    if test_named:
        log = out_dir / "test_access_log.txt"
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime_now()}\tcontinuity check\tgopro_004\t"
                     f"container metadata only, no frames decoded, no "
                     f"annotations read\n")

    for r in rows:
        print(f"{r.get('video'):12s} "
              f"{r.get('ffprobe_width')}x{r.get('ffprobe_height')} "
              f"fps {r.get('ffprobe_avg_frame_rate')} "
              f"dur {r.get('container_duration_s')} "
              f"created {r.get('creation_time')}")
    print()
    for line in cadence:
        print("  " + line)
    print("\nIf 004 and 005 look like one continuous recording, stop and "
          "report. Do not proceed to build the split.")


def datetime_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--stage", choices=["layout", "videos", "continuity"],
                    required=True)
    ap.add_argument("--inspect-test-metadata", action="store_true",
                    help="Required by --stage continuity, which names "
                         "gopro_004. Container metadata only. The access is "
                         "logged to reports/test_access_log.txt.")
    ap.add_argument("--videos", nargs="*", default=GOPRO_GROUP,
                    help="Videos for --stage continuity.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"],
                    cfg["seed"]["cudnn_benchmark"])

    out_dir = repo_path(cfg, cfg["reports"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "layout":
        stage_layout(cfg, out_dir)
    elif args.stage == "videos":
        stage_videos(cfg, out_dir)
    else:
        stage_continuity(cfg, out_dir, args.inspect_test_metadata, args.videos)


if __name__ == "__main__":
    main()
