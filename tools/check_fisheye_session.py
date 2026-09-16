#!/usr/bin/env python3
"""
Fisheye session provenance: are the test and val fisheye videos one occasion?

Authorised provenance verification, 16 Sep 2026. The split carries gopro_004
(test) and gopro_005 (val) as singleton sessions in the gopro_fisheye scene,
so the split itself asserts they are separate occasions without evidence on
the face of it. This script resolves what the disk can resolve.

Method, stated plainly:

  For each fisheye video, sample N frames at evenly spaced indices across the
  whole file and take the per-pixel, per-channel MEDIAN. With a static rig and
  a small moving subject, the median over frames spread across the file is the
  scene without the subject: the background. Two videos shot on one occasion
  from one rig share a background up to noise, exposure drift and whatever
  moved between them. Two videos from different occasions do not.

  A similarity number on its own is not interpretable, so the script computes
  the same measure for EVERY pair of the nine gopro_fisheye videos, not just
  the pair in question. Seven of the nine carry multi-video session labels
  (gopro_treesite, gopro_path), which gives known same-occasion pairs and
  known different-occasion pairs from the same scene, camera and pipeline.
  Those pairs calibrate the scale that the gopro_004 / gopro_005 number is
  read against.

  Measures are computed on the greyscale medians: SSIM (structural
  similarity), mean absolute difference in 8-bit levels, RMSE, and the Pearson
  correlation of the pixel values.

This script reads VIDEO FRAMES ONLY. It reads no annotation file. gopro_004
is in the test partition, so test access is routed through src.splits with
allow_test=True passed from an explicit top-level flag, and the access is
logged by the caller in reports/test_access_log.txt.

Sampling is deterministic: indices are evenly spaced, not drawn. The numpy
seed is set anyway so that nothing added later can quietly become stochastic.

Usage:
    python tools/check_fisheye_session.py --allow-test
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits import load_split  # noqa: E402

SEED = 0
SCENE = "gopro_fisheye"


def fail(msg: str) -> None:
    print(f"FAIL {msg}", file=sys.stderr)
    sys.exit(1)


def sample_indices(n_frames: int, n_samples: int) -> np.ndarray:
    """Evenly spaced frame indices across the whole file, endpoints included."""
    if n_frames < n_samples:
        return np.arange(n_frames)
    return np.unique(np.linspace(0, n_frames - 1, n_samples).astype(int))


def read_at(cap, idx: int, floor: int = 0):
    """Read frame `idx`, stepping back to the nearest readable frame if needed.

    cv2's CAP_PROP_FRAME_COUNT is taken from the container and can overstate
    the number of frames a seek can actually land on: on both fisheye videos
    here the final index reports readable and is not. Stepping back keeps the
    requested SAMPLE COUNT intact rather than silently returning one fewer.
    Returns (frame, actual_index) or (None, None).
    """
    for j in range(idx, floor - 1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ok, frame = cap.read()
        if ok:
            return frame, j
        if idx - j > 30:
            break
    return None, None


def median_background(path: Path, n_samples: int) -> tuple[np.ndarray, list[int]]:
    """Per-pixel per-channel median over evenly spaced sampled frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"cannot open {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        fail(f"{path} reports {n_frames} frames")
    idx = sample_indices(n_frames, n_samples)

    frames, got = [], []
    for i in idx:
        frame, actual = read_at(cap, int(i))
        if frame is None:
            fail(f"{path}: no readable frame at or before index {int(i)}")
        if actual in got:          # fallback landed on an already-sampled frame
            continue
        frames.append(frame)
        got.append(actual)
    cap.release()

    if len(frames) != len(idx):
        fail(f"{path}: asked for {len(idx)} sampled frames, got {len(frames)}. "
             f"Requested indices {list(map(int, idx))}, decoded {got}.")
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8), got


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Similarity measures between two BGR median images, on greyscale."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if ga.shape != gb.shape:
        fail(f"median shapes differ: {ga.shape} vs {gb.shape}")
    diff = ga - gb
    return {
        "ssim": float(structural_similarity(ga, gb, data_range=255.0)),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "pearson": float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1]),
    }



def skyline(med: np.ndarray) -> np.ndarray:
    """Row index of the horizon for each column of a median background.

    Supplementary to the whole-image measures, which on these fisheye
    landscapes are dominated by global layout: every video in the scene has a
    smooth bright upper region, a dark band near the middle and texture below,
    so they score alike whatever the scene actually contains. The SKYLINE is
    not like that. Its shape is set by where the rig stood and which way it
    pointed, and it is unchanged by cloud, sun angle, exposure or whether the
    field has been ploughed, all of which differ freely within one occasion.

    Taken as the strongest vertical gradient per column, on a lightly blurred
    greyscale median. Returns one row index per column, in pixels.
    """
    g = cv2.GaussianBlur(cv2.cvtColor(med, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    sob = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5))
    return np.argmax(sob, axis=0).astype(np.float64)


def label_strip(img: np.ndarray, text: str, height: int = 46) -> np.ndarray:
    strip = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, text, (14, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([strip, img])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--splits", default="data/splits/dvb_splits_v2.0-static.csv")
    ap.add_argument("--data-root",
                    default="~/Desktop/Drone-vs-Bird/videos-and-labels")
    ap.add_argument("--video-dir", default="videos")
    ap.add_argument("--continuity", default="reports/gopro_continuity.csv",
                    help="committed fps evidence used to correct the session "
                         "labels; see reports/gopro_continuity_findings.md")
    ap.add_argument("--samples", type=int, default=50,
                    help="frames sampled per video for the median background")
    ap.add_argument("--contact", type=int, default=5,
                    help="frames per video in the contact sheet")
    ap.add_argument("--out", default="reports/fisheye_session")
    ap.add_argument("--allow-test", action="store_true",
                    help="explicit top-level test access, must be logged")
    args = ap.parse_args()

    np.random.seed(SEED)

    if not args.allow_test:
        fail("gopro_004 is in the test partition. Pass --allow-test and log it.")

    root = Path(os.path.expanduser(args.data_root)) / args.video_dir
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(args.splits).exists():
        fail(f"missing {args.splits}")

    # --- 1. names from the committed split, via the guarded reader -----------
    parts = load_split(args.splits, ["train", "val", "test"], allow_test=True)
    fisheye = []
    for part, rows in parts.items():
        for r in rows:
            if r["scene"].strip() == SCENE:
                fisheye.append({"video": r["video"].strip(),
                                "session": r["session"].strip(),
                                "split": part,
                                "boxes": r["boxes"],
                                "frames_annotated": r.get("annotated_frames", "")})
    fisheye.sort(key=lambda d: d["video"])
    if not fisheye:
        fail(f"no {SCENE} rows in {args.splits}")

    test_v = [d["video"] for d in fisheye if d["split"] == "test"]
    val_v = [d["video"] for d in fisheye if d["split"] == "val"]
    if len(test_v) != 1 or len(val_v) != 1:
        fail(f"expected one test and one val {SCENE} video, "
             f"got test={test_v} val={val_v}")

    print("== 1. fisheye videos in the committed split ==")
    print(f"   {args.splits}")
    for d in fisheye:
        print(f"   {d['video']:<12} session={d['session']:<16} "
              f"split={d['split']:<6} boxes={d['boxes']:>5} "
              f"annotated_frames={d['frames_annotated']:>5}")
    print(f"   test video ACTUAL: {test_v[0]}   (expected gopro_004)")
    print(f"   val  video ACTUAL: {val_v[0]}   (expected gopro_005)")
    print(f"   expectation matches: "
          f"{test_v[0] == 'gopro_004' and val_v[0] == 'gopro_005'}")

    # --- 2. median backgrounds ----------------------------------------------
    print(f"\n== 2. median backgrounds, {args.samples} evenly spaced frames each ==")
    medians: dict[str, np.ndarray] = {}
    for d in fisheye:
        path = root / f"{d['video']}.mp4"
        if not path.exists():
            fail(f"missing video {path}")
        med, got = median_background(path, args.samples)
        medians[d["video"]] = med
        cv2.imwrite(str(out / f"median_{d['video']}.png"), med)
        print(f"   {d['video']:<12} decoded {len(got):>3} frames "
              f"(idx {got[0]}..{got[-1]}) -> median_{d['video']}.png")

    # --- 3. all-pairs similarity, the calibration ---------------------------
    # Reference groups are NOT the split's session column. The committed
    # finding reports/gopro_continuity_findings.md (12 Aug 2026) shows that
    # session gopro_treesite holds gopro_000 and gopro_001 at 25 fps and
    # gopro_002 and gopro_003 at 29.97 fps, and that four clips at two frame
    # rates cannot be one continuous recording. Taking that session column as
    # a same-occasion label would therefore put known non-same pairs into the
    # same-occasion reference band and destroy the comparison. Reference
    # occasion groups are (session label, frame rate) instead, with the rate
    # read from the committed reports/gopro_continuity.csv rather than
    # hardcoded here.
    if not Path(args.continuity).exists():
        fail(f"missing {args.continuity}; the reference bands depend on it")
    fps: dict[str, float] = {}
    with open(args.continuity, newline="") as fh:
        for r in csv.DictReader(fh):
            fps[r["video"].strip()] = round(float(r["ffprobe_avg_frame_rate"]), 1)
    missing = [d["video"] for d in fisheye if d["video"] not in fps]
    if missing:
        fail(f"{args.continuity} has no frame rate for {missing}")

    group = {d["video"]: f"{d['session']}@{fps[d['video']]:.1f}fps" for d in fisheye}
    counts: dict[str, int] = {}
    for d in fisheye:
        counts[group[d["video"]]] = counts.get(group[d["video"]], 0) + 1
    multi = {g for g, c in counts.items() if c > 1}

    print("\n== reference occasion groups (session label corrected by fps) ==")
    for g in sorted(counts):
        members = sorted(v for v in group if group[v] == g)
        role = "reference" if g in multi else "singleton, the open question"
        print(f"   {g:<28} {','.join(members):<28} {role}")

    sess = {d["video"]: group[d["video"]] for d in fisheye}
    split_of = {d["video"]: d["split"] for d in fisheye}
    names = [d["video"] for d in fisheye]

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = compare(medians[a], medians[b])
            same = sess[a] == sess[b]
            known = sess[a] in multi and sess[b] in multi
            rows.append({"a": a, "b": b, "session_a": sess[a], "session_b": sess[b],
                         "split_a": split_of[a], "split_b": split_of[b],
                         "same_session_label": same, "labelled_pair": known, **m})

    with open(out / "pair_similarity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n== 3. all-pairs background similarity ({len(rows)} pairs) ==")
    print(f"   written to {out/'pair_similarity.csv'}")
    print(f"   {'pair':<26}{'SSIM':>8}{'MAE':>8}{'RMSE':>8}{'r':>8}  labels")
    for r in sorted(rows, key=lambda r: -r["ssim"]):
        tag = ("same-session" if r["same_session_label"]
               else "diff-session") if r["labelled_pair"] else "UNLABELLED"
        print(f"   {r['a']}+{r['b']:<13}{r['ssim']:>8.4f}{r['mae']:>8.2f}"
              f"{r['rmse']:>8.2f}{r['pearson']:>8.4f}  {tag}")

    known_same = [r for r in rows if r["labelled_pair"] and r["same_session_label"]]
    known_diff = [r for r in rows if r["labelled_pair"] and not r["same_session_label"]]
    gap_lo = max(r["ssim"] for r in known_diff)
    gap_hi = min(r["ssim"] for r in known_same)
    mae_lo = max(r["mae"] for r in known_same)
    mae_hi = min(r["mae"] for r in known_diff)
    query = [r for r in rows
             if {r["a"], r["b"]} == {test_v[0], val_v[0]}]
    if len(query) != 1:
        fail(f"expected exactly one {test_v[0]}/{val_v[0]} pair, got {len(query)}")
    q = query[0]

    def band(label, rs):
        if not rs:
            return f"   {label:<34} (none)"
        s = [r["ssim"] for r in rs]
        m = [r["mae"] for r in rs]
        return (f"   {label:<34} n={len(rs):<3} "
                f"SSIM {min(s):.4f}..{max(s):.4f}  MAE {min(m):.2f}..{max(m):.2f}")

    print("\n== reference bands, same scene and camera ==")
    print(band("known SAME occasion (one session)", known_same))
    print(band("known DIFFERENT occasion", known_diff))
    print(f"   separation SSIM: different <= {gap_lo:.4f} < {gap_hi:.4f} <= same")
    print(f"   separation MAE : same <= {mae_lo:.2f} < {mae_hi:.2f} <= different")
    print(f"\n   QUERY {q['a']} ({split_of[q['a']]}) vs {q['b']} ({split_of[q['b']]}):"
          f"  SSIM {q['ssim']:.4f}  MAE {q['mae']:.2f}  "
          f"RMSE {q['rmse']:.2f}  r {q['pearson']:.4f}")

    def place(val, lo, hi, lo_name, hi_name):
        if val < lo:
            return f"within the {lo_name} band"
        if val > hi:
            return f"within the {hi_name} band"
        return f"BETWEEN the bands, in the gap {lo:.4g} to {hi:.4g}"

    print(f"   SSIM {q['ssim']:.4f} lies "
          f"{place(q['ssim'], gap_lo, gap_hi, 'different-occasion', 'same-occasion')}")
    print(f"   MAE  {q['mae']:.2f} lies "
          f"{place(q['mae'], mae_lo, mae_hi, 'same-occasion', 'different-occasion')}")
    print("   The two measures do not agree on placement. No verdict is drawn")
    print("   here; the occasion judgement is the author's.")

    # --- 3b. skyline profiles, supplementary ---------------------------------
    sky = {v: skyline(medians[v]) for v in names}
    srows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = sky[a] - sky[b]
            srows.append({
                "a": a, "b": b,
                "same_session_label": sess[a] == sess[b],
                "labelled_pair": sess[a] in multi and sess[b] in multi,
                "skyline_mad_px": float(np.abs(d).mean()),
                "skyline_median_abs_px": float(np.median(np.abs(d))),
                "skyline_pearson": float(np.corrcoef(sky[a], sky[b])[0, 1]),
            })
    with open(out / "skyline_similarity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)

    print(f"\n== 3b. skyline profile agreement (supplementary) ==")
    print(f"   written to {out/'skyline_similarity.csv'}")
    print(f"   {'pair':<26}{'MAD px':>9}{'median px':>11}{'r':>8}  labels")
    for r in sorted(srows, key=lambda r: r["skyline_mad_px"]):
        tag = ("same-session" if r["same_session_label"]
               else "diff-session") if r["labelled_pair"] else "UNLABELLED"
        print(f"   {r['a']}+{r['b']:<13}{r['skyline_mad_px']:>9.2f}"
              f"{r['skyline_median_abs_px']:>11.2f}{r['skyline_pearson']:>8.4f}  {tag}")

    ks = [r for r in srows if r["labelled_pair"] and r["same_session_label"]]
    kd = [r for r in srows if r["labelled_pair"] and not r["same_session_label"]]
    qs = [r for r in srows if {r["a"], r["b"]} == {test_v[0], val_v[0]}][0]
    # The MEDIAN absolute column difference is the statistic to read. The mean
    # is inflated wherever the per-column argmax latches onto a cloud edge
    # instead of the horizon, which happens in the overcast videos: gopro_000
    # and gopro_001 are one occasion by every other measure yet score a mean
    # of 276.69 px against a median of 18 px. The median is unaffected by
    # those columns; the mean is reported alongside it, not relied on.
    for lab, rs in (("known SAME occasion", ks), ("known DIFFERENT occasion", kd)):
        md = [r["skyline_median_abs_px"] for r in rs]
        mn = [r["skyline_mad_px"] for r in rs]
        print(f"   {lab:<34} n={len(rs):<3} median {min(md):.1f}..{max(md):.1f} px"
              f"   (mean {min(mn):.1f}..{max(mn):.1f} px)")
    print(f"   QUERY {qs['a']} vs {qs['b']}: skyline median "
          f"{qs['skyline_median_abs_px']:.1f} px, mean {qs['skyline_mad_px']:.1f} px, "
          f"r {qs['skyline_pearson']:.4f}")
    s_hi = max(r["skyline_median_abs_px"] for r in ks)
    d_lo = min(r["skyline_median_abs_px"] for r in kd)
    print(f"   separation: same <= {s_hi:.1f} px, different >= {d_lo:.1f} px. "
          f"The query at {qs['skyline_median_abs_px']:.1f} px lies in that gap,")
    print(f"   {qs['skyline_median_abs_px'] - s_hi:.1f} px above the same-occasion "
          f"maximum and {d_lo - qs['skyline_median_abs_px']:.1f} px below the "
          f"different-occasion minimum.")

    over = medians[test_v[0]].copy() // 2 + medians[val_v[0]] // 2
    for v, col in ((test_v[0], (0, 255, 255)), (val_v[0], (255, 0, 255))):
        pts = np.stack([np.arange(len(sky[v])), sky[v]], axis=1).astype(np.int32)
        cv2.polylines(over, [pts], False, col, 3, cv2.LINE_AA)
    cv2.imwrite(str(out / "skyline_overlay.png"), label_strip(
        over, f"skyline: {test_v[0]} yellow, {val_v[0]} magenta, on blended medians"))
    print(f"   skyline overlay -> {out/'skyline_overlay.png'}")

    # --- side-by-side of the two medians in question ------------------------
    a_img = label_strip(medians[test_v[0]], f"{test_v[0]}  (test)  median of {args.samples}")
    b_img = label_strip(medians[val_v[0]], f"{val_v[0]}  (val)  median of {args.samples}")
    gap = np.full((a_img.shape[0], 12, 3), 40, dtype=np.uint8)
    sbs = np.hstack([a_img, gap, b_img])
    d = cv2.absdiff(medians[test_v[0]], medians[val_v[0]])
    dvis = label_strip(cv2.applyColorMap(
        cv2.convertScaleAbs(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), alpha=4.0),
        cv2.COLORMAP_INFERNO), "absolute difference of the two medians (x4 gain)")
    pad = np.full((dvis.shape[0], sbs.shape[1] - dvis.shape[1], 3), 40, dtype=np.uint8)
    panel = np.vstack([sbs, np.hstack([dvis, pad])])
    cv2.imwrite(str(out / "median_side_by_side.png"), panel)
    print(f"\n   side-by-side -> {out/'median_side_by_side.png'}")

    # --- 4. contact sheet ---------------------------------------------------
    sheet_rows = []
    for v in (test_v[0], val_v[0]):
        cap = cv2.VideoCapture(str(root / f"{v}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = sample_indices(n, args.contact)
        tiles = []
        for i in want:
            fr, actual = read_at(cap, int(i))
            if fr is None:
                fail(f"{v}: no readable frame at or before index {int(i)}")
            fr = cv2.resize(fr, (480, 270))
            cv2.putText(fr, f"{v} f{actual}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(fr)
        cap.release()
        if len(tiles) != len(want):
            fail(f"{v}: contact sheet wanted {len(want)} tiles, built {len(tiles)}")
        sheet_rows.append(np.hstack(tiles))
    width = max(r.shape[1] for r in sheet_rows)
    if len({r.shape[1] for r in sheet_rows}) != 1:
        fail(f"contact sheet rows differ in width: "
             f"{[r.shape[1] for r in sheet_rows]}. Rows are not padded blindly; "
             f"a width mismatch means a video yielded a different tile count.")
    sheet = np.vstack([r for r in sheet_rows])
    cv2.imwrite(str(out / "contact_sheet.png"), sheet)
    print(f"   contact sheet -> {out/'contact_sheet.png'} "
          f"({args.contact} frames per video, top={test_v[0]}, bottom={val_v[0]})")


if __name__ == "__main__":
    main()
