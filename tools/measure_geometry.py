"""Stage 1a items G1 and H. Reads the CSVs written by measure_instances.py.

G1, kernel versus target size. For each structuring element shape and size in
the reporting grid, how many annotated instances would a morphological opening
erase? Computed by actually eroding a binary rectangle with the real OpenCV
structuring element rather than from an analytic rule, so ellipse and cross
are exact rather than approximated.

H, padding sweep. For each padding fraction and window length, how often does
the target stay inside a crop defined from the centre frame of the window?

NEITHER SCRIPT CHOOSES A VALUE. The kernel sizes, shapes and padding
fractions are reporting grids in the config. G1 answers "what would this
kernel do", not "use this kernel". H answers "what would this padding retain",
not "use this padding".

TWO DEFINITIONS THAT ARE MINE AND MUST BE STATED IN THE METHOD SECTION.

1. G1 treats the foreground blob as a SOLID RECTANGLE the size of the
   annotated box. A real BGS mask is smaller and holier than its bounding box,
   never larger, so every erasure count here is a LOWER BOUND on how many
   targets a kernel actually destroys. The true figure needs proposals and so
   belongs to G2 in Stage 1b-M. Do not present G1 as the answer; present it as
   the floor.

2. H reads "padding fraction p" as a square crop of side
   (1 + 2p) * max(w, h) centred on the centre-frame box centre, following the
   ruled square-pad-in-original-frame geometry. The phrase could instead mean
   p of each dimension separately, which gives a different crop for a
   non-square box. The reading is stated here so it can be overruled rather
   than discovered.

Containment is reported two ways, box fully inside the crop and box centre
inside the crop, because neither is obviously the right definition of "the
target stayed in view" and no threshold is being set.

RESULT THAT LIMITS G1, established before running it on the data. Under the
solid-rectangle assumption, KERNEL SHAPE MAKES NO DIFFERENCE AT ALL. Erosion
of an axis-aligned rectangle A by a structuring element S admits x exactly
when the BOUNDING BOX of S fits, because A is an axis-aligned rectangle and so
contains every point of S+x precisely when it contains bbox(S)+x. Rect,
ellipse and cross all have a k x k bounding box, so all three erode a
rectangle identically. Verified empirically over 3,125 (w, h, k) combinations
with zero disagreements.

So G1 can report on kernel SIZE and nothing else. Kernel shape is a question
about real mask blobs, which are neither rectangular nor solid, and it belongs
to G2 in Stage 1b-M. The three shapes are still computed here so the identity
is visible in the committed CSV rather than asserted.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path, require  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

SHAPES = {"rect": cv2.MORPH_RECT,
          "ellipse": cv2.MORPH_ELLIPSE,
          "cross": cv2.MORPH_CROSS}


# ---------------------------------------------------------------------- G1 --

_survive_cache: dict[tuple, bool] = {}


def survives_opening(w: int, h: int, shape: str, k: int) -> bool:
    """Does a solid w x h rectangle survive an opening with a k x k element?

    Erosion is what removes the blob; the dilation that follows cannot bring
    back what erosion deleted. So this reduces to whether erosion leaves any
    pixel at all.
    """
    key = (w, h, shape, k)
    if key in _survive_cache:
        return _survive_cache[key]
    if w <= 0 or h <= 0:
        _survive_cache[key] = False
        return False
    pad = k + 2
    img = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    img[pad:pad + h, pad:pad + w] = 255
    se = cv2.getStructuringElement(SHAPES[shape], (k, k))
    out = bool(cv2.erode(img, se).any())
    _survive_cache[key] = out
    return out


def odd(k: float) -> int:
    """Nearest odd integer, ties resolved downward.

    OpenCV structuring elements are built on odd sides, so a normalised grid
    cannot land exactly on the ratio it asks for. At small sizes that matters:
    scaling k=3 by 2 wants 6, and the nearest odd values are 5 and 7, an error
    of about 17 per cent either way. The exact scaled value is recorded
    alongside the applied one so the discretisation is visible rather than
    hidden, and rounding is to NEAREST rather than upward so the error does
    not bias systematically toward more erasure at the larger resolution.
    """
    lo = math.floor((k - 1) / 2.0) * 2 + 1
    hi = lo + 2
    chosen = lo if (k - lo) <= (hi - k) else hi
    return max(1, int(chosen))


def scale_for(resolution: str, reference: str) -> float:
    """Linear scale factor from frame AREA, per the 11 Aug ruling that spatial
    thresholds normalise to frame area. Area ratio r gives linear ratio
    sqrt(r), so 4K against 1080p is exactly 2."""
    rw, rh = (int(v) for v in reference.split("x"))
    w, h = (int(v) for v in resolution.split("x"))
    return math.sqrt((w * h) / float(rw * rh))


def item_g1_normalised(boxes: list[dict], shapes: list[str],
                       sizes_ref: list[int], reference: str) -> list[dict]:
    """PRIMARY G1. Kernel size given in reference pixels at the reference
    resolution and scaled per video by the square root of the frame-area
    ratio, which is what the pipeline will apply. The raw-pixel sweep is kept
    separately as a labelled diagnostic and is NOT the pipeline's grid."""
    rows = []
    resolutions = sorted({b["resolution"] for b in boxes})
    actual = {r: {k: odd(k * scale_for(r, reference)) for k in sizes_ref}
              for r in resolutions}

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for b in boxes:
        groups[("all", "all")].append(b)
        groups[("resolution", b["resolution"])].append(b)
        groups[("scene", b["scene"])].append(b)
        groups[("split", b["split"])].append(b)

    for shape in shapes:
        for k_ref in sizes_ref:
            for (gtype, gname), members in sorted(groups.items()):
                erased = 0
                for b in members:
                    k = actual[b["resolution"]][k_ref]
                    w = int(round(float(b["w"])))
                    h = int(round(float(b["h"])))
                    if not survives_opening(w, h, shape, k):
                        erased += 1
                rows.append({
                    "kernel_shape": shape,
                    "kernel_size_reference_px": k_ref,
                    "reference_resolution": reference,
                    "applied_sizes": ";".join(
                        f"{r}:{actual[r][k_ref]}" for r in resolutions),
                    "exact_scaled_sizes": ";".join(
                        f"{r}:{k_ref * scale_for(r, reference):.2f}"
                        for r in resolutions),
                    "group_type": gtype, "group": gname,
                    "instances": len(members), "erased": erased,
                    "erased_frac": round(erased / len(members), 6),
                    "note": "frame-area-normalised grid, the one the pipeline "
                            "applies; solid-rectangle assumption so this is a "
                            "LOWER bound on real erasure",
                })
    return rows


def item_g1(boxes: list[dict], shapes: list[str], sizes: list[int]) -> list[dict]:
    """DIAGNOSTIC ONLY. Raw pixel grid, uniform across resolutions. Kept for
    comparison; it is not what the pipeline applies."""
    rows = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for b in boxes:
        groups[("all", "all")].append(b)
        groups[("resolution", b["resolution"])].append(b)
        groups[("scene", b["scene"])].append(b)
        groups[("split", b["split"])].append(b)

    for shape in shapes:
        for k in sizes:
            for (gtype, gname), members in sorted(groups.items()):
                erased = 0
                for b in members:
                    w = int(round(float(b["w"])))
                    h = int(round(float(b["h"])))
                    if not survives_opening(w, h, shape, k):
                        erased += 1
                rows.append({
                    "kernel_shape": shape, "kernel_size": k,
                    "group_type": gtype, "group": gname,
                    "instances": len(members), "erased": erased,
                    "erased_frac": round(erased / len(members), 6),
                    "note": "solid-rectangle assumption, so this is a LOWER "
                            "bound on real erasure",
                })
    return rows


# ----------------------------------------------------------------------- H --

def item_h(boxes: list[dict], windows: list[int],
           fractions: list[float]) -> list[dict]:
    """Padding retention. Groups boxes into tracks, walks every contiguous
    window, and asks whether each frame's box stays inside the crop defined
    from the centre frame."""
    tracks: dict[tuple, list[dict]] = defaultdict(list)
    for b in boxes:
        tracks[(b["video"], b["track_id"])].append(b)
    for rows in tracks.values():
        rows.sort(key=lambda r: int(r["frame"]))

    tally: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])

    for (video, _tid), rows in tracks.items():
        assoc = rows[0]["associated"] in ("True", "true", True)
        n = len(rows)
        for T in windows:
            if n < T:
                continue
            for s in range(n - T + 1):
                win = rows[s:s + T]
                # Contiguous frames only. A gap is not a window.
                if int(win[-1]["frame"]) - int(win[0]["frame"]) != T - 1:
                    continue
                c = win[(T - 1) // 2]
                cw, ch = float(c["w"]), float(c["h"])
                ccx, ccy = float(c["cx"]), float(c["cy"])
                base = max(cw, ch)
                for p in fractions:
                    side = base * (1.0 + 2.0 * p)
                    x0, x1 = ccx - side / 2.0, ccx + side / 2.0
                    y0, y1 = ccy - side / 2.0, ccy + side / 2.0
                    for f in win:
                        fx, fy = float(f["x"]), float(f["y"])
                        fw, fh = float(f["w"]), float(f["h"])
                        full = (fx >= x0 and fy >= y0
                                and fx + fw <= x1 and fy + fh <= y1)
                        cen = (x0 <= float(f["cx"]) <= x1
                               and y0 <= float(f["cy"]) <= y1)
                        key = (T, p, assoc)
                        tally[key][0] += 1
                        tally[key][1] += int(full)
                        tally[key][2] += int(cen)

    rows = []
    for (T, p, assoc), (total, full, cen) in sorted(tally.items()):
        rows.append({
            "T": T, "padding_fraction": p, "associated": assoc,
            "window_frames": total,
            "fully_inside": full,
            "fully_inside_frac": round(full / total, 6) if total else None,
            "centre_inside": cen,
            "centre_inside_frac": round(cen / total, 6) if total else None,
        })
    return rows


# -------------------------------------------------------------------- main --

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"],
                    cfg["seed"]["cudnn_benchmark"])

    out = repo_path(cfg, cfg["reports"]["dir"])
    boxes_path = out / "instances_boxes.csv"
    if not boxes_path.exists():
        sys.exit(f"{boxes_path} not found. Run tools/measure_instances.py first.")
    boxes = list(csv.DictReader(open(boxes_path, encoding="utf-8")))

    shapes = require(cfg, "morphology.kernel_shapes")
    sizes = require(cfg, "morphology.kernel_sizes")
    windows = require(cfg, "measurement.drift_windows")
    fractions = require(cfg, "padding.fractions")

    sizes_ref = require(cfg, "morphology.kernel_sizes_reference")
    reference = require(cfg, "morphology.reference_resolution")

    g1n = item_g1_normalised(boxes, shapes, sizes_ref, reference)
    g1 = item_g1(boxes, shapes, sizes)
    h = item_h(boxes, windows, fractions)
    write_csv(out / "item_g1_kernel_geometry_normalised.csv", g1n)
    write_csv(out / "item_g1_kernel_geometry_rawpx_diagnostic.csv", g1)
    write_csv(out / "item_h_padding_retention.csv", h)

    print(f"instances               {len(boxes)}")
    print(f"G1 rows                 {len(g1)}  "
          f"({len(shapes)} shapes x {len(sizes)} sizes x groups)")
    print(f"H rows                  {len(h)}  "
          f"({len(windows)} windows x {len(fractions)} fractions x assoc)")
    scales = {r: scale_for(r, reference)
              for r in sorted({b["resolution"] for b in boxes})}
    print(f"\nG1 PRIMARY, frame-area normalised, reference {reference}")
    print("  applied kernel size per resolution: " + ", ".join(
        f"{r} x{scales[r]:.2f}" for r in scales))
    for shape in ["rect"]:
        for gt, gn in [("all", "all")] + [("scene", s2) for s2 in sorted(
                {b["scene"] for b in boxes})]:
            line = f"  {gn:14s}"
            for k in sizes_ref:
                r = next(x for x in g1n if x["kernel_shape"] == shape
                         and x["kernel_size_reference_px"] == k
                         and x["group_type"] == gt and x["group"] == gn)
                line += f"  k{k}: {r['erased_frac']:.3f}"
            print(line)
    print("\nG1 DIAGNOSTIC, raw pixels, NOT the pipeline grid:")
    for shape in shapes:
        line = "  " + f"{shape:8s}"
        for k in sizes:
            r = next(x for x in g1 if x["kernel_shape"] == shape
                     and x["kernel_size"] == k and x["group_type"] == "all")
            line += f"  k={k}: {r['erased_frac']:.3f}"
        print(line)
    print("\nH, pooled, fraction of window frames with the box fully inside:")
    for T in windows:
        line = "  " + f"T={T:<3d}"
        for p in fractions:
            r = [x for x in h if x["T"] == T and x["padding_fraction"] == p]
            tot = sum(x["window_frames"] for x in r)
            fl = sum(x["fully_inside"] for x in r)
            line += f"  p={p}: {fl/tot:.3f}" if tot else f"  p={p}: n/a"
        print(line)
    print(f"\nwritten to {out}")
    print("G1 assumes a solid rectangle the size of the annotated box, so its "
          "erasure counts are a LOWER bound. It measures only the COST side "
          "of the kernel trade; the benefit side is noise suppression, which "
          "needs proposals. G2 in Stage 1b-M resolves the trade empirically.")


if __name__ == "__main__":
    main()
