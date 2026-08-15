"""Stride-2 confirmation pass. Blank scoring sheet only, no new images.

Scheduled 12 August 2026, after item F. With T ruled at 8, only the T=8
stride-2 strips need scoring. They were generated in the same run as item E,
so nothing is decoded here: this reads the item E manifest, selects the T=8
stride-2 rows, and writes a blank sheet plus a duplicate map against the
images that already exist.

WHY STRIDE IS NOW A QUESTION IN ITS OWN RIGHT. At T=8 stride 2 spans roughly
0.47 s against 0.23 s at stride 1, at IDENTICAL compute cost, because the
network still sees eight frames. So stride buys temporal extent for free in
compute terms, and pays for it in retention, which is what this pass measures.

Same four-way scale and usable_frames count as item E, so the two are directly
comparable strip for strip: every stride-2 strip here shares a tubelet and a
padding condition with a stride-1 strip already scored.

The pairing is recorded in the sheet's companion mapping file, NOT in the
sheet, so the rater cannot look up what they gave the stride-1 twin.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path, require  # noqa: E402
from src.seeding import seed_everything  # noqa: E402


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
    rng = random.Random(seed + 2000)      # distinct from E (+0) and F (+1000)

    T = require(cfg, "ruled")["T"]
    man_dir = repo_path(cfg, cfg["reports"]["strip_manifest_dir"])
    sheet_dir = repo_path(cfg, cfg["reports"]["scoring_sheet_dir"])
    strips_dir = repo_path(cfg, cfg["reports"]["strips_dir"])

    man = read_csv(man_dir / "item_e_sampling_manifest.csv")
    pass2 = [r for r in man if int(r["T"]) == T and int(r["stride"]) == 2]
    if not pass2:
        sys.exit(f"no T={T} stride-2 strips in the item E manifest")

    missing = [r["strip_id"] for r in pass2
               if not (strips_dir / f"{r['strip_id']}.png").exists()]
    if missing:
        sys.exit(f"images missing, rerun tools/make_strips.py: {missing}")

    # Pair each stride-2 strip with its stride-1 twin: same tubelet, same
    # padding condition. Recorded outside the sheet.
    twin = {}
    for r in pass2:
        for s in man:
            if (s["video"] == r["video"] and s["track_id"] == r["track_id"]
                    and s["centre_frame"] == r["centre_frame"]
                    and s["condition"] == r["condition"]
                    and int(s["T"]) == T and int(s["stride"]) == 1):
                twin[r["strip_id"]] = s["strip_id"]
                break

    n_dup = max(1, int(round(len(pass2) * cfg["item_e"]["duplicate_fraction"])))
    dups = rng.sample(pass2, n_dup)
    sheet = [{"sheet_row": None, "strip_id": r["strip_id"], "score": "",
              "usable_frames": "", "note": ""} for r in pass2]
    dup_map = []
    for i, r in enumerate(dups):
        rid = f"S9{i:03d}"
        sheet.append({"sheet_row": None, "strip_id": rid, "score": "",
                      "usable_frames": "", "note": ""})
        dup_map.append({"repeat_id": rid, "original_id": r["strip_id"]})
        # Repaint the header band with the new id rather than copying the
        # bytes, so the repeat is not recognisable from the label. The pixel
        # rows below the band are untouched, so no frame is re-decoded and
        # the two images are identical where it matters.
        img = cv2.imread(str(strips_dir / f"{r['strip_id']}.png"))
        if img is None:
            sys.exit(f"could not read {r['strip_id']}.png")
        img[0:26, :] = 32
        cv2.putText(img, rid, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imwrite(str(strips_dir / f"{rid}.png"), img)

    rng.shuffle(sheet)
    for i, row in enumerate(sheet, 1):
        row["sheet_row"] = i

    out = sheet_dir / "item_stride2_scoring_sheet_BLANK.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# Stride-2 confirmation pass, T={T}. Same scale as item E.\n")
        fh.write("# score: usable / degrades / not_usable / cannot_tell\n")
        fh.write("# usable_frames: length of the run in which the drone is "
                 "visible. Required for 'degrades'.\n")
        fh.write("# note: the FIRST frame of that run, 1-based.\n")
        w = csv.DictWriter(fh, fieldnames=["sheet_row", "strip_id", "score",
                                           "usable_frames", "note"])
        w.writeheader()
        w.writerows(sheet)

    with open(man_dir / "item_stride2_duplicate_map.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["repeat_id", "original_id"])
        w.writeheader()
        w.writerows(dup_map)

    with open(man_dir / "item_stride2_twin_map.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["stride2_id", "stride1_id"])
        w.writeheader()
        w.writerows([{"stride2_id": a, "stride1_id": b}
                     for a, b in sorted(twin.items())])

    print(f"T={T} stride-2 strips  {len(pass2)}")
    print(f"paired with stride-1  {len(twin)} of {len(pass2)}")
    print(f"sheet rows            {len(sheet)} ({n_dup} repeated)")
    print(f"\nblank sheet {out}")
    print("Images already exist from tools/make_strips.py; none decoded here.")
    if len(twin) < len(pass2):
        print("WARNING: some stride-2 strips have no stride-1 twin, so the "
              "paired comparison is incomplete for those")


if __name__ == "__main__":
    main()
