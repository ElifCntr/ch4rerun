"""Stage 1a item E analysis. Joins the completed scoring sheet to the sampling
manifest and reports the result.

INPUTS, all committed:
  data/strip_manifests/item_e_sampling_manifest.csv   what each strip is
  data/strip_manifests/item_e_duplicate_map.csv       repeat -> original
  reports/scoring_sheets/item_e_scoring_sheet_COMPLETED.csv

SHEET SEMANTICS, confirmed by the rater 12 Aug 2026:
  score           usable / degrades / not_usable / cannot_tell
  usable_frames   length of the contiguous run in which the drone is visible
  note            the FIRST frame of that run, 1-based within the strip

So note plus usable_frames locate the visible run inside the window. A run
starting at 1 means loss at the trailing end only; a run starting later means
loss at both ends, which is what drift produces when the crop is defined from
the centre frame.

ANALYSIS IS BY SCENE, per the 12 Aug ruling. With 15 tubelets over five
scenes, scene and rate group cannot both be separated; scene is the stronger
stratifier and correlates with rate anyway, so rate is RECORDED per strip and
reported, never used as a stratum.

This script sets no parameter and recommends no T. It reports what was scored.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path  # noqa: E402

SCORES = ["usable", "degrades", "not_usable", "cannot_tell"]


def read_csv(path: Path) -> list[dict]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def as_int(v: str):
    v = (v or "").strip()
    if not v:
        return None
    try:
        return int(v)          # tolerates a leading zero, e.g. "08"
    except ValueError:
        return None


def table(title: str, groups: dict[str, list[dict]]) -> None:
    print(f"\n{title}")
    w = max(len(k) for k in groups) + 2
    head = " " * w + "".join(f"{s:>13s}" for s in SCORES) + f"{'n':>7s}"
    print(head)
    for k in sorted(groups):
        rows = groups[k]
        c = Counter(r["score"] for r in rows)
        line = f"{k:<{w}s}"
        for s in SCORES:
            line += f"{c.get(s, 0):>6d} {c.get(s, 0)/len(rows):>5.0%}"
        line += f"{len(rows):>7d}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    man_dir = repo_path(cfg, cfg["reports"]["strip_manifest_dir"])
    sheet_dir = repo_path(cfg, cfg["reports"]["scoring_sheet_dir"])
    out = repo_path(cfg, cfg["reports"]["dir"])

    man = {r["strip_id"]: r for r in
           read_csv(man_dir / "item_e_sampling_manifest.csv")}
    dup = {r["repeat_id"]: r["original_id"] for r in
           read_csv(man_dir / "item_e_duplicate_map.csv")}
    sheet = read_csv(sheet_dir / "item_e_scoring_sheet_COMPLETED.csv")

    scored = {r["strip_id"]: r for r in sheet if r["score"].strip()}
    missing = [r["strip_id"] for r in sheet if not r["score"].strip()]
    if missing:
        print(f"UNSCORED, excluded: {missing}")

    # ------------------------------------------------ consistency first ---
    agree, disagree = 0, []
    for rep, orig in dup.items():
        a, b = scored.get(rep), scored.get(orig)
        if not a or not b:
            continue
        if a["score"] == b["score"]:
            agree += 1
        else:
            disagree.append((rep, a["score"], orig, b["score"]))
    n_pairs = agree + len(disagree)
    print("INTRA-RATER CONSISTENCY")
    print(f"  {agree} of {n_pairs} repeated strips scored the same "
          f"({agree/n_pairs:.0%})" if n_pairs else "  no usable pairs")
    for rep, sa, orig, sb in disagree:
        t = man.get(orig, {})
        print(f"    {rep} {sa}  vs  {orig} {sb}   "
              f"(T={t.get('T')}, cond {t.get('condition')}, "
              f"{t.get('scene')})")
    print("  Single rater, as with the static/dynamic labels. This figure is "
          "the sheet's own error bar and belongs in the report beside every "
          "percentage below.")

    # Repeats are excluded from the distributions so no strip counts twice.
    rows = []
    for sid, s in scored.items():
        if sid in dup:
            continue
        m = man.get(sid)
        if not m:
            print(f"WARNING: {sid} scored but absent from the manifest")
            continue
        rows.append({**s, **m})

    print(f"\n{len(rows)} distinct strips analysed "
          f"({len(dup)} repeats held out of the distributions)")

    table("BY CLIP LENGTH T",
          {f"T={T}": [r for r in rows if int(r["T"]) == T]
           for T in sorted({int(r["T"]) for r in rows})})
    table("BY PADDING CONDITION",
          {k: [r for r in rows if r["condition"] == k]
           for k in {r["condition"] for r in rows}})
    table("BY SCENE", {k: [r for r in rows if r["scene"] == k]
                       for k in {r["scene"] for r in rows}})

    # T x condition, the cell the ruling actually turns on.
    print("\nBY T AND CONDITION  (fraction scored 'usable')")
    Ts = sorted({int(r["T"]) for r in rows})
    conds = sorted({r["condition"] for r in rows})
    print("  At T=3 the two conditions coincide (both p=0.3), so that row is "
          "one set of strips shown under both.")
    print("        " + "".join(f"{c:>16s}" for c in ("A (p=0.3)", "B (per T)")))
    for T in Ts:
        line = f"  T={T:<4d}"
        for want in ("A", "B"):
            cell = [r for r in rows if int(r["T"]) == T
                    and r["condition"] in (want, "AB")]
            if not cell:
                line += f"{'-':>16s}"
            else:
                u = sum(1 for r in cell if r["score"] == "usable")
                line += f"{u:>4d}/{len(cell):<3d}{u/len(cell):>8.0%}"
        print(line)

    # ------------------------------------------------- visible-run shape ---
    print("\nDEGRADES: where the visible run sits inside the window")
    deg = [r for r in rows if r["score"] == "degrades"]
    no_count = [r["strip_id"] for r in deg if as_int(r["usable_frames"]) is None]
    print(f"  {len(deg)} degrades strips, {len(no_count)} without a count "
          f"and excluded: {no_count or 'none'}")
    lead_only, both_ends = 0, 0
    for r in deg:
        n = as_int(r["usable_frames"])
        start = as_int(r["note"])
        if n is None:
            continue
        T = int(r["T"])
        if start in (None, 1):
            lead_only += 1
        else:
            both_ends += 1
        print(f"    {r['strip_id']}  T={T:<3d} cond {r['condition']:<3s} "
              f"{r['scene']:<14s} visible {start or 1}..{(start or 1)+n-1} "
              f"of {T}  ({n}/{T} = {n/T:.0%})")
    print(f"  run starts at frame 1 (loss at the trailing end only): {lead_only}")
    print(f"  run starts later (loss at BOTH ends, the drift signature): "
          f"{both_ends}")

    # ------------------------------------------------------------ output ---
    with open(out / "item_e_scored.csv", "w", newline="",
              encoding="utf-8") as fh:
        fields = ["strip_id", "score", "usable_frames", "note", "T", "stride",
                  "condition", "padding_fraction", "scene", "resolution",
                  "fps", "video", "track_id", "centre_frame",
                  "centre_equiv_side"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["strip_id"]))
    print(f"\njoined rows written to {out / 'item_e_scored.csv'}")
    print("No T is recommended here. This reports what was scored.")


if __name__ == "__main__":
    main()
