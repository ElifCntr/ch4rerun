"""Stride-2 confirmation pass, paired analysis. T = 8.

Every stride-2 strip shares a tubelet and a padding condition with a stride-1
strip already scored in item E. Comparing the pair directly removes tubelet
variation, which is the dominant source of noise at n=15, and is a stronger
test than group against group.

WHY STRIDE IS A QUESTION AT ALL. At T = 8 the network sees eight frames
whatever the stride, so stride 2 buys temporal extent at IDENTICAL compute
cost: roughly 0.47 s of footage against 0.23 s. What it pays is retention,
which is what this pass measures.

ONE ARTEFACT OF MY OWN WINDOW CONVENTION, which must be stated in the method
section. For even T the lead is (T-1)//2 = 3, so a T=8 window runs from
centre-3*stride to centre+4*stride. That is ASYMMETRIC, one more frame after
the centre than before. At stride 2 the window is centre-6 to centre+8. So a
loss at the trailing end is partly built in, and "losses fall at the tail"
must not be read as a property of drone motion.

SPAN, which is the variable that appears to matter. A T=8 stride-2 window
spans 15 frames; a T=16 stride-1 window spans 16. If retention is governed by
elapsed time rather than frame count, those two should behave alike, and the
item E scores let that be checked rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path, require  # noqa: E402

SCORES = ["usable", "degrades", "not_usable", "cannot_tell"]
RANK = {"usable": 3, "degrades": 2, "not_usable": 1, "cannot_tell": 0}


def read_csv(path: Path) -> list[dict]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def as_int(v):
    v = (v or "").strip()
    try:
        return int(v)
    except ValueError:
        return None


def dist(rows: list[dict], label: str) -> None:
    c = Counter(r["score"] for r in rows)
    line = f"  {label:<22s}"
    for s in SCORES:
        line += f"{c.get(s, 0):>5d} {c.get(s, 0)/len(rows):>6.0%}"
    print(line + f"   n={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    T = require(cfg, "ruled")["T"]
    man_dir = repo_path(cfg, cfg["reports"]["strip_manifest_dir"])
    sheet_dir = repo_path(cfg, cfg["reports"]["scoring_sheet_dir"])
    out = repo_path(cfg, cfg["reports"]["dir"])

    man = {r["strip_id"]: r for r in
           read_csv(man_dir / "item_e_sampling_manifest.csv")}
    twin = {r["stride2_id"]: r["stride1_id"] for r in
            read_csv(man_dir / "item_stride2_twin_map.csv")}
    dup2 = {r["repeat_id"]: r["original_id"] for r in
            read_csv(man_dir / "item_stride2_duplicate_map.csv")}

    s1 = {r["strip_id"]: r for r in
          read_csv(sheet_dir / "item_e_scoring_sheet_COMPLETED.csv")
          if r["score"].strip()}
    s2 = {r["strip_id"]: r for r in
          read_csv(sheet_dir / "item_stride2_scoring_sheet_COMPLETED.csv")
          if r["score"].strip()}

    # ------------------------------------------------------- consistency ---
    agree, dis = 0, []
    for rep, orig in dup2.items():
        a, b = s2.get(rep), s2.get(orig)
        if not a or not b:
            continue
        if a["score"] == b["score"]:
            agree += 1
        else:
            dis.append((rep, a["score"], orig, b["score"]))
    n = agree + len(dis)
    print("INTRA-RATER CONSISTENCY (stride-2 pass)")
    if n:
        print(f"  {agree} of {n} repeats scored the same")
        for rep, x, orig, y in dis:
            print(f"    {rep} {x}  vs  {orig} {y}")
    print("  Three repeats only. Indicative, not an error bar.")

    # ---------------------------------------------------------- grouped ---
    pass2 = [s2[i] for i in s2 if i in twin]
    pass1_T8 = [s1[i] for i in s1
                if i in man and int(man[i]["T"]) == T
                and int(man[i]["stride"]) == 1]
    pass1_T16 = [s1[i] for i in s1
                 if i in man and int(man[i]["T"]) == 16
                 and int(man[i]["stride"]) == 1]

    print(f"\nGROUPED  (span in frames in brackets)")
    print(f"  {'':<22s}" + "".join(f"{s:>12s}" for s in SCORES))
    dist(pass1_T8, f"T={T} stride 1  [{T}]")
    dist(pass2, f"T={T} stride 2  [{(T-1)*2+1}]")
    dist(pass1_T16, "T=16 stride 1  [16]")
    print("  If retention is governed by ELAPSED TIME rather than frame "
          "count, the middle and bottom rows should resemble each other more "
          "than either resembles the top.")

    # ----------------------------------------------------------- paired ---
    print("\nPAIRED  (each stride-2 strip against its own stride-1 twin)")
    better = worse = same = 0
    rows = []
    for sid2, sid1 in sorted(twin.items()):
        a, b = s2.get(sid2), s1.get(sid1)
        if not a or not b:
            continue
        d = RANK[a["score"]] - RANK[b["score"]]
        if d > 0:
            better += 1
        elif d < 0:
            worse += 1
        else:
            same += 1
        m = man[sid2]
        rows.append({
            "stride2_id": sid2, "stride1_id": sid1,
            "scene": m["scene"], "condition": m["condition"],
            "video": m["video"], "centre_frame": m["centre_frame"],
            "score_stride1": b["score"], "score_stride2": a["score"],
            "usable_frames_stride1": b["usable_frames"],
            "usable_frames_stride2": a["usable_frames"],
            "direction": "worse" if d < 0 else ("better" if d > 0 else "same"),
        })
    total = better + worse + same
    print(f"  worse at stride 2: {worse}/{total}   "
          f"same: {same}/{total}   better: {better}/{total}")
    if total and worse and not better:
        print("  Every pair that moved, moved the SAME WAY. At n="
              f"{total} pairs with {worse} changes and none in the other "
              "direction, the direction is not in doubt even though the "
              "magnitude is not well estimated.")

    # ------------------------------------------------------------ yield ---
    def yield_of(rows_, T_):
        vals = []
        for r in rows_:
            if r["score"] == "usable":
                vals.append(T_)
            elif r["score"] == "degrades":
                v = as_int(r["usable_frames"])
                if v is not None:
                    vals.append(v)
            elif r["score"] in ("not_usable", "cannot_tell"):
                vals.append(0)
        return (sum(vals) / len(vals)) if vals else None

    print("\nUSABLE-FRAME YIELD  (cannot_tell counted as zero)")
    for label, rows_, T_ in (("T=8 stride 1", pass1_T8, T),
                             ("T=8 stride 2", pass2, T),
                             ("T=16 stride 1", pass1_T16, 16)):
        y = yield_of(rows_, T_)
        print(f"  {label:<16s} {y:.2f} of {T_}" if y is not None
              else f"  {label:<16s} n/a")
    print("  Arms 2, 3 and 4 consume every frame, so a frame with no visible "
          "target enters the aggregate as noise rather than as a neutral "
          "absence. Fewer usable frames at identical compute is a cost with "
          "no offsetting saving.")

    # ------------------------------------------------------ where it fails -
    deg = [r for r in pass2 if r["score"] == "degrades"]
    starts = Counter(as_int(r["note"]) or 1 for r in deg)
    print(f"\nDEGRADES AT STRIDE 2: {len(deg)} strips")
    print(f"  visible run starts at frame 1 in {starts.get(1, 0)} of them")
    print("  CAUTION, artefact of my window convention: for even T the lead "
          f"is (T-1)//2 = {(T-1)//2}, so a T={T} stride-2 window runs "
          f"centre-{(T-1)//2*2} to centre+{(T-1-(T-1)//2)*2}. There is one "
          "more frame after the centre than before, so a trailing-end loss "
          "is partly built in and must not be read as drone behaviour.")

    with open(out / "stride2_paired.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\npaired rows written to {out / 'stride2_paired.csv'}")


if __name__ == "__main__":
    main()
