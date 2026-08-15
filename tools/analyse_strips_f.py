"""Stage 1a item F analysis. Motion separability, drone against background.

Joins the completed scoring sheet to the sampling manifest and reports whether
the rater could separate drone tubelets from background, and WHAT decided it.

CUE FIELD REPAIR, and it is a defect in my interface, not in the scoring.
The scorer's keyboard handler ignores keystrokes typed into a text input. The
rater's cue keys landed in the note box instead of the cue buttons, so the cue
column came back almost empty with single letters in the notes. The rater
confirmed the mapping on 12 August 2026:

    note "a"                        -> cue appearance
    note "b"                        -> cue both
    blank cue on a `background` call -> cue appearance

The third is the rater's own statement: a background call was decided by
appearance, since "what told you it was a drone" does not apply. That also
exposes a wording flaw in the sheet, which asked the cue question in
drone-shaped terms. Both the repair and the flaw belong in the method section.

Nothing is inferred beyond the mapping above. Any row that does not match one
of those three patterns is reported as unrepaired and excluded from the cue
analysis.

WHAT THE SCENE BREAKDOWN CANNOT DO. Drone tubelets are balanced across the
five scenes, three each, but background tubelets are not: 5 industrial,
4 seaside, 2 fisheye, 2 meadow, 2 lake, because that is where the zero-drone
frames live. So scene is confounded with the drone/background label and a
per-scene accuracy could reflect which label that scene mostly carries. The
breakdown is printed because the pattern is worth seeing, and flagged because
n per scene is 5 or 6.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path  # noqa: E402

CUES = ["appearance", "motion", "both"]


def read_csv(path: Path) -> list[dict]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


NOTE_MAP = {"a": "appearance", "b": "both", "m": "motion"}


def repair_cue(row: dict) -> tuple[str, str]:
    """Return (cue, provenance). See the module docstring for the mapping."""
    cue = (row.get("cue") or "").strip().lower()
    note = (row.get("note") or "").strip().lower()
    if cue in CUES and note in NOTE_MAP and NOTE_MAP[note] != cue:
        # Both fields populated and disagreeing. Do not pick one silently.
        return "", f"CONFLICT: cue '{cue}' vs note '{note}'"
    if cue in CUES:
        return cue, "as scored"
    if row["call"] == "cannot_tell" and not note:
        return "", "n/a, no call was made"
    if note == "a":
        return "appearance", "repaired from note 'a'"
    if note == "b":
        return "both", "repaired from note 'b'"
    if note == "m":
        return "motion", "repaired from note 'm'"
    if not note and row["call"] == "background":
        return "appearance", "background call, rater states appearance"
    return "", "UNREPAIRED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    man_dir = repo_path(cfg, cfg["reports"]["strip_manifest_dir"])
    sheet_dir = repo_path(cfg, cfg["reports"]["scoring_sheet_dir"])
    out = repo_path(cfg, cfg["reports"]["dir"])

    man = {r["strip_id"]: r for r in
           read_csv(man_dir / "item_f_sampling_manifest.csv")}
    dup = {r["repeat_id"]: r["original_id"] for r in
           read_csv(man_dir / "item_f_duplicate_map.csv")}
    sheet = read_csv(sheet_dir / "item_f_scoring_sheet_COMPLETED.csv")

    # ------------------------------------------------------- consistency ---
    scored = {r["strip_id"]: r for r in sheet if r["call"].strip()}
    agree, disagree = 0, []
    for rep, orig in dup.items():
        a, b = scored.get(rep), scored.get(orig)
        if not a or not b:
            continue
        if a["call"] == b["call"]:
            agree += 1
        else:
            disagree.append((rep, a["call"], orig, b["call"],
                             man.get(orig, {}).get("kind", "?")))
    n_pairs = agree + len(disagree)
    print("INTRA-RATER CONSISTENCY")
    if n_pairs:
        print(f"  {agree} of {n_pairs} repeated strips called the same")
        for rep, ca, orig, cb, kind in disagree:
            print(f"    {rep} {ca}  vs  {orig} {cb}   (truth: {kind})")
    print("  Only three repeats here against ten in item E, so this is "
          "INDICATIVE, not an error bar.")

    # Repeats excluded from the distributions so no strip counts twice.
    rows = []
    unrepaired = []
    for sid, s in scored.items():
        if sid in dup:
            continue
        m = man.get(sid)
        if not m:
            print(f"WARNING: {sid} scored but absent from the manifest")
            continue
        cue, prov = repair_cue(s)
        if prov == "UNREPAIRED" or prov.startswith("CONFLICT"):
            unrepaired.append(f"{sid} ({prov})")
        rows.append({**s, **m, "cue_repaired": cue, "cue_provenance": prov,
                     "correct": (s["call"] == m["kind"])})

    print(f"\n{len(rows)} distinct strips  "
          f"({len(dup)} repeats held out)")
    n_na = sum(1 for r in rows if r["cue_provenance"].startswith("n/a"))
    if n_na:
        print(f"  {n_na} cannot_tell rows carry no cue, which is correct "
              f"rather than missing")
    for u in unrepaired:
        print(f"  EXCLUDED from the cue analysis: {u}")

    # ---------------------------------------------------------- accuracy ---
    print("\nCONFUSION  (rows = truth, columns = call)")
    calls = ["drone", "background", "cannot_tell"]
    print(f"{'':14s}" + "".join(f"{c:>14s}" for c in calls))
    for truth in ("drone", "background"):
        sub = [r for r in rows if r["kind"] == truth]
        line = f"{truth:<14s}"
        for c in calls:
            line += f"{sum(1 for r in sub if r['call'] == c):>14d}"
        print(line + f"    n={len(sub)}")

    decided = [r for r in rows if r["call"] in ("drone", "background")]
    corr = sum(1 for r in decided if r["correct"])
    print(f"\nDecided calls: {len(decided)} of {len(rows)}  "
          f"({len(rows) - len(decided)} cannot_tell)")
    if decided:
        print(f"Correct when decided: {corr}/{len(decided)} "
              f"({corr/len(decided):.0%})")
    for truth in ("drone", "background"):
        sub = [r for r in decided if r["kind"] == truth]
        if sub:
            c = sum(1 for r in sub if r["correct"])
            print(f"  {truth:<12s} {c}/{len(sub)} ({c/len(sub):.0%})")

    # --------------------------------------------------------------- cue ---
    print("\nCUE  (what decided the call)")
    cue_rows = [r for r in rows if r["cue_repaired"]]
    cnt = Counter(r["cue_repaired"] for r in cue_rows)
    for c in CUES:
        n = cnt.get(c, 0)
        print(f"  {c:<12s} {n:>3d}  {n/len(cue_rows) if cue_rows else 0:>5.0%}")
    print("\n  accuracy by cue, decided calls only:")
    for c in CUES:
        sub = [r for r in cue_rows
               if r["cue_repaired"] == c and r["call"] in ("drone", "background")]
        if not sub:
            print(f"    {c:<12s} n=0")
            continue
        k = sum(1 for r in sub if r["correct"])
        print(f"    {c:<12s} {k}/{len(sub)} ({k/len(sub):.0%})")
    if cnt.get("motion", 0) == 0:
        print("\n  MOTION ALONE NEVER DECIDED A CALL. On this sample a human "
              "separating these crops did not need the temporal dimension. "
              "That is a finding about the DATA and about human judgement, "
              "not about what a network can exploit, and the chapter must "
              "state it beside Claim A rather than around it.")

    # ------------------------------------------------------------- scene ---
    print("\nBY SCENE  (confounded: background tubelets are not balanced "
          "across scenes)")
    print(f"{'scene':<16s}{'n':>4s}{'drone':>7s}{'bg':>5s}{'correct':>10s}")
    for sc in sorted({r["scene"] for r in rows}):
        sub = [r for r in rows if r["scene"] == sc]
        d = sum(1 for r in sub if r["kind"] == "drone")
        dec = [r for r in sub if r["call"] in ("drone", "background")]
        k = sum(1 for r in dec if r["correct"])
        acc = f"{k}/{len(dec)}" if dec else "-"
        print(f"{sc:<16s}{len(sub):>4d}{d:>7d}{len(sub)-d:>5d}{acc:>10s}")

    # ------------------------------------------------------------ output ---
    with open(out / "item_f_scored.csv", "w", newline="",
              encoding="utf-8") as fh:
        fields = ["strip_id", "kind", "call", "correct", "cue_repaired",
                  "cue_provenance", "note", "scene", "video", "resolution",
                  "fps", "centre_frame", "equiv_side", "T", "padding_fraction"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["strip_id"]))
    print(f"\njoined rows written to {out / 'item_f_scored.csv'}")


if __name__ == "__main__":
    main()
