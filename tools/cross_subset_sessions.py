#!/usr/bin/env python3
"""
Cross-subset session table for the evaluated partitions.

Lists every val and test video with its scene, session label, frame rate and
any date carried in its filename, then FLAGS, without verdict, each scene in
which a val video and a test video could plausibly share a recording occasion.

The flags are the three the author asked for:

  DATE     a filename date appears on both sides of the val/test boundary
           within one scene.
  ADJACENT camera file numbering runs across the boundary within one scene,
           that is the smallest gap between a test number and a val number is
           small enough that consecutive recordings are possible.
  AUDIT    the session assignment rests on the 2026-08-04 filename and
           thumbnail audit alone, with no corroborating metadata: no shared
           filename date and no shared camera file number tying the session's
           members together, and no creation_time anywhere in the inventory.

A flag is a statement that the disk cannot separate two sessions, not a claim
that they are one. Occasion judgements are the author's.

Provenance of the session column: data/provenance/dvb_splits_v1_README.md
records "Sessions move whole; scenes documented from the 2026-08-04 filename +
thumbnail audit". tools/build_split.py carries the session column through from
that manifest and asserts zero session overlap between partitions, so the
integrity guarantee is only ever as strong as the label the audit produced.

Usage:
    python tools/cross_subset_sessions.py
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import date as _date
from collections import defaultdict
from pathlib import Path
from textwrap import wrap

SPLIT = "data/splits/dvb_splits_v2.0-static.csv"
INVENTORY = "reports/video_inventory.csv"
JUDGEMENTS = "data/occasion_judgements.csv"
OUT_CSV = Path("reports/cross_subset_sessions.csv")

DATE_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_")
# GoPro chaptered names: GOPR<file> is chapter 1, GP<nn><file> later chapters.
NUM_RE = re.compile(r"(?:^|_)(GOPR|GP\d{2}|C|gopro_)(\d+)")
ADJACENT_GAP = 1


SPLIT_MARK = {"train": "train", "val": "VAL", "test": "TEST"}


def read_judgements(path: str) -> dict:
    """scene -> judgement row, from the committed register.

    The register records the AUTHOR'S disposition of a flag. It never changes
    what the metadata shows: a flag stays flagged in the output below, and the
    judgement is printed beside it, so the measured position and the judged
    position can always be read apart.
    """
    if not Path(path).exists():
        return {}
    rows = [ln for ln in open(path, encoding="utf-8")
            if ln.strip() and not ln.lstrip().startswith("#")]
    return {r["scene"]: r for r in csv.DictReader(rows)}


def days(d1: str, d2: str) -> int:
    """Exact whole days between two ISO filename dates."""
    a = _date(*(int(x) for x in d1.split("-")))
    b = _date(*(int(x) for x in d2.split("-")))
    return abs((a - b).days)


def parse_name(v: str):
    m = DATE_RE.match(v)
    date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
    n = NUM_RE.search(v)
    if not n:
        return date, "", None, ""
    # GOPR1234 and GP011234 are chapters of one camera file, so both map to the
    # GOPR series on the same number.
    series = "GOPR" if n.group(1).startswith(("GOPR", "GP")) else n.group(1)
    return date, series.rstrip("_"), int(n.group(2)), n.group(0).lstrip("_")


def main() -> None:
    for p in (SPLIT, INVENTORY):
        if not Path(p).exists():
            print(f"FAIL missing {p}", file=sys.stderr)
            sys.exit(1)

    inv = {r["video"]: r for r in csv.DictReader(open(INVENTORY, newline=""))}
    judged = read_judgements(JUDGEMENTS)
    rows = []
    for r in csv.DictReader(open(SPLIT, newline="")):
        if r["split"] not in ("val", "test"):
            continue
        v = r["video"]
        if v not in inv:
            print(f"FAIL {v} is in the split and not in {INVENTORY}",
                  file=sys.stderr)
            sys.exit(1)
        i = inv[v]
        date, series, num, token = parse_name(v)
        rows.append({
            "video": v, "scene": r["scene"], "session": r["session"],
            "split": r["split"],
            "fps": round(float(i["ffprobe_avg_frame_rate"]), 3),
            "resolution": f'{i["ffprobe_width"]}x{i["ffprobe_height"]}',
            "duration_s": round(float(i["duration_s"]), 3),
            "filename_date": date or "-",
            "camera_series": series or "-",
            "camera_number": num if num is not None else "",
            "camera_token": token,
            "creation_time": i["creation_time"] or "-",
        })
    rows.sort(key=lambda r: (r["scene"], r["split"], r["video"]))

    # every video, for the numbering-run context and the adjacency calibration
    allrows = []
    for r in csv.DictReader(open(SPLIT, newline="")):
        d, se, n, tk = parse_name(r["video"])
        allrows.append({**r, "date": d, "series": se, "num": n, "token": tk})

    print("== val and test videos, from the committed split ==")
    print(f"   {SPLIT}\n")
    hdr = (f"{'video':<34}{'scene':<15}{'session':<17}{'split':<6}"
           f"{'fps':>8}{'dur_s':>9}  {'resolution':<11}{'date':<12}{'cam':<12}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for r in rows:
        cam = r["camera_token"] or "-"
        print(f"{r['video']:<34}{r['scene']:<15}{r['session']:<17}"
              f"{r['split']:<6}{r['fps']:>8.3f}{r['duration_s']:>9.3f}  "
              f"{r['resolution']:<11}{r['filename_date']:<12}{cam:<12}")
    ct = {r["creation_time"] for r in rows}
    print(f"\n   creation_time across all {len(rows)} videos: {sorted(ct)}  "
          f"(none survives the dataset re-encode)")

    # ---- session corroboration -------------------------------------------
    by_sess = defaultdict(list)
    for r in allrows:
        by_sess[r["session"]].append(r)

    print("\n== what each evaluated session's assignment rests on ==")
    print(f"   {'session':<17}{'scene':<15}{'split':<6}{'n':>3}  "
          f"corroborating metadata")
    audit_only = set()
    for s in sorted({r["session"] for r in rows}):
        mem = by_sess[s]
        scene = mem[0]["scene"]
        splits = sorted({m["split"] for m in mem})
        dates = {m["date"] for m in mem if m["date"]}
        nums = {m["num"] for m in mem if m["num"] is not None}
        ev = []
        if len(mem) > 1 and len(dates) == 1:
            ev.append(f"shared filename date {dates.pop()}")
        if len(mem) > 1 and len(nums) == 1:
            ev.append(f"shared camera file number {nums.pop()}")
        if not ev and len(mem) == 1 and not dates:
            ev.append("NONE, audit only")
            audit_only.add(s)
        elif not ev and len(mem) == 1:
            ev.append(f"filename date {sorted(dates)[0]} only")
        print(f"   {s:<17}{scene:<15}{'/'.join(splits):<6}{len(mem):>3}  "
              f"{'; '.join(ev)}")

    # ---- flags per scene ---------------------------------------------------
    print("\n== flags, by scene, val against test ==")
    flagged = 0
    closed: set[str] = set()
    open_scenes: set[str] = set()
    no_verdict: set[str] = set()
    for scene in sorted({r["scene"] for r in rows}):
        tv = [r for r in rows if r["scene"] == scene and r["split"] == "test"]
        vv = [r for r in rows if r["scene"] == scene and r["split"] == "val"]
        flags = []

        shared = ({r["filename_date"] for r in tv if r["filename_date"] != "-"}
                  & {r["filename_date"] for r in vv if r["filename_date"] != "-"})
        if shared:
            flags.append(f"DATE: filename date {sorted(shared)} on both sides")

        # A pair whose filename dates both exist and differ is already
        # separated by the date, which is stronger than any numbering
        # argument; adjacency is not reported for those. This is why
        # industrial is not flagged on C0001 appearing in both partitions.
        pairs = [(abs(a["camera_number"] - b["camera_number"]), a, b)
                 for a in tv for b in vv
                 if a["camera_number"] != "" and b["camera_number"] != ""
                 and a["camera_series"] == b["camera_series"]
                 and not (a["filename_date"] != "-" and b["filename_date"] != "-"
                          and a["filename_date"] != b["filename_date"])]
        if pairs:
            g, a, b = min(pairs, key=lambda p: p[0])
            if g <= ADJACENT_GAP:
                flags.append(
                    f"ADJACENT: {a['camera_token']} (test) and "
                    f"{b['camera_token']} (val) differ by {g} in "
                    f"{a['camera_series']} numbering")
        sep = [(a, b) for a in tv for b in vv
               if a["filename_date"] != "-" and b["filename_date"] != "-"
               and a["filename_date"] != b["filename_date"]]

        aud = sorted({r["session"] for r in tv + vv} & audit_only)
        if aud:
            flags.append(f"AUDIT: session(s) {aud} rest on the audit alone")

        print(f"\n   {scene}:")
        if flags:
            flagged += 1
            for f in flags:
                print(f"     FLAG  {f}")
        else:
            print("     no flag on any of the three criteria")
        j = judged.get(scene)
        if j:
            closed.add(scene)
            if j["verdict"] == "none_stated":
                no_verdict.add(scene)
                print(f"     WORKED  {j['session_a']} vs {j['session_b']}, "
                      f"basis {j['basis']}, {j['rater']}, {j['date']}. "
                      f"NO VERDICT RECORDED.")
            else:
                print(f"     JUDGED  {j['verdict']} ({j['session_a']} vs "
                      f"{j['session_b']}), basis {j['basis']}, {j['rater']}, "
                      f"{j['date']}")
            for line in wrap(j["note"], 68):
                print(f"             {line}")
            if j.get("corroboration"):
                print(f"             Corroborated by {j['corroboration']}. The "
                      f"verdict rests on the")
                print(f"             inspection and precedes that measurement; "
                      f"it is not derived from it.")
            print(f"             Judgement label, not a measurement. "
                  f"Source {JUDGEMENTS}.")
        elif flags:
            open_scenes.add(scene)
        # numbering run in this scene, all splits, for context
        if not flags and sep:
            print(f"     the filename dates separate them: test "
                  f"{sorted({a['filename_date'] for a, _ in sep})} against val "
                  f"{sorted({b['filename_date'] for _, b in sep})}")
        # numbering context, PER SERIES, since two series in one scene share no
        # counter and must never be printed as one run
        by_series = defaultdict(dict)
        for r in allrows:
            if r["scene"] == scene and r["num"] is not None:
                by_series[r["series"]].setdefault(
                    r["num"], SPLIT_MARK[r["split"]])
        for se in sorted(by_series):
            seq = " ".join(f"{n}[{by_series[se][n]}]"
                           for n in sorted(by_series[se]))
            print(f"     {se} numbering across all splits: {seq}")

    # ---- what an ADJACENT flag is worth, calibrated on the dated videos ----
    print("\n== how much an ADJACENT flag is worth ==")
    print("   Only the industrial videos carry filename dates, so they are the "
          "only\n   place the numbering can be calibrated against a known "
          "date. Two facts,\n   pulling in opposite directions:")
    dated = [r for r in allrows if r["num"] is not None and r["date"]]
    shown = 0
    for se in sorted({r["series"] for r in dated}):
        mem = [r for r in dated if r["series"] == se]
        same_num_diff_date = [(a, b) for i, a in enumerate(mem)
                              for b in mem[i + 1:]
                              if a["num"] == b["num"] and a["date"] != b["date"]]
        diff_num_diff_date = [(abs(a["num"] - b["num"]), a, b)
                              for i, a in enumerate(mem) for b in mem[i + 1:]
                              if a["num"] != b["num"] and a["date"] != b["date"]]
        for a, b in same_num_diff_date[:1]:
            shown += 1
            print(f"\n   {se} numbering RESETS. {a['video']} and\n   "
                  f"{b['video']} share the token {a['token']} yet carry dates "
                  f"{a['date']} and {b['date']},\n   {days(a['date'], b['date'])} "
                  f"days apart. Equal numbers do not imply one occasion.")
        if diff_num_diff_date:
            g, a, b = min(diff_num_diff_date, key=lambda p: p[0])
            shown += 1
            print(f"\n   {se} numbering PERSISTS across outings. {a['video']}\n"
                  f"   and {b['video']} are {g} apart in\n   numbering yet carry "
                  f"dates {a['date']} and {b['date']}, {days(a['date'], b['date'])} "
                  f"days apart.\n   A gap of {g} therefore spans more than one "
                  f"occasion here.")
    if not shown:
        print("   No dated pair shares a camera series, so no calibration is "
              "available.")
    print("\n   Taken together: a small numbering gap is CONSISTENT with one "
          "occasion and\n   does not establish it, and the meadow and fisheye "
          "videos carry no date at\n   all against which to check. The flag "
          "marks what the disk cannot separate,\n   not what it joins.")

    print("\n== one caveat on the fisheye numbering ==")
    print("   gopro_000 to gopro_008 are dataset-assigned names. No committed "
          "artefact\n   records whether that order reflects camera recording "
          "order, so the\n   ADJACENT flag on gopro_004 and gopro_005 rests on "
          "an ordering whose\n   provenance is not established. "
          "reports/gopro_continuity_findings.md and\n   "
          "reports/fisheye_session/FINDINGS.md carry what has been measured on "
          "that pair.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_scenes = len({r["scene"] for r in rows})
    print(f"\n{len(rows)} videos, {n_scenes} scenes. {flagged} flagged on "
          f"metadata, {len(closed)} dispositioned in the\nregister, "
          f"{len(open_scenes)} flagged with no disposition recorded.")
    if no_verdict:
        print(f"Dispositioned WITHOUT a verdict, basis on record only: "
              f"{sorted(no_verdict)}.")
    print("This script draws no verdict of its own. Where a scene carries a "
          "judgement it is\nthe author's, read from the committed register "
          "and reproduced verbatim.")
    if open_scenes:
        print(f"OPEN, no disposition recorded: {sorted(open_scenes)}")
    else:
        print("No flagged scene is left without a recorded disposition: the "
              "cross-subset\nsession check is COMPLETE.")
    print(f"Written to {OUT_CSV}")


if __name__ == "__main__":
    main()
