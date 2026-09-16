#!/usr/bin/env python3
"""
Derive the val and test denominator deficits exactly, from committed artefacts.

The split file gives test 6,905 and val 5,259 annotated boxes. The evaluator
reported 6,877 and 5,242. This script accounts for the 28 and the 17 per
video, and ASSERTS that each sums exactly. It exits non-zero on any shortfall
rather than reporting an approximate reconciliation.

THE RULE, as implemented. src/streaming_eval.py:41-43 sets T = 8,
LEAD = (T-1)//2 = 3 and TRAIL = T-1-LEAD = 4. The denominator is built by
count_ground_truth at src/streaming_eval.py:168, whose frame loop at line 176
is

    for f in range(LEAD, n - TRAIL):

so the evaluated centres of a video with n decoded frames are 3 .. n-5
inclusive. Boxes on the first LEAD = 3 and the last TRAIL = 4 decoded frames
are never counted, 7 frames per video, because those frames cannot sit at the
centre of a full T = 8 window. It is the same arithmetic the scoring loop uses
at lines 135-136, where a centre is emitted only once the buffer is full.

n is decoded_frames, not the container frame count: evaluate_final.py:278-280
reads it from reports/extraction_stats.csv.

WHY extraction_stats.csv CAN SUPPLY THE ANSWER. Its gt_boxes_in_valid_window
column is built by tools/extract_proposals.py at lines 256-257 and 281-286,
over first_valid = max(lead, warmup) to last_valid = n_dec - 1 - trail. With
bgs.warmup_frames ruled 0 (configs/ch4.yaml:105) that is max(3, 0) = 3 to
n-5, the identical window, and both scripts take n from the same column. The
column is therefore the evaluator's denominator per video, already committed.

Test has no committed per-frame box record, so its deficits are derived from
that aggregate. Val does have one, reports/instances_boxes.csv, so val is
ALSO derived independently at frame level and the two derivations are asserted
against each other. That cross-check is what establishes the rule empirically;
the test arithmetic then applies the same rule through shared code.

Usage:
    python tools/derive_window_exclusions.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SPLIT = "data/splits/dvb_splits_v2.0-static.csv"
STATS = "reports/extraction_stats.csv"
BOXES = "reports/instances_boxes.csv"
RESULTS = {"test": "reports/test_final_results.json",
           "val": "reports/val_final_results.json"}
OUT = Path("reports/window_exclusions.csv")

T = 8
LEAD = (T - 1) // 2
TRAIL = T - 1 - LEAD


def fail(lines: list[str]) -> None:
    print("FAIL the deficit does not close exactly.", file=sys.stderr)
    for ln in lines:
        print(f"  {ln}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    for p in (SPLIT, STATS, BOXES, *RESULTS.values()):
        if not Path(p).exists():
            print(f"FAIL missing {p}", file=sys.stderr)
            sys.exit(1)

    split = {r["video"]: r for r in csv.DictReader(open(SPLIT, newline=""))}
    stats = {r["video"]: r for r in csv.DictReader(open(STATS, newline=""))}
    reported = {k: json.load(open(v))["n_gt_boxes"] for k, v in RESULTS.items()}

    # per-frame val boxes, for the independent derivation
    val_bf: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    val_tot: dict[str, int] = defaultdict(int)
    for r in csv.DictReader(open(BOXES, newline="")):
        if r["split"] == "val":
            val_bf[r["video"]][int(r["frame"])] += 1
            val_tot[r["video"]] += 1

    print(f"T = {T}, LEAD = {LEAD}, TRAIL = {TRAIL}   "
          f"(src/streaming_eval.py:41-43)")
    print(f"evaluated centres = range(LEAD, n - TRAIL) = frames {LEAD} .. n-{TRAIL + 1}"
          f"   (src/streaming_eval.py:176)")
    print(f"excluded per video = LEAD + TRAIL = {LEAD + TRAIL} frames "
          f"(first {LEAD}, last {TRAIL})\n")

    out_rows, problems = [], []
    for part in ("test", "val"):
        vids = sorted(v for v, r in split.items() if r["split"] == part)
        print(f"===== {part.upper()} =====")
        print(f"{'video':<34}{'boxes':>7}{'decoded':>9}{'centres':>9}"
              f"{'counted':>9}{'excluded':>9}")
        tot_b = tot_c = tot_e = 0
        for v in vids:
            s = stats[v]
            b = int(split[v]["boxes"])
            n = int(s["decoded_frames"])
            counted = int(s["gt_boxes_in_valid_window"])
            vc = int(s["valid_centres"])
            exc = b - counted
            if vc != max(0, n - LEAD - TRAIL):
                problems.append(f"{v}: valid_centres {vc} != n - LEAD - TRAIL "
                                f"= {max(0, n - LEAD - TRAIL)}")
            if exc < 0:
                problems.append(f"{v}: counted {counted} exceeds split boxes {b}")
            print(f"{v:<34}{b:>7}{n:>9}{vc:>9}{counted:>9}{exc:>9}")
            out_rows.append({"split": part, "video": v, "scene": split[v]["scene"],
                             "split_boxes": b, "decoded_frames": n,
                             "valid_centres": vc, "counted_in_window": counted,
                             "excluded": exc,
                             "excluded_head_frames": f"0..{LEAD - 1}",
                             "excluded_tail_frames": f"{n - TRAIL}..{n - 1}"})
            tot_b += b
            tot_c += counted
            tot_e += exc
        print(f"{'TOTAL':<34}{tot_b:>7}{'':>9}{'':>9}{tot_c:>9}{tot_e:>9}")
        rep = reported[part]
        print(f"  {tot_b} - {tot_e} = {tot_b - tot_e};  evaluator reported {rep}")
        if tot_b - tot_e != rep:
            problems.append(f"{part}: {tot_b} - {tot_e} = {tot_b - tot_e}, "
                            f"evaluator reported {rep}, difference "
                            f"{rep - (tot_b - tot_e)}")
        if tot_c != rep:
            problems.append(f"{part}: counted column sums to {tot_c}, "
                            f"evaluator reported {rep}")
        print(f"  closes exactly: {tot_b - tot_e == rep and tot_c == rep}\n")

    # independent frame-level derivation, val only
    print("===== VAL, independent frame-level derivation "
          "(reports/instances_boxes.csv) =====")
    print(f"{'video':<34}{'head':>6}{'tail':>6}{'excl':>6}{'aggregate':>11}"
          f"   excluded frames carrying boxes")
    tot_f = 0
    for v in sorted(val_bf):
        n = int(stats[v]["decoded_frames"])
        b = int(split[v]["boxes"])
        if val_tot[v] != b:
            problems.append(f"{v}: instances_boxes.csv has {val_tot[v]} boxes, "
                            f"split says {b}")
        head = sum(c for f, c in val_bf[v].items() if f < LEAD)
        tail = sum(c for f, c in val_bf[v].items() if f > n - 1 - TRAIL)
        agg = b - int(stats[v]["gt_boxes_in_valid_window"])
        frames = sorted(f for f in val_bf[v] if f < LEAD or f > n - 1 - TRAIL)
        if head + tail != agg:
            problems.append(f"{v}: frame-level {head + tail} != aggregate {agg}")
        print(f"{v:<34}{head:>6}{tail:>6}{head + tail:>6}{agg:>11}   "
              f"{frames if frames else ''}")
        tot_f += head + tail
    print(f"{'TOTAL':<34}{'':>6}{'':>6}{tot_f:>6}")
    if tot_f != 17:
        problems.append(f"val frame-level total {tot_f}, expected 17")

    if problems:
        fail(problems)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nOK test deficit 28 and val deficit 17 both close exactly, and the "
          f"val\n   frame-level derivation agrees with the aggregate video by "
          f"video.\n   Written to {OUT}")


if __name__ == "__main__":
    main()
