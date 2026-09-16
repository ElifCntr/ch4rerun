#!/usr/bin/env python3
"""
Assert that every figure quoted in reports/fisheye_session/FINDINGS.md is
present in the run artefacts it cites.

The report is prose and the artefacts are the record. This script exists so a
figure cannot drift out of the prose unnoticed: it recomputes each band and
each quoted statistic from the CSVs and fails with a list of mismatches rather
than reconciling anything. Exits non-zero on any mismatch.

Usage:
    python tools/verify_fisheye_findings.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

OUT = Path("reports/fisheye_session")
DOC = OUT / "FINDINGS.md"
QUERY = {"gopro_004", "gopro_005"}


def bands(rows: list[dict], key: str):
    same = [float(r[key]) for r in rows
            if r["labelled_pair"] == "True" and r["same_session_label"] == "True"]
    diff = [float(r[key]) for r in rows
            if r["labelled_pair"] == "True" and r["same_session_label"] == "False"]
    if not same or not diff:
        print("FAIL no labelled reference pairs; the bands cannot be built",
              file=sys.stderr)
        sys.exit(1)
    return same, diff


def main() -> None:
    for p in (DOC, OUT / "pair_similarity.csv", OUT / "skyline_similarity.csv"):
        if not p.exists():
            print(f"FAIL missing {p}. Run tools/check_fisheye_session.py "
                  f"--allow-test first.", file=sys.stderr)
            sys.exit(1)

    doc = DOC.read_text()
    rows = list(csv.DictReader(open(OUT / "pair_similarity.csv")))
    sky = list(csv.DictReader(open(OUT / "skyline_similarity.csv")))
    q = [r for r in rows if {r["a"], r["b"]} == QUERY][0]
    qs = [r for r in sky if {r["a"], r["b"]} == QUERY][0]

    # Parse the two result tables by row label and compare the PARSED value
    # to the recomputed one. An earlier version of this script only asserted
    # that each number appeared somewhere in the report, which passes even
    # when the table is wrong, because the same figure is also quoted in the
    # prose. Presence is not agreement.
    def row(label: str, ncols: int) -> list[str]:
        for line in doc.splitlines():
            if line.strip().startswith("|") and label in line:
                cells = [c.strip().strip("*").strip()
                         for c in line.strip().strip("|").split("|")]
                if len(cells) == ncols:
                    return cells
        print(f"FAIL no {ncols}-column table row matching {label!r} in {DOC}",
              file=sys.stderr)
        sys.exit(1)

    bad: list[str] = []

    def eq(what: str, got: str, exp: str) -> None:
        if got != exp:
            bad.append(f"{what}\n      report:     {got}\n      recomputed: {exp}")

    def rng(vals, fmt, unit=""):
        return f"{min(vals):{fmt}} .. {max(vals):{fmt}}{unit}"

    ssim_s, ssim_d = bands(rows, "ssim")
    mae_s, mae_d = bands(rows, "mae")
    sky_s, sky_d = bands(sky, "skyline_median_abs_px")

    # background table: | band | n | SSIM | MAE |
    r = row("known SAME occasion | ", 4)
    eq("background, same band, n", r[1], str(len(ssim_s)))
    eq("background, same band, SSIM", r[2], rng(ssim_s, ".4f"))
    eq("background, same band, MAE", r[3], rng(mae_s, ".2f"))

    r = row("known DIFFERENT occasion | ", 4)
    eq("background, different band, n", r[1], str(len(ssim_d)))
    eq("background, different band, SSIM", r[2], rng(ssim_d, ".4f"))
    eq("background, different band, MAE", r[3], rng(mae_d, ".2f"))

    r = row("gopro_004 vs gopro_005", 4)
    eq("background, query SSIM", r[2], f"{float(q['ssim']):.4f}")
    eq("background, query MAE", r[3], f"{float(q['mae']):.2f}")

    # skyline table: | band | n | median abs difference |
    r = row("known SAME occasion | ", 3)
    eq("skyline, same band, n", r[1], str(len(sky_s)))
    eq("skyline, same band, range", r[2], rng(sky_s, ".0f", " px"))

    r = row("known DIFFERENT occasion | ", 3)
    eq("skyline, different band, n", r[1], str(len(sky_d)))
    eq("skyline, different band, range", r[2], rng(sky_d, ".0f", " px"))

    r = row("gopro_004 vs gopro_005", 3)
    eq("skyline, query median", r[2],
       f"{float(qs['skyline_median_abs_px']):.0f} px")

    # figures quoted only in prose
    prose = [
        ("query RMSE", f"RMSE {float(q['rmse']):.2f}"),
        ("query Pearson", f"Pearson r {float(q['pearson']):.4f}"),
        ("MAE distance above same band",
         f"{float(q['mae']) - max(mae_s):.1f} levels above"),
        ("MAE distance below different band",
         f"{min(mae_d) - float(q['mae']):.1f} levels below"),
        ("skyline distance above same band",
         f"{float(qs['skyline_median_abs_px']) - max(sky_s):.0f} px above"),
        ("skyline distance below different band",
         f"{min(sky_d) - float(qs['skyline_median_abs_px']):.0f} px below"),
    ]
    missing = [f"{what}: {text!r}" for what, text in prose if text not in doc]

    if bad or missing:
        print(f"FAIL {DOC} does not match the artefacts.", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        for m in missing:
            print(f"  - absent from the report: {m}", file=sys.stderr)
        sys.exit(1)

    print(f"OK every figure quoted in {DOC} is present in "
          f"pair_similarity.csv and skyline_similarity.csv")
    print(f"   reference bands: same n={len(ssim_s)}, different n={len(ssim_d)}")
    print(f"   query {sorted(QUERY)}: SSIM {float(q['ssim']):.4f}, "
          f"MAE {float(q['mae']):.2f}, skyline median "
          f"{float(qs['skyline_median_abs_px']):.0f} px")


if __name__ == "__main__":
    main()
