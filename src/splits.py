"""Split access with a hard guard on the test partition.

The test partition is unreachable unless a caller passes allow_test=True at the
top level. Stage 1a never passes it. The guard exists because the original
Chapter 4 work tuned a decision threshold on test, which voided the result.
"""

from __future__ import annotations

import csv
from pathlib import Path

TEST_PARTITION = "test"


class TestSetAccessError(RuntimeError):
    pass


def load_split(split_file: str | Path, partitions: list[str],
               allow_test: bool = False) -> dict[str, list[dict]]:
    """Return {partition: [row, ...]} for the requested partitions.

    Raises unless allow_test is explicitly True when test is requested.
    """
    wanted = [p.strip().lower() for p in partitions]
    if TEST_PARTITION in wanted and not allow_test:
        raise TestSetAccessError(
            "The test partition was requested without allow_test=True. "
            "Stage 1a is train and val only. If a later stage genuinely needs "
            "test, the calling script must take an explicit top-level flag and "
            "pass allow_test=True, and the run must be logged as touching test."
        )

    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run tools/build_split.py first; it writes "
            f"the split only after every integrity check passes."
        )

    out: dict[str, list[dict]] = {p: [] for p in wanted}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            part = row["split"].strip().lower()
            if part in out:
                out[part].append(row)

    missing = [p for p, rows in out.items() if not rows]
    if missing:
        raise ValueError(f"No rows found for partition(s): {missing}")
    return out


def session_overlap(rows_by_partition: dict[str, list[dict]]) -> dict:
    """Sessions appearing in more than one partition. Empty dict means clean."""
    seen: dict[str, set[str]] = {}
    for part, rows in rows_by_partition.items():
        for row in rows:
            seen.setdefault(row["session"], set()).add(part)
    return {s: sorted(p) for s, p in seen.items() if len(p) > 1}
