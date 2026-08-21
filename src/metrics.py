"""Region-level average precision over fixed motion proposals.

THE DENOMINATOR IS EVERY GROUND-TRUTH BOX, not every matched one. A drone the
subtractor never proposed is a miss no classifier can recover, so it must
count against recall. Dividing by matched boxes instead would hide the
proposal ceiling entirely and report a number about the classifier that reads
as a number about the system. That ceiling is the chapter's headline
limitation, so the metric has to carry it.

IGNORED PROPOSALS ARE EXCLUDED FROM BOTH COUNTS. A blob overlapping a drone
partially is neither a hit nor a false alarm, which is what the ignore band
was ruled for. They are dropped before ranking, never scored as negatives.

ALL-POINT INTERPOLATION, not the 11-point variant. AP is the sum over each
recall increment of the precision at that point, which is the area under the
precision-recall curve without smoothing. The 11-point form is a coarser
approximation kept for historical comparability with old VOC numbers and
nothing here needs it.

ONE PROPOSAL PER GROUND-TRUTH BOX. Greedy one-to-one assignment already ran at
extraction, so no ground-truth box has two positives and duplicate-detection
handling is unnecessary. This module asserts that rather than assuming it.

NOT RULED, AND THIS MODULE DOES NOT CHOOSE: how the operating threshold is
selected from the precision-recall curve for the reported operating point. AP
itself is threshold-free, so nothing here needs it, but the results section
will.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

POSITIVE, NEGATIVE, IGNORED = "positive", "negative", "ignored"


def average_precision(scores, labels, n_gt_boxes, gt_index=None):
    """All-point AP.

    scores      per-proposal score for the positive class
    labels      per-proposal label, one of positive / negative / ignored
    n_gt_boxes  EVERY ground-truth box in the evaluated set, including those
                no proposal covered. This is what bounds recall.
    gt_index    optional, per-proposal ground-truth index; when given, the
                one-to-one property is asserted rather than trusted.

    Returns (ap, precision, recall, n_scored, n_ignored).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels differ in length")
    if n_gt_boxes <= 0:
        raise ValueError("n_gt_boxes must be positive; it is the denominator")

    keep = labels != IGNORED
    n_ignored = int((~keep).sum())
    s, lab = scores[keep], labels[keep]

    if gt_index is not None:
        gi = np.asarray(gt_index)[keep]
        pos = gi[lab == POSITIVE]
        if pos.size != np.unique(pos).size:
            raise ValueError(
                "a ground-truth box has more than one positive proposal; "
                "greedy one-to-one assignment should make that impossible")

    if s.size == 0:
        return 0.0, np.array([1.0]), np.array([0.0]), 0, n_ignored

    order = np.argsort(-s, kind="stable")
    hit = (lab[order] == POSITIVE).astype(np.float64)
    tp = np.cumsum(hit)
    fp = np.cumsum(1.0 - hit)

    recall = tp / float(n_gt_boxes)
    precision = tp / np.maximum(tp + fp, 1e-12)

    # all-point: sum of precision at each point where recall increases
    prev_r = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        if r > prev_r:
            ap += p * (r - prev_r)
            prev_r = r
    return float(ap), precision, recall, int(s.size), n_ignored


def proposal_ceiling(labels, n_gt_boxes):
    """The recall no classifier can exceed: matched boxes over all boxes."""
    labels = np.asarray(labels)
    return float((labels == POSITIVE).sum()) / float(n_gt_boxes)


def by_group(scores, labels, groups, n_gt_by_group, gt_index=None):
    """AP and ceiling per group, for the per-scene and per-partition tables.

    n_gt_by_group must count EVERY ground-truth box in each group, including
    unproposed ones, so a group's ceiling is honest.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    gi = None if gt_index is None else np.asarray(gt_index)

    out = {}
    for g in sorted(set(groups.tolist()) | set(n_gt_by_group)):
        m = groups == g
        n_gt = n_gt_by_group.get(g)
        if not n_gt:
            out[g] = {"ap": None, "ceiling": None, "n_gt": 0,
                      "note": "no ground-truth boxes in this group"}
            continue
        ap, _, _, n_scored, n_ign = average_precision(
            scores[m], labels[m], n_gt, None if gi is None else gi[m])
        out[g] = {"ap": ap,
                  "ceiling": proposal_ceiling(labels[m], n_gt),
                  "n_gt": int(n_gt), "n_scored": n_scored,
                  "n_ignored": n_ign}
    return out


def recall_at_score(scores, labels, n_gt_boxes, threshold):
    """Recall and precision at one operating threshold, for the results table
    once a threshold has been selected elsewhere."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    keep = (labels != IGNORED) & (scores >= threshold)
    tp = int((labels[keep] == POSITIVE).sum())
    fp = int((labels[keep] == NEGATIVE).sum())
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp,
        "recall": tp / float(n_gt_boxes),
        "precision": tp / max(tp + fp, 1),
        "n_selected": tp + fp,
    }


def size_band_recall(labels, area_fracs, band_edges, n_gt_by_band):
    """Per-size-band recall, reported adjacent to pooled AP.

    Bands are the benchmark-derived REPORTING bands and are not the
    stratification bands used for the negative draw. Passing the wrong ones
    silently produces a plausible table, so the caller names them.
    """
    labels = np.asarray(labels)
    area_fracs = np.asarray(area_fracs, dtype=np.float64)
    idx = np.digitize(area_fracs, band_edges, right=False)
    out = {}
    for b in range(len(band_edges) + 1):
        m = idx == b
        n_gt = n_gt_by_band.get(b, 0)
        out[b] = {
            "matched": int((labels[m] == POSITIVE).sum()),
            "n_gt": int(n_gt),
            "recall": (float((labels[m] == POSITIVE).sum()) / n_gt)
            if n_gt else None,
        }
    return out
