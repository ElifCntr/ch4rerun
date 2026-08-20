"""Validation for the ViBe transcription. Run before any sweep uses it.

Three things named in the ruling, plus the checks that make them meaningful.

1. DETERMINISM BOTH WAYS. Same seed must give identical output, because the
   rerun's whole protocol rests on reproducibility. Different seeds must give
   SLIGHTLY DIFFERENT output, because the paper states outright that ViBe is
   non-deterministic and that rerunning the same sequence always differs a
   little. An implementation that is identical across seeds has lost the
   random replacement policy and is not ViBe.

2. SEGMENTATION FROM THE SECOND FRAME. The paper's stated advantage over
   methods needing dozens of initialisation frames.

3. GHOST ABSORPTION. A moving object present in the first frame poisons the
   neighbourhood-initialised model and leaves a ghost. The paper says the
   update mechanism absorbs it over subsequent frames. The ghost must appear
   AND then fade; a ghost that never appears means initialisation is not doing
   what the paper describes, and one that never fades means the update is not.

Every threshold in this file is a property of the synthetic sequence, chosen
so a correct implementation passes comfortably and a broken one fails. None is
a pipeline parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vibe import ViBe  # noqa: E402

results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append(("PASS" if ok else "FAIL", name))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    return ok


def sequence(n=80, size=100, box=10, start_in_frame=False):
    """Textured static background, one bright square moving left to right.

    start_in_frame=True places the square inside the frame from frame 0, which
    is the condition that produces a ghost.
    """
    rng = np.random.default_rng(0)
    bg = rng.integers(70, 120, (size, size), dtype=np.uint8)
    frames = []
    for i in range(n):
        f = bg.copy()
        if start_in_frame:
            x = 20 + int((size - 40 - box) * i / max(1, n - 1))
        else:
            x = -box + int((size + box) * i / max(1, n - 1))
        y = size // 2
        x0, x1 = max(0, x), min(size, x + box)
        if x1 > x0:
            f[y:y + box, x0:x1] = 240
        frames.append(f)
    return frames, bg


def run(frames, seed):
    v = ViBe(seed=seed)
    return [v.apply(f) for f in frames]


def main() -> None:
    print("ViBe validation, Barnich & Van Droogenbroeck Appendix A "
          "transcription\n")

    print("SEED REQUIRED")
    try:
        ViBe(seed=None)
        check("constructor rejects a missing seed", False, "it did not")
    except ValueError:
        check("constructor rejects a missing seed", True)

    frames, bg = sequence()

    print("\nDETERMINISM")
    a = run(frames, 0)
    b = run(frames, 0)
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    check("same seed gives identical masks on every frame", same)

    c = run(frames, 1)
    diffs = [int((x != y).sum()) for x, y in zip(a, c)]
    total_diff = sum(diffs)
    check("different seed gives DIFFERENT masks", total_diff > 0,
          f"{total_diff} differing pixels across {len(frames)} frames")
    # "Slightly" different, per the paper. If two seeds disagree on most of
    # the frame the model is not converging and something is wrong.
    px = frames[0].size * len(frames)
    check("the difference is SLIGHT, not wholesale",
          total_diff / px < 0.05, f"{total_diff/px:.4f} of all pixels")

    print("\nSEGMENTATION FROM THE SECOND FRAME")
    check("frame 1 returns an empty mask by construction",
          int((a[0] > 0).sum()) == 0, f"{int((a[0] > 0).sum())} px")
    check("frame 2 already segments something",
          int((a[1] > 0).sum()) > 0, f"{int((a[1] > 0).sum())} px foreground")

    print("\nTARGET IS FOUND AND THE BACKGROUND IS NOT")
    late = a[-15:]
    fg = [int((m > 0).sum()) for m in late]
    check("foreground present in the last 15 frames", all(v > 0 for v in fg),
          f"counts {fg[:5]}...")
    frac = max(fg) / float(frames[0].size)
    check("foreground stays a small fraction of the frame", frac < 0.10,
          f"max {frac:.4f}")

    print("\nGHOST APPEARANCE AND ABSORPTION")
    print("  The absorption RATE is governed by phi. The paper reports a "
          "ghost fading in about 2 s at 30 fps with phi=1, and taking 2 min "
          "at phi=64, so the timescale is roughly linear in phi and phi=16 "
          "implies on the order of a thousand frames. The mechanism is "
          "therefore validated at phi=1, which is a MECHANISM CHECK and not "
          "an operating value; the phi=16 timescale is reported separately "
          "below because it bears on 1b-M.")
    gframes, _ = sequence(n=200, start_in_frame=True)
    size = gframes[0].shape[0]
    # The object starts at x=20 and moves right, so the ghost sits on its
    # initial footprint. Measure only after the object has left that footprint
    # entirely, or the object is counted as its own ghost.
    ghost_region = (slice(size // 2, size // 2 + 10), slice(20, 30))

    # TIMING, which is easy to get wrong. While the object still covers its
    # own starting footprint it MATCHES the model that was initialised from
    # it, so it reads as background and there is no ghost. The ghost appears
    # only once the object moves off and the real background is exposed
    # against an object-derived model. Here the object clears x=20..30 at
    # about frame 40.
    v1 = ViBe(seed=0, phi=1)
    g1 = [v1.apply(f) for f in gframes]
    at_start = int((g1[2][ghost_region] > 0).sum())
    exposed = max(int((g1[i][ghost_region] > 0).sum())
                  for i in range(45, 90))
    final1 = int((g1[-1][ghost_region] > 0).sum())
    check("object hides itself where it initialised the model",
          at_start == 0, f"{at_start} px at frame 3")
    check("phi=1: a ghost appears once the object moves off", exposed > 0,
          f"peak {exposed} px over frames 45-90")
    check("phi=1: the ghost is then absorbed", final1 < exposed,
          f"peak {exposed} px, final {final1} px")

    v16 = ViBe(seed=0, phi=16)
    g16 = [v16.apply(f) for f in gframes]
    exp16 = max(int((g16[i][ghost_region] > 0).sum())
                for i in range(45, 90))
    final16 = int((g16[-1][ghost_region] > 0).sum())
    print(f"  phi=16, the paper's default, over {len(gframes)} frames: "
          f"peak {exp16} px, final {final16} px. NOT a pass/fail check. "
          f"Recorded because absorption is slow at phi=16 by design, and "
          f"that bears on 1b-M: see the note the script prints at the end.")

    print("\nPARAMETERS ARE THE PAPER'S")
    d = ViBe(seed=0).describe()
    check("N=20, R=20, Nmin=2, phi=16",
          (d["N"], d["R"], d["Nmin"], d["phi"]) == (20, 20, 2, 16), str(d))

    n_fail = sum(1 for s, _ in results if s == "FAIL")
    print(f"\n{len(results)} checks: {len(results) - n_fail} pass, "
          f"{n_fail} fail")
    if n_fail:
        print("\nDo not use this ViBe in any sweep until these pass.")
        sys.exit(1)
    print("\nViBe behaves as the paper describes. Every parameter is the "
          "paper's default, declared and not chosen.")
    print("\nNOTE FOR 1b-M, from the ghost timing above. The paper reports "
          "ghost suppression in about 2 s at 30 fps with phi=1 and about "
          "2 min with phi=64, so the timescale is roughly linear in phi and "
          "phi=16 implies on the order of a thousand frames. Most DvB static "
          "clips are shorter than that: durations run from 177 to 3,481 "
          "frames with a median near 570. So on this dataset a ViBe ghost "
          "will persist through much or all of a clip wherever a drone is "
          "present in the first frame. That is expected behaviour, not a "
          "fault, but it is a real cost of ViBe's single-frame "
          "initialisation on short clips and it should be measured in 1b-M "
          "rather than assumed either way.")


if __name__ == "__main__":
    main()
