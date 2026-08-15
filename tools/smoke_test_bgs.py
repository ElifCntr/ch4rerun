"""OpenCV 5 smoke test for MOG2 and KNN. Run before any 1b-M sweep.

The cloned ch4rerun environment carries opencv-python 5.0.0.93. OpenCV 5 is a
major version and the background-subtraction API has not been verified on it.
A changed constructor signature, a changed apply() contract or a changed
shadow encoding would invalidate every sweep run behind it, silently in the
worst case.

This checks behaviour, not just existence. Each test states what it expects
and why, and a failure names what to do about it.

It sets no parameter for the pipeline. The values used here are OpenCV's own
documented defaults plus a synthetic sequence, and nothing measured here
should be carried into 1b-M as a chosen value.

WHAT IT DOES NOT COVER. ViBe is not in OpenCV at any version, so nothing here
speaks to it. The ViBe arm is transcribed from Barnich and Van Droogenbroeck
Appendix A and validated separately.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    status = PASS if ok else (WARN if warn_only else FAIL)
    results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def synthetic(n: int = 60, size: int = 120, box: int = 8):
    """Static textured background with one small bright square moving across.

    Textured rather than flat, because a flat background makes any subtractor
    look perfect and would hide a real regression. Box size 8 is chosen to sit
    near the measured DvB median of about 18 px equivalent side scaled down to
    this frame, so the test exercises the small-target regime the chapter
    cares about rather than a large blob.
    """
    rng = np.random.default_rng(0)
    bg = rng.integers(60, 110, (size, size), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    frames = []
    for i in range(n):
        f = bg.copy()
        x = 10 + int((size - 20 - box) * i / max(1, n - 1))
        y = size // 2
        f[y:y + box, x:x + box] = 230
        frames.append(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    return frames


def main() -> None:
    print(f"OpenCV {cv2.__version__}")
    print(f"numpy  {np.__version__}\n")

    frames = synthetic()

    # ------------------------------------------------------ construction ---
    print("CONSTRUCTION")
    try:
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=16, detectShadows=False)
        check("MOG2 keyword constructor", True)
    except Exception as exc:  # noqa: BLE001
        check("MOG2 keyword constructor", False, str(exc))
        mog2 = None
    try:
        knn = cv2.createBackgroundSubtractorKNN(
            history=200, dist2Threshold=400.0, detectShadows=False)
        check("KNN keyword constructor", True)
    except Exception as exc:  # noqa: BLE001
        check("KNN keyword constructor", False, str(exc))
        knn = None
    if mog2 is None or knn is None:
        print("\nConstruction failed. Everything downstream is void; stop "
              "and report before writing any sweep.")
        sys.exit(1)

    # --------------------------------------------------------- accessors ---
    print("\nACCESSORS  (1b-M declares history from the Zivkovic absorption "
          "formula, so setHistory must work)")
    try:
        mog2.setHistory(321)
        check("MOG2 setHistory/getHistory round trip",
              mog2.getHistory() == 321, f"got {mog2.getHistory()}")
        mog2.setVarThreshold(19.0)
        check("MOG2 setVarThreshold round trip",
              abs(mog2.getVarThreshold() - 19.0) < 1e-6,
              f"got {mog2.getVarThreshold()}")
        check("MOG2 getDetectShadows is False as constructed",
              mog2.getDetectShadows() is False,
              f"got {mog2.getDetectShadows()}")
        mog2.setHistory(200)
        mog2.setVarThreshold(16.0)
    except Exception as exc:  # noqa: BLE001
        check("MOG2 accessors", False, str(exc))
    try:
        knn.setHistory(321)
        check("KNN setHistory round trip", knn.getHistory() == 321,
              f"got {knn.getHistory()}")
        knn.setHistory(200)
    except Exception as exc:  # noqa: BLE001
        check("KNN accessors", False, str(exc))

    # ------------------------------------------------------------ apply ----
    print("\nAPPLY CONTRACT")
    for name, sub in (("MOG2", mog2), ("KNN", knn)):
        m = sub.apply(frames[0])
        check(f"{name} apply returns ndarray", isinstance(m, np.ndarray))
        check(f"{name} mask is single channel", m.ndim == 2,
              f"shape {m.shape}")
        check(f"{name} mask is uint8", m.dtype == np.uint8, f"{m.dtype}")
        check(f"{name} mask matches frame size",
              m.shape[:2] == frames[0].shape[:2],
              f"{m.shape[:2]} vs {frames[0].shape[:2]}")

    print("\nLEARNING RATE  (1b-M runs at the OpenCV default, declared as a "
          "default; -1 means auto)")
    try:
        cv2.createBackgroundSubtractorMOG2().apply(frames[0], learningRate=-1)
        check("apply accepts learningRate=-1", True)
    except Exception as exc:  # noqa: BLE001
        check("apply accepts learningRate=-1", False, str(exc))
    try:
        cv2.createBackgroundSubtractorMOG2().apply(frames[0],
                                                   learningRate=0.001)
        check("apply accepts an explicit learningRate", True)
    except Exception as exc:  # noqa: BLE001
        check("apply accepts an explicit learningRate", False, str(exc))

    # -------------------------------------------------------- behaviour ----
    print("\nBEHAVIOUR ON A MOVING SMALL TARGET")
    for name, sub in (("MOG2", cv2.createBackgroundSubtractorMOG2(
                            history=200, varThreshold=16,
                            detectShadows=False)),
                      ("KNN", cv2.createBackgroundSubtractorKNN(
                            history=200, dist2Threshold=400.0,
                            detectShadows=False))):
        masks = [sub.apply(f) for f in frames]
        late = masks[-10:]
        fg = [int((m > 0).sum()) for m in late]
        check(f"{name} finds foreground in the last 10 frames",
              all(v > 0 for v in fg), f"pixel counts {fg}")
        # The target is 8x8 = 64 px. A subtractor that flags a large fraction
        # of a 120x120 frame is not segmenting, it is failing.
        frac = max(fg) / float(frames[0].shape[0] * frames[0].shape[1])
        check(f"{name} foreground is a small fraction of the frame",
              frac < 0.10, f"max {frac:.3f} of the frame")
        # Contours are how proposals are actually formed downstream.
        cnts, _ = cv2.findContours(late[-1], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        check(f"{name} findContours works on the mask", True,
              f"{len(cnts)} contours in the final frame")

    print("\nSHADOW ENCODING  (detectShadows=True marks shadows 127, not 255; "
          "1b-M sets it False, so this only checks the encoding has not moved)")
    s = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    for f in frames:
        sm = s.apply(f)
    vals = set(np.unique(sm).tolist())
    check("shadow mask values are a subset of {0, 127, 255}",
          vals <= {0, 127, 255}, f"observed {sorted(vals)}", warn_only=True)

    # ---------------------------------------------------------- determinism
    print("\nDETERMINISM  (MOG2 and KNN should be deterministic; ViBe is NOT, "
          "by design, and is validated separately)")
    for name, mk in (("MOG2", lambda: cv2.createBackgroundSubtractorMOG2(
                          history=200, varThreshold=16, detectShadows=False)),
                     ("KNN", lambda: cv2.createBackgroundSubtractorKNN(
                          history=200, dist2Threshold=400.0,
                          detectShadows=False))):
        s1, s2 = mk(), mk()
        for f in frames:
            m1, m2 = s1.apply(f), s2.apply(f)
        check(f"{name} two instances give identical masks",
              bool(np.array_equal(m1, m2)))

    # ----------------------------------------------------------- summary ---
    n_fail = sum(1 for st, _, _ in results if st == FAIL)
    n_warn = sum(1 for st, _, _ in results if st == WARN)
    print(f"\n{len(results)} checks: {len(results) - n_fail - n_warn} pass, "
          f"{n_warn} warn, {n_fail} fail")
    if n_fail:
        print("\nFAILURES:")
        for st, name, detail in results:
            if st == FAIL:
                print(f"  {name}  -- {detail}")
        print("\nDo not proceed to the sweeps. Report the failure; a "
              "major-version API change invalidates everything behind it.")
        sys.exit(1)
    print("\nMOG2 and KNN behave as expected on OpenCV "
          f"{cv2.__version__}. Nothing measured here is a chosen value.")


if __name__ == "__main__":
    main()
