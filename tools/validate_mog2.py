#!/usr/bin/env python3
"""
1b-M step 3, task 1. MOG2 learningRate / history interaction check.

Runs BEFORE anything else in step 3. If an explicit learningRate bypasses
history, the history ruling is inert and the absorption-target question is
moot.

Three tests, all on a seeded synthetic sequence so the result depends on the
installed OpenCV build and on nothing else.

  A. Two histories at DEFAULT learningRate (-1). Masks must DIFFER.
  B. Two histories at EXPLICIT learningRate. Masks must be IDENTICAL,
     which is what "history is bypassed" means.
  C. Observed absorption time for a static object, against the Zivkovic
     prediction ln(1-cf)/ln(1-alpha) at terminal alpha = 1/history.

Test C exists because OpenCV ramps alpha as 1/min(2*nframes, history), so
alpha only reaches 1/history after history/2 frames. Absorption is therefore
FASTER than the terminal-alpha prediction, and C measures by how much.

Writes reports/mog2_learning_rate_check.json and prints a summary.
Exits non-zero if A or B fails.

Usage:
    python scripts/validate_mog2.py [--out reports/]
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

# Synthetic sequence, fixed so the check is reproducible.
H, W = 240, 320
WARMUP = 60          # frames of background only before the object appears
NFRAMES = 900        # total frames
SQ = 40              # side of the static object, px
SQ_TOP, SQ_LEFT = 100, 140
NOISE_SIGMA = 3.0
SEED = 20260815


def make_sequence(seed=SEED):
    """Static textured background, mild per-frame noise, then a static bright
    square from WARMUP onward that never moves."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 180, size=(H, W), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3.0)
    frames = []
    for t in range(NFRAMES):
        f = base.astype(np.float32) + rng.normal(0.0, NOISE_SIGMA, size=(H, W))
        if t >= WARMUP:
            f[SQ_TOP:SQ_TOP + SQ, SQ_LEFT:SQ_LEFT + SQ] = 250.0
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


def run(frames, history, learning_rate):
    """Return the list of foreground masks. learning_rate None means the
    OpenCV default is used, i.e. apply() is called with no rate at all."""
    sub = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=16.0, detectShadows=False
    )
    masks = []
    for f in frames:
        if learning_rate is None:
            m = sub.apply(f)
        else:
            m = sub.apply(f, learningRate=learning_rate)
        masks.append(m)
    return masks


def n_differing(a, b):
    return sum(1 for x, y in zip(a, b) if not np.array_equal(x, y))


def absorption_frame(masks, min_run=5, frac=0.5):
    """First frame index t >= WARMUP such that the object region is below
    `frac` foreground for `min_run` consecutive frames. Returns the offset
    from WARMUP, or None if never absorbed."""
    below = 0
    for t in range(WARMUP, len(masks)):
        region = masks[t][SQ_TOP:SQ_TOP + SQ, SQ_LEFT:SQ_LEFT + SQ]
        fg = float((region > 0).mean())
        if fg < frac:
            below += 1
            if below >= min_run:
                return (t - min_run + 1) - WARMUP
        else:
            below = 0
    return None


def predicted_absorption(cf, history):
    """Zivkovic ln(1-cf)/ln(1-alpha) at terminal alpha = 1/history."""
    alpha = 1.0 / history
    return math.log(1.0 - cf) / math.log(1.0 - alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = make_sequence()
    probe = cv2.createBackgroundSubtractorMOG2()

    env = {
        "opencv_version": cv2.__version__,
        "opencl_available": bool(cv2.ocl.haveOpenCL()),
        "opencl_in_use": bool(cv2.ocl.useOpenCL()),
        "default_history": int(probe.getHistory()),
        "default_backgroundRatio": float(probe.getBackgroundRatio()),
        "default_varThreshold": float(probe.getVarThreshold()),
        "default_varThresholdGen": float(probe.getVarThresholdGen()),
        "default_nmixtures": int(probe.getNMixtures()),
        "default_complexityReductionThreshold": float(
            probe.getComplexityReductionThreshold()
        ),
        "default_detectShadows": bool(probe.getDetectShadows()),
        "sequence": {
            "h": H, "w": W, "nframes": NFRAMES, "warmup": WARMUP,
            "square_side": SQ, "noise_sigma": NOISE_SIGMA, "seed": SEED,
        },
    }

    # cf implied by the OpenCV default, on the source-comment correspondence
    # backgroundRatio = 1 - cf. Recorded, not assumed elsewhere.
    cf_implied = 1.0 - env["default_backgroundRatio"]

    h_lo, h_hi = 100, 1000

    # A. default learningRate, history must matter
    a_lo = run(frames, h_lo, None)
    a_hi = run(frames, h_hi, None)
    a_diff = n_differing(a_lo, a_hi)
    a_pass = a_diff > 0

    # B. explicit learningRate, history must NOT matter
    explicit_lr = 0.001  # the value the original Ch4 config passed
    b_lo = run(frames, h_lo, explicit_lr)
    b_hi = run(frames, h_hi, explicit_lr)
    b_diff = n_differing(b_lo, b_hi)
    b_pass = b_diff == 0

    # C. observed vs predicted absorption, default learningRate
    obs = {}
    for hist in (100, 200, 500, 1000, 2000):
        masks = run(frames, hist, None)
        t_obs = absorption_frame(masks)
        t_pred = predicted_absorption(cf_implied, hist)
        obs[hist] = {
            "observed_frames": t_obs,
            "predicted_terminal_alpha_frames": round(t_pred, 1),
            "ratio_observed_over_predicted": (
                round(t_obs / t_pred, 3) if t_obs is not None else None
            ),
            "history_exceeds_2x_clip": bool(hist > 2 * NFRAMES),
        }

    # C2. the same, with the explicit rate, to show what the original ran at
    masks_lr = run(frames, 200, explicit_lr)
    obs_explicit = {
        "history_set": 200,
        "learningRate": explicit_lr,
        "observed_frames": absorption_frame(masks_lr),
        "predicted_at_lr_frames": round(
            math.log(1 - cf_implied) / math.log(1 - explicit_lr), 1
        ),
        "predicted_at_history_frames": round(
            predicted_absorption(cf_implied, 200), 1
        ),
    }

    result = {
        "environment": env,
        "cf_implied_by_default_backgroundRatio": cf_implied,
        "test_A_default_rate_history_matters": {
            "history_low": h_lo, "history_high": h_hi,
            "frames_with_differing_masks": a_diff,
            "total_frames": NFRAMES, "pass": a_pass,
        },
        "test_B_explicit_rate_history_bypassed": {
            "history_low": h_lo, "history_high": h_hi,
            "learningRate": explicit_lr,
            "frames_with_differing_masks": b_diff,
            "total_frames": NFRAMES, "pass": b_pass,
        },
        "test_C_absorption_observed_vs_predicted": obs,
        "test_C2_original_config_shape": obs_explicit,
    }

    path = os.path.join(args.out, "mog2_learning_rate_check.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"OpenCV {env['opencv_version']}  "
          f"OpenCL available={env['opencl_available']} "
          f"in_use={env['opencl_in_use']}")
    print(f"default backgroundRatio {env['default_backgroundRatio']} "
          f"-> cf implied {cf_implied:.3f}")
    print(f"default complexityReductionThreshold "
          f"{env['default_complexityReductionThreshold']} "
          f"(Zivkovic 2004 Sec 4 suggests cT=0.01)")
    print()
    print(f"A default rate, history 100 vs 1000: "
          f"{a_diff}/{NFRAMES} frames differ -> "
          f"{'PASS history is live' if a_pass else 'FAIL history is inert'}")
    print(f"B explicit rate {explicit_lr}, history 100 vs 1000: "
          f"{b_diff}/{NFRAMES} frames differ -> "
          f"{'PASS history is bypassed' if b_pass else 'FAIL'}")
    print()
    print("C absorption, default rate. history / observed / predicted "
          "at terminal alpha / ratio")
    for hist, r in obs.items():
        print(f"   {hist:>5}  {str(r['observed_frames']):>7}  "
              f"{r['predicted_terminal_alpha_frames']:>8}  "
              f"{r['ratio_observed_over_predicted']}")
    print()
    print(f"C2 original config shape, history 200 with explicit lr "
          f"{explicit_lr}: observed {obs_explicit['observed_frames']}, "
          f"predicted at lr {obs_explicit['predicted_at_lr_frames']}, "
          f"predicted at history {obs_explicit['predicted_at_history_frames']}")
    print(f"\nwritten to {path}")

    if not (a_pass and b_pass):
        sys.exit(1)


if __name__ == "__main__":
    main()
