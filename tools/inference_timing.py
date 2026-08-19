#!/usr/bin/env python3
"""
Forward-only inference timing. Run on the P5000.

Authorised 18 August. The validation cost estimate rested on an assumption
that inference is about a third of training, which was never measured and is
load-bearing: at the ruled 7.23e-06 limit val holds 666,013 proposals, and the
assumption put one full validation pass at about 10.5 hours.

IT IMPORTS build_arm FROM epoch_timing RATHER THAN REBUILDING THE MODELS. The
figure produced here is divided against the 0.170 s per tubelet per epoch
training coefficient, so any difference in model definition would silently
corrupt the ratio. Identical definitions by construction, not by care.

TWO DIFFERENCES FROM TRAINING, both deliberate and both noted by the manager.
  - eval() and no_grad(), so no activations are stored for backward.
  - The batch ceiling is found FORWARD-ONLY and is therefore not the training
    ceiling of 162. Inference should fit far more.

Reports seconds per tubelet per arm, the whole-matrix coefficient over four
arms and three seeds, the measured ratio against training, and the recomputed
full-validation cost at the ruled limit and the fallback.

Nothing is chosen here.

Usage:
    python tools/inference_timing.py
    python tools/inference_timing.py --t 8 --ceiling 4096
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epoch_timing import ARMS, N_CLASSES, build_arm  # noqa: E402
from src.config import load_config  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

# Measured 12 Aug at T=8, batch 162, from reports/epoch_timing.csv.
# Training seconds per ITERATION, order as in ARMS.
TRAIN_SEC_PER_ITER_T8 = {
    "2d_single": 0.124,
    "2d_tframe_logitavg": 0.990,
    "tsm": 1.250,
    "r3d_18": 6.812,
}
TRAIN_BATCH_T8 = 162
SEEDS = 3

# Measured 18 Aug, tools/size_storage.py. Val proposals at each candidate
# normalised lower limit.
VAL_PROPOSALS = {7.23e-06: 666_013, 1.2e-05: 322_416, 2e-05: 162_881}
# Estimated, NOT measured: test has not been run. val's 120.2 proposals per
# frame applied to test's 7,227 frames.
TEST_PROPOSALS_ESTIMATE = {7.23e-06: 868_685}


def fits(arm, t, batch, device) -> bool:
    try:
        model, shape = build_arm(arm, t)
        model = model.to(device).eval()
        x = torch.randn(batch, *shape, device=device)
        with torch.no_grad():
            for _ in range(2):
                model(x)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        torch.cuda.empty_cache()


def max_batch(arm, t, device, ceiling) -> int:
    best, b = 0, 1
    while b <= ceiling and fits(arm, t, b, device):
        best, b = b, b * 2
    if best == 0:
        return 0
    b = best + max(best // 8, 1)
    while b <= ceiling and fits(arm, t, b, device):
        best = b
        b += max(best // 8, 1)
    return best


def time_forward(arm, t, batch, device, warmup=5, iters=20) -> float:
    model, shape = build_arm(arm, t)
    model = model.to(device).eval()
    x = torch.randn(batch, *shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
    del model, x
    torch.cuda.empty_cache()
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--t", type=int, default=8, help="ruled 12 Aug")
    ap.add_argument("--ceiling", type=int, default=4096)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"],
                    cfg["seed"]["cudnn_benchmark"])
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. This check must run on the P5000.")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    print(f"{props.name}, {props.total_memory // 1024**2} MiB, "
          f"torch {torch.__version__}, T={args.t}")
    print("Forward only, eval mode, no_grad. Batch found forward-only, so it "
          "is not the training ceiling of 162.\n")

    rows, total_per_tubelet = [], 0.0
    for arm in ARMS:
        cap = max_batch(arm, args.t, device, args.ceiling)
        if cap == 0:
            print(f"{arm:<22} does not fit at any batch")
            continue
        sec_iter = time_forward(arm, args.t, cap, device)
        per_tub = sec_iter / cap
        total_per_tubelet += per_tub

        train_iter = TRAIN_SEC_PER_ITER_T8.get(arm)
        train_per_tub = train_iter / TRAIN_BATCH_T8 if train_iter else None
        ratio = per_tub / train_per_tub if train_per_tub else None

        rows.append({
            "arm": arm, "T": args.t,
            "inference_batch": cap,
            "inference_sec_per_iteration": round(sec_iter, 6),
            "inference_sec_per_tubelet": per_tub,
            "training_sec_per_tubelet": train_per_tub,
            "inference_over_training": ratio,
            "batch_gain_over_training": cap / TRAIN_BATCH_T8,
        })
        print(f"{arm:<22} batch {cap:>5}  "
              f"{sec_iter:>8.4f} s/iter  {per_tub:.3e} s/tubelet  "
              f"ratio to training {ratio:.3f}" if ratio else "")

    matrix_per_tubelet = total_per_tubelet * SEEDS
    train_matrix = 0.170   # measured 12 Aug, four arms x three seeds, T=8

    print()
    print(f"WHOLE MATRIX (4 arms x {SEEDS} seeds)")
    print(f"  inference  {matrix_per_tubelet:.4f} s per tubelet")
    print(f"  training   {train_matrix:.4f} s per tubelet per epoch "
          f"(measured 12 Aug)")
    print(f"  MEASURED RATIO {matrix_per_tubelet / train_matrix:.3f}, "
          f"against the ONE-THIRD ASSUMPTION of 0.333")
    print()
    print("RECOMPUTED FULL-VALIDATION COST, whole matrix")
    for frac, n in sorted(VAL_PROPOSALS.items()):
        h = matrix_per_tubelet * n / 3600
        print(f"  min_frac {frac:>9.3g}   {n:>9,} proposals   {h:>7.2f} h")
    print("  (assumed cost at the one-third rule was 10.5 h at 7.23e-06)")
    print()
    print("TEST, ONCE. ESTIMATE ONLY, the test partition has not been run.")
    for frac, n in sorted(TEST_PROPOSALS_ESTIMATE.items()):
        h = matrix_per_tubelet * n / 3600
        print(f"  min_frac {frac:>9.3g}   ~{n:>8,} proposals   {h:>7.2f} h")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    with open(out / "inference_timing.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "inference_timing_meta.json", "w") as fh:
        json.dump({
            "device": props.name,
            "total_memory_mib": props.total_memory // 1024**2,
            "torch": torch.__version__,
            "T": args.t, "n_classes": N_CLASSES, "seeds": SEEDS,
            "matrix_inference_sec_per_tubelet": matrix_per_tubelet,
            "matrix_training_sec_per_tubelet_per_epoch": train_matrix,
            "measured_inference_over_training": matrix_per_tubelet
            / train_matrix,
            "assumed_ratio": 1 / 3,
            "note": "Model definitions imported from tools/epoch_timing.py, "
                    "not rebuilt, so the ratio is against identical models. "
                    "Synthetic tensors, so data loading is excluded and every "
                    "figure is a floor.",
        }, fh, indent=2)
    print(f"\nwritten to {out}/inference_timing.csv and "
          f"{out}/inference_timing_meta.json")


if __name__ == "__main__":
    main()
