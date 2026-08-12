"""Epoch-timing check. Run on the P5000.

What this measures:
  1. Parameter counts for all four arms with a 2-class head, replacing the
     400-way-head counts recorded on 10 August.
  2. The largest batch size that fits the 3D arm at each candidate T, which is
     the ruled way of setting the batch size for every arm.
  3. Wall-clock seconds per training iteration (forward, backward, step) for
     each arm at that batch size.

What this CANNOT give, and why:
  Epoch time is iterations x seconds-per-iteration. The iteration count is
  ceil(training tubelets / batch), and the tubelet count does not exist until
  proposals are extracted, which is Stage 1b and is HELD. This script therefore
  reports seconds per iteration and leaves epoch time as a multiplication to be
  done once the tubelet count is known. No tubelet count is assumed or
  estimated here.

Timing runs on synthetic tensors of the correct shape. That is deliberate: it
isolates model cost from data-loading cost, and it needs no extraction. Data
loading will add to the real figure, so treat every number here as a floor.

Nothing in this script chooses a value. T is swept over the candidate set the
strip measurements already use.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.video import r3d_18

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

CROP = 112          # ruled, r3d_18's Kinetics weights were trained at 112
N_CLASSES = 2       # ruled, binary drone against background
T_CANDIDATES = (3, 5, 8, 16)


# ------------------------------------------------------------------- arms ---

class ResidualShiftBlock(nn.Module):
    """Residual temporal shift wrapped around a torchvision BasicBlock.

    TIMING STUB. Its only job is to make the TSM arm's per-iteration cost
    representative. The Stage 2 arm is written fresh against Lin et al. and
    reviewed; nothing here is promoted to it.

    Implements the ruled configuration, because the fraction changes the
    timing materially. Lin et al. report full shift at roughly 12.4 per cent
    GPU latency overhead against about 3 per cent at one eighth, so a stub
    with the wrong fraction gives an unrepresentative number:

      - one quarter of channels shifted in total, one eighth in each direction
      - RESIDUAL placement, so the shift applies inside the residual branch
        only and the identity path carries unshifted activations
      - inserted at every residual block

    The residual placement is the part a naive wrapper gets wrong. Calling
    block(shift(x)) shifts the identity path too, which is in-place shift, not
    residual shift, and it has a different memory-traffic profile.
    """

    def __init__(self, block: nn.Module, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.block = block
        self.n_segment = n_segment
        self.fold_div = fold_div      # 1/8 each direction, 1/4 in total

    def shift(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                   # shift forward
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]   # shift backward
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]              # unshifted
        return out.view(nt, c, h, w)

    def forward(self, x):
        b = self.block
        identity = x                       # unshifted, this is the point
        out = b.conv1(self.shift(x))
        out = b.relu(b.bn1(out))
        out = b.bn2(b.conv2(out))
        if b.downsample is not None:
            identity = b.downsample(x)
        return b.relu(out + identity)


class LogitAverage(nn.Module):
    """TSN / f-R2D style. Per-frame 2D forward, logits averaged over T."""

    def __init__(self, backbone: nn.Module, n_segment: int):
        super().__init__()
        self.backbone = backbone
        self.n_segment = n_segment

    def forward(self, x):                # x: (B, T, 3, H, W)
        b, t = x.shape[:2]
        logits = self.backbone(x.flatten(0, 1))
        return logits.view(b, t, -1).mean(dim=1)


def build_arm(name: str, t: int) -> tuple[nn.Module, tuple]:
    """Return (model, input shape without the batch dimension)."""
    if name == "2d_single":
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
        return m, (3, CROP, CROP)

    if name == "2d_tframe_logitavg":
        b = resnet18(weights=None)
        b.fc = nn.Linear(b.fc.in_features, N_CLASSES)
        return LogitAverage(b, t), (t, 3, CROP, CROP)

    if name == "tsm":
        b = resnet18(weights=None)
        b.fc = nn.Linear(b.fc.in_features, N_CLASSES)
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(b, layer_name)
            for i, block in enumerate(layer):
                layer[i] = ResidualShiftBlock(block, n_segment=t)
        return LogitAverage(b, t), (t, 3, CROP, CROP)

    if name == "r3d_18":
        m = r3d_18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
        return m, (3, t, CROP, CROP)

    raise ValueError(name)


ARMS = ("2d_single", "2d_tframe_logitavg", "tsm", "r3d_18")


# ---------------------------------------------------------------- timing ----

def one_iteration(model, x, y, opt, crit) -> None:
    opt.zero_grad(set_to_none=True)
    loss = crit(model(x), y)
    loss.backward()
    opt.step()


def fits(arm: str, t: int, batch: int, device) -> bool:
    try:
        model, shape = build_arm(arm, t)
        model = model.to(device).train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(batch, *shape, device=device)
        y = torch.randint(0, N_CLASSES, (batch,), device=device)
        for _ in range(2):
            one_iteration(model, x, y, opt, crit)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        torch.cuda.empty_cache()


def max_batch(arm: str, t: int, device, ceiling: int = 512) -> int:
    """Largest power of two that fits, then a linear refine upward."""
    best = 0
    b = 1
    while b <= ceiling and fits(arm, t, b, device):
        best = b
        b *= 2
    if best == 0:
        return 0
    b = best + max(best // 8, 1)
    while b <= ceiling and fits(arm, t, b, device):
        best = b
        b += max(best // 8, 1)
    return best


def time_iterations(arm: str, t: int, batch: int, device,
                    warmup: int = 5, iters: int = 20) -> float:
    model, shape = build_arm(arm, t)
    model = model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, *shape, device=device)
    y = torch.randint(0, N_CLASSES, (batch,), device=device)

    for _ in range(warmup):
        one_iteration(model, x, y, opt, crit)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        one_iteration(model, x, y, opt, crit)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    del model, opt, x, y
    torch.cuda.empty_cache()
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--t", type=int, nargs="*", default=list(T_CANDIDATES))
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"],
                    cfg["seed"]["cudnn_benchmark"])

    if not torch.cuda.is_available():
        sys.exit("No CUDA device. This check must run on the P5000.")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    params = {}
    for arm in ARMS:
        model, _ = build_arm(arm, max(args.t))
        params[arm] = sum(p.numel() for p in model.parameters())
        del model

    rows = []
    for t in args.t:
        cap = max_batch("r3d_18", t, device)
        if cap == 0:
            rows.append({"T": t, "batch": 0, "arm": "r3d_18",
                         "seconds_per_iteration": None,
                         "note": "nothing fits, not even batch 1"})
            continue
        for arm in ARMS:
            try:
                secs = time_iterations(arm, t, cap, device)
                note = ""
            except torch.cuda.OutOfMemoryError:
                secs, note = None, "OOM at the 3D-determined batch size"
            rows.append({"T": t, "batch": cap, "arm": arm,
                         "seconds_per_iteration":
                             round(secs, 5) if secs else None,
                         "note": note})
            print(f"T={t:2d} batch={cap:3d} {arm:20s} "
                  f"{secs if secs else 'OOM'}")

    out_dir = repo_path(cfg, cfg["reports"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "epoch_timing.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["T", "batch", "arm",
                                           "seconds_per_iteration", "note"])
        w.writeheader()
        w.writerows(rows)

    meta = {
        "device": props.name,
        "total_memory_MiB": props.total_memory // (1024 ** 2),
        "torch": torch.__version__,
        "crop": CROP,
        "n_classes": N_CLASSES,
        "parameter_counts_2class_head": params,
        "batch_rule": "largest that fits r3d_18, then applied to every arm",
        "synthetic_data": True,
        "epoch_time": "NOT COMPUTED. Requires the training tubelet count, "
                      "which does not exist until Stage 1b extraction. "
                      "epoch_seconds = ceil(tubelets / batch) x "
                      "seconds_per_iteration, plus data loading.",
    }
    (out_dir / "epoch_timing_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWritten to {out_dir}/epoch_timing.csv and epoch_timing_meta.json")
    print("Epoch time is not computed here. See the meta file for why.")


if __name__ == "__main__":
    main()
