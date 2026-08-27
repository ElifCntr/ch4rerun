#!/usr/bin/env python3
"""
Why does r3d_18 sit at chance while 2d_single does not?

THE OBSERVATION. All three r3d_18 seeds finished epoch 1 with subset AP of
0.0084, 0.0051 and 0.0104 against roughly 0.006 for random ranking, and loss
0.576, 0.581, 0.581 against about 0.636 for predicting the class prior. Three
seeds agreeing that closely is systematic rather than stochastic. On identical
data 2d_single reached loss 0.328 and AP 0.617 in its first epoch.

THE HYPOTHESIS, and it is a hypothesis. BatchNorm running statistics are
frozen by ruling. r3d_18 carries KINETICS statistics and sees upscaled drone
crops against sky, so the activation distributions are far from what those
frozen estimates expect, and the network cannot adapt them, only fight them
through the weights at a base learning rate of 0.001. The 2D arms tolerate the
same freeze because ImageNet statistics sit closer to these crops.

WHAT THIS RUNS. Three short arms over the same data, one epoch each, differing
only in how BatchNorm is handled. Everything else comes from the ruled block.

  frozen      running statistics held, exactly as ruled. The control.
  unfrozen    statistics update during training.
  reset       statistics reinitialised, then updated. Distinguishes "the
              Kinetics estimates are wrong for this data" from "the layer
              needs to adapt at all".

IT ALSO REPORTS THE ACTIVATION STATISTICS the first BatchNorm layer actually
sees against the ones it carries. If those disagree by a wide margin the
mechanism is visible directly rather than inferred from the AP.

THIS IS A DIAGNOSTIC, NOT A PROTOCOL CHANGE. It writes nothing that any run
consumes and changes no ruled value. Unfreezing in the matrix would be a
protocol change and needs a ruling, which is what this exists to inform.

Usage:
    python tools/diagnose_r3d_bn.py --epochs 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from epoch_timing import build_arm  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ruled import ruled  # noqa: E402
from src.seeding import seed_everything  # noqa: E402
from src.tubelets import TubeletDataset  # noqa: E402

ARM = "r3d_18"


def bn_modules(model):
    return [m for m in model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]


def set_bn(model, mode):
    bns = bn_modules(model)
    if mode == "frozen":
        for m in bns:
            m.eval()
    elif mode == "reset":
        for m in bns:
            m.reset_running_stats()
            m.train()
    elif mode == "unfrozen":
        for m in bns:
            m.train()
    else:
        raise ValueError(mode)
    return len(bns)


def observed_stats(model, loader, device, n_batches=8):
    """Mean and variance the first BatchNorm actually receives, against the
    running estimates it carries. A wide gap is the mechanism made visible."""
    first = bn_modules(model)[0]
    seen = []

    def hook(_m, inp, _out):
        x = inp[0].detach()
        dims = [0] + list(range(2, x.dim()))
        seen.append((x.mean(dim=dims).cpu().numpy(),
                     x.var(dim=dims).cpu().numpy()))

    h = first.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x.to(device))
            if i + 1 >= n_batches:
                break
    h.remove()
    obs_m = np.mean([s[0] for s in seen], axis=0)
    obs_v = np.mean([s[1] for s in seen], axis=0)
    return {
        "observed_mean": float(obs_m.mean()),
        "observed_var": float(obs_v.mean()),
        "running_mean": float(first.running_mean.mean().cpu()),
        "running_var": float(first.running_var.mean().cpu()),
        "channels": int(first.num_features),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--store", default="data/tubelets/tubelets.npy")
    ap.add_argument("--manifest", default="data/tubelets/manifest.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--modes", default="frozen,unfrozen,reset")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    batch = args.batch or ruled(cfg, "window.batch")
    mean = ruled(cfg, f"normalisation.{ARM}.mean")
    std = ruled(cfg, f"normalisation.{ARM}.std")
    T = ruled(cfg, "window.T")

    if not torch.cuda.is_available():
        sys.exit("No CUDA device.")
    device = torch.device("cuda")

    results = {}
    for mode in [m.strip() for m in args.modes.split(",")]:
        seed_everything(args.seed, True, False)   # identical init every mode
        ds = TubeletDataset(
            args.store, args.manifest, args.seed, ARM, mean, std,
            augment=True,
            hflip_p=ruled(cfg, "augmentation.hflip_p"),
            brightness=tuple(ruled(cfg, "augmentation.brightness")),
            contrast=tuple(ruled(cfg, "augmentation.contrast")))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch, shuffle=True, num_workers=4,
            pin_memory=True)

        model, _ = build_arm(ARM, T)
        model = model.to(device)
        n_bn = set_bn(model, mode)

        stats = observed_stats(model, loader, device) if mode == "frozen" \
            else None

        opt = torch.optim.SGD(
            model.parameters(), lr=ruled(cfg, "training.base_lr"),
            momentum=ruled(cfg, "training.momentum"),
            weight_decay=ruled(cfg, "training.weight_decay"))
        crit = nn.CrossEntropyLoss()

        losses = []
        for ep in range(args.epochs):
            model.train()
            set_bn(model, mode if mode != "reset" else "unfrozen")
            tot, n, t0 = 0.0, 0, time.time()
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                tot += float(loss) * y.numel()
                n += y.numel()
            losses.append(tot / max(n, 1))
            print(f"  {mode:<9} epoch {ep + 1}  loss {losses[-1]:.4f}  "
                  f"{time.time() - t0:.0f}s")
        results[mode] = {"losses": losses, "bn_layers": n_bn,
                         "first_bn_stats": stats}
        del model, opt, loader, ds
        torch.cuda.empty_cache()

    chance = -np.log(1.0 / 3.0) * (1 / 3) - np.log(2.0 / 3.0) * (2 / 3)
    L = ["r3d_18 BATCHNORM DIAGNOSTIC",
         f"seed {args.seed}, {args.epochs} epoch(s), batch {batch}, "
         "everything else from the ruled block.",
         f"Loss for a model predicting the class prior is about "
         f"{chance:.4f}.",
         "For reference the matrix runs gave loss 0.576, 0.581 and 0.581 with "
         "subset AP 0.0084, 0.0051 and 0.0104, against chance AP of roughly "
         "0.006.",
         ""]
    for mode, r in results.items():
        L.append(f"  {mode:<9} losses {['%.4f' % x for x in r['losses']]}  "
                 f"({r['bn_layers']} BatchNorm layers)")
    L.append("")
    fs = results.get("frozen", {}).get("first_bn_stats")
    if fs:
        L.append("FIRST BATCHNORM, what it carries against what it receives")
        L.append(f"  running mean {fs['running_mean']:+.4f}   observed mean "
                 f"{fs['observed_mean']:+.4f}")
        L.append(f"  running var  {fs['running_var']:.4f}   observed var  "
                 f"{fs['observed_var']:.4f}")
        L.append(f"  averaged over {fs['channels']} channels")
        L.append("  A wide gap means the frozen estimates do not describe "
                 "this data, which is the mechanism the hypothesis names.")
    L.append("")
    L.append("READING. If unfrozen and reset fall well below frozen, the "
             "freeze is the cause and unfreezing for this arm becomes a "
             "decision with a measured basis. If all three sit together near "
             "the prior loss, the cause is elsewhere and this rules the "
             "freeze out.")

    txt = "\n".join(L)
    Path(args.out).mkdir(exist_ok=True)
    p = Path(args.out) / "r3d_bn_diagnostic.txt"
    p.write_text(txt)
    with open(Path(args.out) / "r3d_bn_diagnostic.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print()
    print(txt)
    print(f"\nwritten to {p}")


if __name__ == "__main__":
    main()
