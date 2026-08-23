"""Model construction for the classifier matrix, with pretrained weights.

WHY THIS EXISTS SEPARATELY FROM tools/epoch_timing.py. That module builds every
arm with weights=None, which is correct for a timing stub: measuring
per-iteration cost needs the architecture and nothing else, and downloading a
checkpoint to time an iteration would be waste. train_arm.py imported it to
guarantee identical model definitions between the timing coefficients and the
real runs, and inherited weights=None with them. Nine runs trained from scratch
before the fault surfaced.

epoch_timing.py IS NOT EDITED. It is committed evidence for the 0.170 s
coefficient, and rewriting it would make that figure describe models that were
never timed. The chapter states plainly that the coefficient was measured on
randomly initialised models, which changes nothing, since weights do not affect
per-iteration cost.

THE PREFLIGHT GUARD. verify_pretrained() refuses to return a model whose
weights are at initialisation defaults. It checks the exact signature that
caught the fault:

  BATCHNORM RUNNING STATISTICS against init. A freshly constructed BatchNorm
  carries running_mean exactly 0 and running_var exactly 1. A pretrained one
  carries accumulated estimates that are not those. This is what exposed the
  fault: running mean +0.0000 and running var 1.0000, to four decimals.

  FIRST CONVOLUTION CHECKSUM against the checkpoint. The running statistics
  alone would not catch a model built with weights loaded and then
  reinitialised, so the first layer's weights are compared against a
  reference built independently from the same checkpoint.

A failure raises. It does not warn, because a warning is what a tired person
scrolls past at two in the morning.

THE ARM NAMES AND THE SHIFT IMPLEMENTATION MATCH epoch_timing.py EXACTLY, so
the timing coefficients still describe these architectures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import R3D_18_Weights, r3d_18

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from epoch_timing import ARMS, CROP, LogitAverage, ResidualShiftBlock  # noqa

N_CLASSES = 2

# Ruled 22 August. Stated in the chapter as the claim it is.
IMAGENET = ResNet18_Weights.IMAGENET1K_V1
KINETICS = R3D_18_Weights.KINETICS400_V1

WEIGHTS_FOR = {
    "2d_single": ("resnet18", IMAGENET),
    "2d_tframe_logitavg": ("resnet18", IMAGENET),
    "tsm": ("resnet18", IMAGENET),
    "r3d_18": ("r3d_18", KINETICS),
}


class NotPretrained(RuntimeError):
    """A model reached training with weights at initialisation defaults."""


def _backbone(kind, weights):
    return resnet18(weights=weights) if kind == "resnet18" \
        else r3d_18(weights=weights)


def first_conv(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            return m
    raise RuntimeError("no convolution found")


def bn_layers(model):
    return [m for m in model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]


def verify_pretrained(model, arm, verbose=True):
    """Refuse a model whose weights are at initialisation defaults.

    Returns a dict of the evidence, for the run log. Raises NotPretrained on
    failure rather than warning.
    """
    kind, weights = WEIGHTS_FOR[arm]
    bns = bn_layers(model)
    if not bns:
        raise NotPretrained(f"{arm}: no BatchNorm layers to check")

    # 1. running statistics against init defaults
    at_init = 0
    for m in bns:
        if (torch.allclose(m.running_mean, torch.zeros_like(m.running_mean))
                and torch.allclose(m.running_var,
                                   torch.ones_like(m.running_var))):
            at_init += 1

    first = bns[0]
    stats = {
        "bn_layers": len(bns),
        "bn_at_init_defaults": at_init,
        "first_bn_running_mean": float(first.running_mean.mean()),
        "first_bn_running_var": float(first.running_var.mean()),
    }

    # 2. first convolution against an independently built reference
    ref = _backbone(kind, weights)
    got = first_conv(model).weight.detach().cpu()
    want = first_conv(ref).weight.detach().cpu()
    del ref
    matches = got.shape == want.shape and torch.allclose(got, want, atol=1e-6)
    stats["first_conv_checksum"] = float(got.double().abs().sum())
    stats["first_conv_matches_checkpoint"] = bool(matches)
    stats["weights"] = str(weights)

    if verbose:
        print(f"  preflight {arm}: {len(bns)} BatchNorm layers, "
              f"{at_init} at init defaults; first BN mean "
              f"{stats['first_bn_running_mean']:+.6f} var "
              f"{stats['first_bn_running_var']:.6f}; first conv checksum "
              f"{stats['first_conv_checksum']:.4f}, matches checkpoint "
              f"{matches}")

    if at_init == len(bns):
        raise NotPretrained(
            f"{arm}: every one of {len(bns)} BatchNorm layers carries "
            f"running_mean 0 and running_var 1, which are initialisation "
            f"defaults. This model is NOT pretrained. Nine runs were lost to "
            f"exactly this in August 2026.")
    if not matches:
        raise NotPretrained(
            f"{arm}: the first convolution does not match {weights}. The "
            f"model may have been built pretrained and then reinitialised.")
    return stats


def build_arm(name: str, t: int, verify: bool = True):
    """Return (model, input shape without batch). Mirrors epoch_timing's
    build_arm exactly, except that weights are loaded and checked."""
    if name not in WEIGHTS_FOR:
        raise ValueError(f"unknown arm '{name}'; expected one of {ARMS}")
    kind, weights = WEIGHTS_FOR[name]

    if name == "2d_single":
        m = _backbone(kind, weights)
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
        model, shape = m, (3, CROP, CROP)

    elif name == "2d_tframe_logitavg":
        b = _backbone(kind, weights)
        b.fc = nn.Linear(b.fc.in_features, N_CLASSES)
        model, shape = LogitAverage(b, t), (t, 3, CROP, CROP)

    elif name == "tsm":
        b = _backbone(kind, weights)
        b.fc = nn.Linear(b.fc.in_features, N_CLASSES)
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(b, layer_name)
            for i, block in enumerate(layer):
                layer[i] = ResidualShiftBlock(block, n_segment=t)
        model, shape = LogitAverage(b, t), (t, 3, CROP, CROP)

    else:  # r3d_18
        m = _backbone(kind, weights)
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
        model, shape = m, (3, t, CROP, CROP)

    if verify:
        # the classification head is replaced above and is deliberately NOT
        # pretrained; the guard checks the backbone, which is what transfers
        verify_pretrained(model, name)
    return model, shape
