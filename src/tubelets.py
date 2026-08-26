"""Dataset over the materialised tubelet store.

The store is a single memory-mapped uint8 array of shape (N, T, 112, 112, 3)
with a manifest whose `seeds` column selects each run's rows. Positives carry
an empty seeds field and belong to every run; a negative belongs to a run only
if that run's seed appears in its list. A proposal drawn by two seeds is stored
once and referenced twice.

AUGMENTATION IS PER TUBELET, NEVER PER FRAME. One flip decision and one
brightness and contrast factor per tubelet, applied identically to all eight
frames. Per-frame draws would manufacture apparent motion, which is the one
thing a study comparing temporal models must not do.

PER-ARM TENSOR LAYOUT. The arms want different shapes and this is the only
place that knows it:
    2d_single           (3, H, W)      the centre frame alone
    2d_tframe_logitavg  (T, 3, H, W)   per-frame 2D, logits averaged
    tsm                 (T, 3, H, W)   as above, with channel shift inside
    r3d_18              (3, T, H, W)   channels first, then time

THE CENTRE FRAME IS INDEX 3. A T=8 window runs centre-3 to centre+4 under the
ruled asymmetric convention, so the annotated frame sits at offset 3, not 4.

NORMALISATION IS NOT YET RULED. torchvision's resnet18 expects ImageNet
statistics and r3d_18 expects Kinetics ones. Each backbone's own pretraining
statistics are passed in by the caller rather than chosen here, and this module
refuses to invent them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CENTRE = 3          # (T - 1) // 2 at T = 8, the ruled asymmetric convention

LAYOUTS = {
    "2d_single": "centre",
    "2d_tframe_logitavg": "tchw",
    "tsm": "tchw",
    "r3d_18": "cthw",
}


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def rows_for_seed(manifest: list[dict], seed: int) -> list[tuple[int, int]]:
    """Return (store_index, label) for one run. Positives are in every run;
    a negative is in this run only if its seeds field names this seed."""
    out = []
    for r in manifest:
        if r["label"] == "positive":
            out.append((int(r["index"]), 1))
        elif str(seed) in (r["seeds"] or "").split("|"):
            out.append((int(r["index"]), 0))
    return out


class TubeletDataset(Dataset):
    """One run's training tubelets.

    mean and std are per-channel sequences in [0, 1] scale, supplied by the
    caller. augment=False gives the deterministic view used for evaluation.
    """

    def __init__(self, store, manifest, seed, arm, mean, std,
                 augment=True, hflip_p=0.5, brightness=None, contrast=None):
        if arm not in LAYOUTS:
            raise ValueError(f"unknown arm '{arm}'")
        if augment and (brightness is None or contrast is None):
            raise ValueError("augment=True needs brightness and contrast "
                             "ranges; this module does not default them")
        self.path = str(store)
        self.arm = arm
        self.layout = LAYOUTS[arm]
        self.augment = augment
        self.hflip_p = hflip_p
        self.brightness = brightness
        self.contrast = contrast
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.rows = rows_for_seed(read_manifest(manifest), seed)
        self._store = None          # opened lazily, per worker

    def __len__(self):
        return len(self.rows)

    @property
    def n_positive(self):
        return sum(1 for _, y in self.rows if y == 1)

    def _array(self):
        # np.load with mmap in each worker; sharing one handle across forked
        # workers is what corrupts reads under num_workers > 0
        if self._store is None:
            self._store = np.load(self.path, mmap_mode="r")
        return self._store

    def __getitem__(self, i):
        idx, label = self.rows[i]
        clip = np.asarray(self._array()[idx], dtype=np.float32) / 255.0

        if self.augment:
            # One draw per tubelet, all frames share it. NOT via a fresh
            # torch.Generator(): that carries a fixed default seed, so
            # torch.rand through it returns the same value every call and the
            # flip becomes a constant rather than a coin toss.
            if np.random.random() < self.hflip_p:
                clip = clip[:, :, ::-1, :]
            b = np.random.uniform(*self.brightness)
            c = np.random.uniform(*self.contrast)
            clip = clip * b
            clip = (clip - clip.mean()) * c + clip.mean()
            clip = np.clip(clip, 0.0, 1.0)

        # (T, H, W, C) -> (T, C, H, W), then normalise per channel
        x = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)
        x = (x - torch.from_numpy(self.mean)) / torch.from_numpy(self.std)

        if self.layout == "centre":
            x = x[CENTRE]
        elif self.layout == "cthw":
            x = x.permute(1, 0, 2, 3)
        # "tchw" is already correct

        return x, label
