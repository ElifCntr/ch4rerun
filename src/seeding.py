"""Global seeding. Call seed_everything from every entry point.

The original Chapter 4 work set no seeds anywhere, which is why its runs were
not reproducible and produced divergent number sets. Nothing in this project
runs unseeded.
"""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # measurement scripts run without torch installed
    torch = None

_SEEDED: dict[str, int] = {}


def seed_everything(seed: int, deterministic: bool = True,
                    cudnn_benchmark: bool = False) -> int:
    """Seed every generator this project uses. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cudnn.deterministic = bool(deterministic)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)

    _SEEDED["seed"] = int(seed)
    return int(seed)


def assert_seeded() -> int:
    """Fail loudly if an entry point forgot to seed."""
    if "seed" not in _SEEDED:
        raise RuntimeError(
            "seed_everything was never called. Every entry point must seed "
            "before doing any work."
        )
    return _SEEDED["seed"]


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding. Pass as worker_init_fn to every DataLoader."""
    if torch is None:
        return
    info = torch.utils.data.get_worker_info()
    base = int(info.seed) if info is not None else worker_id
    base = base % (2 ** 32 - 1)
    random.seed(base)
    np.random.seed(base)


def make_generator(seed: int):
    """Seeded generator for DataLoader shuffling. Never shuffle unseeded."""
    if torch is None:
        raise RuntimeError("torch is required for make_generator")
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
