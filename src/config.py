"""Config loading. The data path lives in YAML, never in code."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def data_root(cfg: dict) -> Path:
    root = cfg["data"]["root"]
    if not root:
        raise ValueError("data.root is not set in the config.")
    return Path(os.path.expanduser(root)).resolve()


def repo_path(cfg: dict, rel: str | None) -> Path | None:
    if rel is None:
        return None
    p = Path(os.path.expanduser(rel))
    return p if p.is_absolute() else (REPO_ROOT / p)


def require(cfg: dict, dotted: str) -> Any:
    """Fetch a config value, refusing None.

    Fields the layout report fills in start as null so that no downstream step
    can run on a guess about the dataset layout.
    """
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    if node is None:
        raise ValueError(
            f"Config field '{dotted}' is null. Run "
            f"tools/inventory.py --stage layout, fill it in from the report, "
            f"commit the change, then rerun."
        )
    return node
