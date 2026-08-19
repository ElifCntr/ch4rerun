"""Access to the ruled parameter block in configs/ch4.yaml.

Added 19 August 2026 after an independent review found the extraction
constants hardcoded in tools/extract_proposals.py, contradicting the
repository's own rule that ruled parameters live in the config.

THE POINT OF THIS MODULE IS THAT IT HAS NO DEFAULTS. Every accessor raises on
a missing key. A parameter that has not been ruled must stop the run, not
silently acquire a value, because a silent default is exactly how the original
Chapter 4 pipeline ended up with a history of 200 and a min_area of 50 that
nobody had chosen.

Usage:
    from src.config import load_config
    from src.ruled import ruled

    cfg = load_config("configs/ch4.yaml")
    T = ruled(cfg, "window.T")
    hist = ruled(cfg, "bgs.history")
    lr = ruled(cfg, "bgs.learning_rate", allow_null=True)   # null is a value
"""

from __future__ import annotations

from typing import Any

_SENTINEL = object()


class RuledValueMissing(KeyError):
    """A ruled parameter was requested and the config does not carry it."""


def ruled(cfg: dict, key: str, allow_null: bool = False) -> Any:
    """Return cfg['ruled'][<dotted key>].

    Raises RuledValueMissing if any part of the path is absent. A value of
    None raises too unless allow_null is passed, because a null in the config
    is meaningful for exactly two settings (learning_rate and morphology) and
    a typo elsewhere should not read as a deliberate null.
    """
    if "ruled" not in cfg:
        raise RuledValueMissing(
            "configs/ch4.yaml has no 'ruled' block. Ruled parameters live "
            "there; nothing downstream may supply its own."
        )
    node: Any = cfg["ruled"]
    walked: list[str] = []
    for part in key.split("."):
        walked.append(part)
        if not isinstance(node, dict) or part not in node:
            raise RuledValueMissing(
                f"ruled.{'.'.join(walked)} is not in configs/ch4.yaml. "
                f"If this parameter has been ruled, add it; if it has not, "
                f"it cannot be used."
            )
        node = node[part]
    if node is None and not allow_null:
        raise RuledValueMissing(
            f"ruled.{key} is null. Pass allow_null=True if null is the ruled "
            f"value, otherwise the key is unset."
        )
    return node


def require_all(cfg: dict, keys: list[str]) -> dict:
    """Fetch several ruled values at once, reporting EVERY missing key rather
    than stopping at the first. Called at the top of a script so a run fails
    before it does any work, not halfway through a video."""
    got, missing = {}, []
    for k in keys:
        try:
            got[k] = ruled(cfg, k, allow_null=True)
        except RuledValueMissing as e:
            missing.append(f"  {e}")
    if missing:
        raise RuledValueMissing(
            "Missing ruled parameters:\n" + "\n".join(missing))
    return got
