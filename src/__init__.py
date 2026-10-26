"""Product discovery pipeline package. Shared paths and config loading."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml from the project root (or a given path)."""
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Seed Python and numpy RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
