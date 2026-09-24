"""Console + JSONL logging. One logger, no framework."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s %(levelname).1s %(name)s | %(message)s"


def get_logger(name: str = "latent_aero_wam") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


class JsonlWriter:
    """Append-only metric log, one JSON object per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=float) + "\n")
