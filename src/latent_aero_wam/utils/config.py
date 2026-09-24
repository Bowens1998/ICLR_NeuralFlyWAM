"""Minimal YAML config loading with ``defaults`` composition and CLI overrides.

One config system, no framework. A config may declare ``defaults: [path, ...]``
whose entries are merged (deep, left to right) underneath it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _expand(value: Any) -> Any:
    """Expand ``${ENV_VAR}`` and ``${ENV_VAR:-default}`` inside strings."""
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str) and "${" in value:
        out, i = [], 0
        while i < len(value):
            if value.startswith("${", i):
                j = value.index("}", i)
                body = value[i + 2 : j]
                name, _, default = body.partition(":-")
                out.append(os.environ.get(name, default))
                i = j + 1
            else:
                out.append(value[i])
                i += 1
        return "".join(out)
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, resolving ``defaults`` and ``a.b.c=value`` overrides."""
    path = Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path) if (REPO_ROOT / path).exists() else path.resolve()
    raw = yaml.safe_load(path.read_text()) or {}

    merged: dict = {}
    for dep in raw.pop("defaults", []) or []:
        merged = _deep_merge(merged, load_config(dep))
    merged = _deep_merge(merged, raw)

    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = merged
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)

    return _expand(merged)


def config_hash(cfg: dict) -> str:
    """Stable short hash of a resolved config, used to bind runs to configs."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def resolve_path(p: str | Path) -> Path:
    """Resolve a possibly repo-relative path to an absolute Path."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)
