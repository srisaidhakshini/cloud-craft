"""YAML config loading. Every tunable lives in the config (FR-13).

Any config other than base.yaml is layered on top of the base.yaml next to it.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text()) or {}
    base = p.with_name("base.yaml")
    if p.name != "base.yaml" and base.exists():
        cfg = deep_update(yaml.safe_load(base.read_text()), cfg)
    return cfg
