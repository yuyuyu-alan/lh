"""Configuration helpers for loading YAML files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def _validate_mapping(config: Any, source: Path) -> Dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, dict):
        message = f"Expected a YAML mapping in {source}, got {type(config).__name__}."
        logger.error(message)
        raise ValueError(message)
    return config


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    source = Path(path)
    if not source.exists():
        message = f"Config file not found: {source}"
        logger.error(message)
        raise FileNotFoundError(message)

    logger.info("Loading YAML config from %s", source)
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return _validate_mapping(config, source)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_merge(default_path: str | Path, override_path: str | Path) -> Dict[str, Any]:
    """Load default and override YAML files, returning a merged dictionary."""
    defaults = load_yaml(default_path)
    overrides = load_yaml(override_path)
    logger.info("Merging YAML config %s over %s", override_path, default_path)
    return _merge_dicts(defaults, overrides)
