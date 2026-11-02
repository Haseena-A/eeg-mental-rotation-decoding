"""Load the YAML configuration and resolve paths from the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read a configuration file. Relative paths stay relative to the project root."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration at {config_path} is not a mapping")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve(config: dict[str, Any], key: str) -> Path:
    """Turn a path entry such as ``paths.figures_dir`` into an absolute path."""
    section, name = key.split(".", 1)
    relative = Path(config[section][name])
    if relative.is_absolute():
        return relative
    return Path(config["_project_root"]) / relative
