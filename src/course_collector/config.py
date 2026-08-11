from __future__ import annotations

from pathlib import Path
from typing import Any
import json

DEFAULT_CONFIG: dict[str, Any] = {
    "logging": {
        "path": "golf_course_collection.log",
    },
    "checkpoint": {
        "path": ".course_collector/checkpoint.json",
    },
    "targets": {
        "usa": {
            "regions_file": "states_list.txt",
            "data_dir": "data/usa",
            "combined_csv": "data/usa/combined.csv",
            "postal_csv": "data/usa/postal_codes.csv",
            "matched_csv": "data/usa/Fully_Matched_Golf_Courses.csv",
        },
        "world": {
            "regions_file": "states_list.txt",
            "data_dir": "data/world",
            "combined_csv": "data/world/combined.csv",
            "postal_csv": "data/world/postal_codes.csv",
            "matched_csv": "data/world/Fully_Matched_Golf_Courses.csv",
        },
    },
    "inputs": {
        "golflink_csv": "data/golfLinkData.csv",
    },
    "scrape": {
        "golflink": True,
        "golfcanada": False,
        "golfdigest": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Config root must be a mapping in {path}")
    return parsed


def load_config(repo_root: Path, config_path: str | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path:
        cfg = _load_yaml((repo_root / config_path).resolve())
        config = _deep_merge(config, cfg)

    # Lightweight env override for API key location and target selector can be expanded later.
    return config


def dump_default_config_json() -> str:
    return json.dumps(DEFAULT_CONFIG, indent=2)
