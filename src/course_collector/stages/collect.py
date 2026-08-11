from __future__ import annotations

from pathlib import Path
from typing import Iterable
import importlib


def _read_regions(regions_file: Path) -> list[str]:
    lines = regions_file.read_text(encoding="utf-8-sig").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _run_state_list(module_name: str, regions: Iterable[str]) -> dict:
    mod = importlib.import_module(module_name)
    if hasattr(mod, "MAKE_MAP"):
        mod.MAKE_MAP = False

    processed = 0
    skipped = 0

    for region in regions:
        slug = region.lower().replace(" ", "_")
        data_dir = Path("data/usa") if module_name.endswith("golfSearchUSA") else Path("data/world")
        geo = data_dir / f"golf_courses_{slug}.geojson"
        csv = data_dir / f"golf_courses_{slug}.csv"

        if geo.exists() and csv.exists():
            skipped += 1
            print(f"Skipping existing region: {region}")
            continue

        if hasattr(mod, "check_memory_usage") and not mod.check_memory_usage():
            print(f"Skipping {region} due to memory check failure")
            skipped += 1
            continue

        mod.run_states(region)
        processed += 1

        if hasattr(mod, "cleanup_memory"):
            mod.cleanup_memory()

    return {"processed": processed, "skipped": skipped}


def run_collect(target: str, regions_file: Path) -> dict:
    regions = _read_regions(regions_file)
    if target == "usa":
        return _run_state_list("legacy.golfSearchUSA", regions)
    if target == "world":
        return _run_state_list("legacy.golfSearchWorld", regions)
    raise ValueError(f"Unsupported target for collect stage: {target}")
