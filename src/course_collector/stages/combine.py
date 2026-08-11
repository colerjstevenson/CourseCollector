from __future__ import annotations

from pathlib import Path
import importlib


def run_combine(data_dir: Path, combined_csv: Path) -> dict:
    cleaner = importlib.import_module("legacy.cleaner")
    combined_geojson = combined_csv.with_suffix(".geojson")

    cleaner.combine_csvs(data_dir, combined_csv)
    cleaner.combine_geojsons(data_dir, combined_geojson)

    return {
        "combined_csv": str(combined_csv),
        "combined_geojson": str(combined_geojson),
    }
