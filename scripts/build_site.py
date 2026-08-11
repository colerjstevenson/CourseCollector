from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_SRC = ROOT / "docs"
MAPS_OUT = ROOT / "maps"


def _iter_combined_csv_files(data_root: Path) -> list[Path]:
    csv_files = sorted(data_root.glob("*/combined.csv"))
    if not csv_files:
        raise SystemExit(f"No combined.csv files found under {data_root}")
    return csv_files


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_coordinate(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


def _read_course_points(csv_files: list[Path]) -> tuple[list[dict], Counter]:
    features: list[dict] = []
    source_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for csv_path in csv_files:
        source_region = csv_path.parent.name
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                gcid = (row.get("gcid") or "").strip()
                name = (row.get("name") or "Unnamed Course").strip()
                province = (row.get("province") or source_region).strip()
                lat = _parse_float(row.get("lat") or "")
                lon = _parse_float(row.get("lon") or "")

                if lat is None or lon is None or not _valid_coordinate(lat, lon):
                    continue

                dedupe_key = gcid or f"{name.lower()}|{lat:.6f}|{lon:.6f}"
                if dedupe_key in seen_ids:
                    continue
                seen_ids.add(dedupe_key)

                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat],
                        },
                        "properties": {
                            "id": gcid or None,
                            "name": name,
                            "province": province,
                            "source_region": source_region,
                        },
                    }
                )
                source_counts[source_region] += 1

    return features, source_counts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_templates(templates_dir: Path, destination_dir: Path) -> None:
    for html_file in templates_dir.glob("*.html"):
        shutil.copy2(html_file, destination_dir / html_file.name)


def _sync_dataset_files(dataset_root: Path, output_data_dir: Path) -> None:
    wanted_rel_paths = [
        "golf_courses.geojson",
        "cities.json",
        "city_demographics.json",
        "amenities.json",
        "site_links.json",
        "derived/city_course_counts.json",
        "manifest.json",
    ]

    for rel_path in wanted_rel_paths:
        source = dataset_root / rel_path
        if not source.exists():
            continue
        destination = output_data_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _compute_summary_from_geojson(geojson_payload: dict, cities_payload: list[dict]) -> dict:
    features = geojson_payload.get("features") or []
    source_counts: Counter[str] = Counter()

    for feature in features:
        properties = feature.get("properties") or {}
        source_region = (
            properties.get("province")
            or properties.get("source_region")
            or properties.get("country")
            or "Unknown"
        )
        source_counts[str(source_region)] += 1

    top_source_regions = [
        {"name": name, "count": count}
        for name, count in source_counts.most_common(12)
    ]

    total_population = 0
    for city in cities_payload:
        population = city.get("population")
        if isinstance(population, int):
            total_population += population

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_courses": len(features),
        "source_region_count": len(source_counts),
        "source_regions": dict(source_counts),
        "top_source_regions": top_source_regions,
        "total_cities": len(cities_payload),
        "population_covered": total_population,
    }


def _build_from_dataset(dataset_root: Path, output_dir: Path) -> None:
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    required = [
        dataset_root / "golf_courses.geojson",
        dataset_root / "cities.json",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Required dataset file not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_out = output_dir / "data"
    assets_out = output_dir / "assets"

    if data_out.exists():
        shutil.rmtree(data_out)
    data_out.mkdir(parents=True, exist_ok=True)

    _copy_tree(SITE_SRC / "assets", assets_out)
    _copy_templates(SITE_SRC / "templates", output_dir)
    _sync_dataset_files(dataset_root, data_out)

    geojson_payload = _read_json(data_out / "golf_courses.geojson")
    cities_payload = _read_json(data_out / "cities.json")
    summary_payload = _compute_summary_from_geojson(geojson_payload, cities_payload)
    _write_json(data_out / "landing_summary.json", summary_payload)

    if not (data_out / "site_links.json").exists():
        _write_json(data_out / "site_links.json", {"visuals": [], "dataDownloads": []})

    print(f"Output directory: {output_dir}")
    print(f"Dataset source: {dataset_root}")
    print(f"Course features written: {summary_payload['total_courses']}")
    print(f"Cities written: {summary_payload['total_cities']}")


def build_site(data_root: Path, output_dir: Path) -> None:
    csv_files = _iter_combined_csv_files(data_root)
    features, source_counts = _read_course_points(csv_files)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_out = output_dir / "data"
    assets_out = output_dir / "assets"

    _copy_tree(SITE_SRC / "assets", assets_out)
    _copy_templates(SITE_SRC / "templates", output_dir)

    geojson_payload = {
        "type": "FeatureCollection",
        "features": features,
    }
    _write_json(data_out / "golf_courses.geojson", geojson_payload)

    top_source_regions = [
        {"name": name, "count": count}
        for name, count in source_counts.most_common(12)
    ]

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_courses": len(features),
        "source_region_count": len(source_counts),
        "source_regions": dict(source_counts),
        "top_source_regions": top_source_regions,
    }
    _write_json(data_out / "landing_summary.json", summary_payload)

    links_path = SITE_SRC / "config" / "site_links.json"
    if links_path.exists():
        shutil.copy2(links_path, data_out / "site_links.json")
    else:
        _write_json(data_out / "site_links.json", {"visuals": [], "dataDownloads": []})

    print(f"Output directory: {output_dir}")
    print(f"Course features written: {len(features)}")
    print(f"Source region files used: {len(csv_files)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static landing page and world map artifacts.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", default="maps")
    args = parser.parse_args()

    output_dir = (ROOT / args.output).resolve()

    if args.dataset_root:
        dataset_root = (ROOT / args.dataset_root).resolve()
        _build_from_dataset(dataset_root=dataset_root, output_dir=output_dir)
        return 0

    data_root = (ROOT / args.data_root).resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")

    build_site(data_root=data_root, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
