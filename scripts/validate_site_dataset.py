from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "golf_courses.geojson",
    "cities.json",
    "city_demographics.json",
    "amenities.json",
    "derived/city_course_counts.json",
    "site_links.json",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_coordinates(features: list[dict]) -> list[str]:
    errors: list[str] = []
    for idx, feature in enumerate(features):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            errors.append(f"features[{idx}] missing point coordinates")
            continue

        lon = coordinates[0]
        lat = coordinates[1]
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            errors.append(f"features[{idx}] non-numeric coordinates")
            continue

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            errors.append(f"features[{idx}] out-of-range coordinates: ({lat}, {lon})")

    return errors


def validate_dataset(dataset_root: Path, min_cities: int) -> None:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        path = dataset_root / rel
        if not path.exists():
            errors.append(f"Missing required file: {rel}")

    if errors:
        raise SystemExit("\n".join(errors))

    geojson = _load_json(dataset_root / "golf_courses.geojson")
    if geojson.get("type") != "FeatureCollection":
        errors.append("golf_courses.geojson must be a FeatureCollection")

    features = geojson.get("features")
    if not isinstance(features, list) or not features:
        errors.append("golf_courses.geojson must include non-empty features")
    else:
        errors.extend(_check_coordinates(features))

    cities = _load_json(dataset_root / "cities.json")
    if not isinstance(cities, list):
        errors.append("cities.json must be a JSON array")
    else:
        if len(cities) < min_cities:
            errors.append(f"cities.json has {len(cities)} cities; minimum required is {min_cities}")
        for idx, city in enumerate(cities):
            if not isinstance(city, dict):
                errors.append(f"cities[{idx}] must be an object")
                continue
            for field in ("city_name", "city_slug", "golf_course_count"):
                if field not in city:
                    errors.append(f"cities[{idx}] missing field: {field}")

    demographics = _load_json(dataset_root / "city_demographics.json")
    if not isinstance(demographics, dict):
        errors.append("city_demographics.json must be a JSON object")

    amenities = _load_json(dataset_root / "amenities.json")
    if not isinstance(amenities, dict):
        errors.append("amenities.json must be a JSON object")

    course_counts = _load_json(dataset_root / "derived/city_course_counts.json")
    if not isinstance(course_counts, dict):
        errors.append("derived/city_course_counts.json must be a JSON object")

    links = _load_json(dataset_root / "site_links.json")
    if not isinstance(links, dict):
        errors.append("site_links.json must be a JSON object")

    if errors:
        raise SystemExit("\n".join(errors))

    print("Dataset validation passed")
    print(f"Dataset root: {dataset_root}")
    print(f"World features: {len(features)}")
    print(f"Cities: {len(cities)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate static-site dataset artifacts.")
    parser.add_argument("--dataset-root", default="data/site_dataset")
    parser.add_argument("--min-cities", type=int, default=10)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    validate_dataset(dataset_root=dataset_root, min_cities=args.min_cities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
