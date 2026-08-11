from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}

CANADA_PROVINCE_CODES = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}


@dataclass
class CityAggregate:
    count: int = 0
    lat_sum: float = 0.0
    lon_sum: float = 0.0
    province_counter: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.province_counter is None:
            self.province_counter = Counter()


def _parse_float(value: str | float | int | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _valid_coordinate(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


def _slugify_city_name(name: str) -> str:
    compact = " ".join(str(name or "").strip().split())
    if not compact:
        return ""
    slug = compact.lower().replace(".", "").replace("'", "").replace("-", " ")
    return "_".join(token for token in slug.split() if token)


def _city_label_from_slug(slug: str) -> str:
    if not slug:
        return ""
    parts = slug.split("_")
    if len(parts) >= 2 and len(parts[-1]) <= 3:
        parts = parts[:-1]
    return " ".join(token.capitalize() for token in parts)


def _country_from_province_code(code: str) -> str | None:
    code_norm = (code or "").strip().upper()
    if code_norm in US_STATE_CODES:
        return "United States"
    if code_norm in CANADA_PROVINCE_CODES:
        return "Canada"
    return None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise SystemExit(f"Expected object JSON at {path}")


def _iter_world_courses(combined_csv: Path) -> Iterable[dict]:
    if not combined_csv.exists():
        raise SystemExit(f"World combined CSV not found: {combined_csv}")

    with combined_csv.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            lat = _parse_float(row.get("lat"))
            lon = _parse_float(row.get("lon"))
            if lat is None or lon is None or not _valid_coordinate(lat, lon):
                continue

            gcid = (row.get("gcid") or "").strip() or None
            name = (row.get("name") or "Unnamed Course").strip()
            province = (row.get("province") or "").strip() or None

            yield {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "id": gcid,
                    "name": name,
                    "city_slug": None,
                    "province": province,
                    "country": None,
                    "latitude": lat,
                    "longitude": lon,
                    "holes": None,
                    "course_type": None,
                    "source": "world/combined.csv",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
            }


def _read_matched_city_aggregates(files: Iterable[Path]) -> dict[str, CityAggregate]:
    aggregates: dict[str, CityAggregate] = defaultdict(CityAggregate)

    for path in files:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                city = (row.get("City") or "").strip()
                if not city or city.upper() == "NOMATCH":
                    continue

                lat = _parse_float(row.get("latitude"))
                lon = _parse_float(row.get("longitude"))
                if lat is None or lon is None or not _valid_coordinate(lat, lon):
                    continue

                slug = _slugify_city_name(city)
                region = (row.get("Region") or "").strip().upper()

                entry = aggregates[slug]
                entry.count += 1
                entry.lat_sum += lat
                entry.lon_sum += lon
                if region and region != "NOMATCH":
                    entry.province_counter[region] += 1

    return aggregates


def _build_amenities_by_slug(raw_amenities: dict) -> tuple[dict[str, dict], dict[str, str]]:
    amenities_by_slug: dict[str, dict] = {}
    slug_labels: dict[str, str] = {}

    for city_name, value in raw_amenities.items():
        slug = _slugify_city_name(city_name)
        if not slug:
            continue

        categories = value if isinstance(value, dict) else {}
        total_count = 0
        top_amenity = None
        top_amenity_count = -1

        for amenity_name, payload in categories.items():
            if not isinstance(payload, dict):
                continue
            count = int(_parse_float(payload.get("count")) or 0)
            total_count += count
            if count > top_amenity_count:
                top_amenity = str(amenity_name)
                top_amenity_count = count

        amenities_by_slug[slug] = {
            "city_name": str(city_name),
            "categories": categories,
            "total_amenity_count": total_count,
            "top_amenity": top_amenity,
            "top_amenity_count": max(top_amenity_count, 0),
        }
        slug_labels[slug] = str(city_name)

    return amenities_by_slug, slug_labels


def _build_city_records(
    demographics: dict,
    amenities_by_slug: dict[str, dict],
    slug_labels: dict[str, str],
    aggregates: dict[str, CityAggregate],
) -> list[dict]:
    city_slugs = set(demographics.keys()) | set(amenities_by_slug.keys()) | set(aggregates.keys())
    records: list[dict] = []

    for slug in sorted(city_slugs):
        demo = demographics.get(slug, {})
        amen = amenities_by_slug.get(slug, {})
        agg = aggregates.get(slug)

        city_name = slug_labels.get(slug) or _city_label_from_slug(slug)

        province = None
        slug_parts = slug.split("_")
        if slug_parts and len(slug_parts[-1]) <= 3:
            province = slug_parts[-1].upper()

        if agg and agg.province_counter:
            province = agg.province_counter.most_common(1)[0][0]

        country = _country_from_province_code(province or "")

        lat = None
        lon = None
        golf_count = 0
        if agg and agg.count > 0:
            lat = round(agg.lat_sum / agg.count, 6)
            lon = round(agg.lon_sum / agg.count, 6)
            golf_count = agg.count

        population = _parse_float(demo.get("population_total")) if isinstance(demo, dict) else None
        median_income = _parse_float(demo.get("median_household_income")) if isinstance(demo, dict) else None

        records.append(
            {
                "city_name": city_name,
                "city_slug": slug,
                "province": province,
                "country": country,
                "lat": lat,
                "lon": lon,
                "population": int(population) if population is not None else None,
                "median_household_income": round(median_income, 2) if median_income is not None else None,
                "golf_course_count": golf_count,
                "amenity_total_count": int(amen.get("total_amenity_count", 0)) if isinstance(amen, dict) else 0,
                "top_amenity": amen.get("top_amenity") if isinstance(amen, dict) else None,
                "city_map_path": None,
            }
        )

    return records


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_site_links(repo_root: Path, output_dir: Path) -> None:
    links_path = repo_root / "site_src" / "config" / "site_links.json"
    if links_path.exists():
        payload = json.loads(links_path.read_text(encoding="utf-8"))
    else:
        payload = {"visuals": [], "dataDownloads": []}
    _write_json(output_dir / "site_links.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare site dataset artifacts for Hugging Face and static-site build."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/site_dataset")
    parser.add_argument("--world-combined", default="world/combined.csv")
    parser.add_argument(
        "--matched-files",
        nargs="*",
        default=[
            "usa/Fully_Matched_Golf_Courses.csv",
            "canada/Fully_Matched_Golf_Courses.csv",
            "mexico/Fully_Matched_Golf_Courses.csv",
            "world/Fully_Matched_Golf_Courses.csv",
        ],
    )
    parser.add_argument("--amenities-json", default="city_amenities.json")
    parser.add_argument("--demographics-json", default="city_demographics.json")
    args = parser.parse_args()

    data_root = (ROOT / args.data_root).resolve()
    output_root = (ROOT / args.output).resolve()

    if not data_root.exists() or not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")

    world_combined = data_root / args.world_combined
    matched_files = [data_root / rel for rel in args.matched_files]
    amenities_path = data_root / args.amenities_json
    demographics_path = data_root / args.demographics_json

    features = list(_iter_world_courses(world_combined))
    if not features:
        raise SystemExit("No valid world course points were generated from combined.csv")

    demographics = _read_json(demographics_path)
    raw_amenities = _read_json(amenities_path)
    amenities_by_slug, slug_labels = _build_amenities_by_slug(raw_amenities)
    aggregates = _read_matched_city_aggregates(matched_files)
    city_records = _build_city_records(demographics, amenities_by_slug, slug_labels, aggregates)

    geojson_payload = {"type": "FeatureCollection", "features": features}
    _write_json(output_root / "golf_courses.geojson", geojson_payload)
    _write_json(output_root / "cities.json", city_records)
    _write_json(output_root / "city_demographics.json", demographics)
    _write_json(output_root / "amenities.json", amenities_by_slug)
    _write_json(
        output_root / "derived" / "city_course_counts.json",
        {record["city_slug"]: record["golf_course_count"] for record in city_records},
    )

    _copy_site_links(ROOT, output_root)

    print(f"Output: {output_root}")
    print(f"World course features: {len(features)}")
    print(f"Cities in dataset: {len(city_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
