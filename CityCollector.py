"""
Standalone city data collector pipeline.

This script orchestrates golf-course normalization, amenities collection,
demographics collection, and census cache generation.

Golf handling in this version is file-system based: it reads existing golf data
from the data folder and writes normalized outputs, without scraping network data.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_CENSUS_TRACTS_CACHE: Dict[str, Any] = {}
_ENGLISH_CSV_CACHE: Dict[str, Path] = {}
_DGUID_MAP_CACHE: Dict[str, Dict[str, str]] = {}


DEFAULT_CANADIAN_CITIES: List[Tuple[str, str]] = [
    ("Toronto", "Ontario"),
    ("Montreal", "Quebec"),
    ("Vancouver", "British Columbia"),
    ("Calgary", "Alberta"),
    ("Edmonton", "Alberta"),
    ("Ottawa", "Ontario"),
    ("Winnipeg", "Manitoba"),
]

DEFAULT_US_CITIES: List[Tuple[str, str]] = [
    ("New York", "New York"),
    ("Los Angeles", "California"),
    ("Chicago", "Illinois"),
    ("Houston", "Texas"),
    ("Phoenix", "Arizona"),
    ("Philadelphia", "Pennsylvania"),
    ("San Antonio", "Texas"),
    ("San Diego", "California"),
    ("Palm Springs", "California"),
    ("Scottsdale", "Arizona"),
    ("Dallas", "Texas"),
    ("San Jose", "California"),
    ("Austin", "Texas"),
    ("Jacksonville", "Florida"),
    ("Fort Worth", "Texas"),
    ("Columbus", "Ohio"),
    ("Indianapolis", "Indiana"),
    ("Charlotte", "North Carolina"),
    ("San Francisco", "California"),
    ("Seattle", "Washington"),
    ("Denver", "Colorado"),
    ("Washington", "District of Columbia"),
    ("Boston", "Massachusetts"),
    ("Nashville", "Tennessee"),
    ("Detroit", "Michigan"),
    ("Portland", "Oregon"),
    ("Las Vegas", "Nevada"),
]

DEFAULT_AMENITY_TYPES: List[str] = [
    "parks",
    "pools",
    "hockey rinks",
    "golf courses",
    "playgrounds",
    "sports centres",
    "basketball courts",
    "tennis courts",
    "soccer fields",
    "baseball fields",
    "libraries",
    "Schools",
    "hospitals",
]


DEFAULT_GOLF_SOURCE_GLOBS: List[str] = [
    "canada/Fully_Matched_Golf_Courses.csv",
    "canada/golf_canada_full.csv",
    "canada/golf_canada_full.json",
    "canada/golf_canada_data*.csv",
    "canada/golf_canada_data*.json",
    "canada/golf_courses_*.csv",
    "canada/combined.csv",
    "usa/Fully_Matched_Golf_Courses.csv",
    "mexico/Fully_Matched_Golf_Courses.csv",
    "world/combined.csv",
    "golfLinkData.csv",
    "golfLinkData.json",
]


DEFAULT_CONFIG: Dict[str, Any] = {
    "sources": {
        "golf": True,
        "amenities": True,
        "demographics": True,
        "census_cache": True,
    },
    "cities": {
        "regions": ["canada", "us"],
        "canada_pool": DEFAULT_CANADIAN_CITIES,
        "us_pool": DEFAULT_US_CITIES,
        "list": [],
        "include": [],
        "exclude": [],
    },
    "amenities": {
        "types": DEFAULT_AMENITY_TYPES,
    },
    "outputs": {
        "data_root": "data",
        "amenities_json": "data/city_amenities.json",
        "demographics_json": "data/city_demographics.json",
        "golf_json": "data/canada/golf_canada_full.json",
        "golf_csv": "data/canada/golf_canada_full.csv",
        "legacy_golf_json": "data/golf_canada_data_full.json",
        "legacy_golf_csv": "data/golf_canada_data_full.csv",
        "emit_legacy_golf_files": False,
        "golf_hf_repo_id": "colerjstevenson/GolfGulf",
        "golf_hf_repo_type": "dataset",
        "golf_hf_subdir": "",
        "golf_hf_token": None,
        "golf_source_globs": DEFAULT_GOLF_SOURCE_GLOBS,
        "census_root": "data/censusShape",
    },
    "behavior": {
        "dry_run": False,
        "force_refresh": False,
        "skip_existing": True,
        "verbose": False,
        "progress_interval": 25,
        "amenity_delay_seconds": 0.0,
        "amenity_save_every": 25,
        "amenity_buffer_km": 0.0,
        "small_test_limit": None,
        "golf_use_hf": True,
        "golf_file_limit": None,
        "max_rows_per_source_file": None,
        "census_cache_provider": "legacy",
    },
    "validation": {
        "enabled": True,
        "expected_files": [],
        "min_amenity_types_per_city": 1,
        "min_demographic_cities": 1,
        "min_golf_records": 1,
    },
}


@dataclass
class CityTarget:
    city: str
    region: str
    country: str


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if not path:
        return config

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from exc
        with open(file_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

    return _deep_merge(config, user_cfg)


def _fallback_province_slug(province: str) -> str:
    mapping = {
        "Ontario": "on",
        "Quebec": "qc",
        "British Columbia": "bc",
        "Alberta": "ab",
        "Manitoba": "mb",
        "Saskatchewan": "sk",
        "Nova Scotia": "ns",
        "New Brunswick": "nb",
        "Newfoundland and Labrador": "nl",
        "Prince Edward Island": "pe",
    }
    return mapping.get(province, province.lower().replace(" ", "_"))


def _fallback_resolve_state_abbr(state_name_or_abbr: str) -> Optional[str]:
    if not state_name_or_abbr:
        return None

    if len(state_name_or_abbr.strip()) == 2:
        return state_name_or_abbr.strip().upper()

    mapping = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
        "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
        "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
        "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
        "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
        "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
        "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
        "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
        "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
        "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
        "district of columbia": "DC", "washington, d.c.": "DC", "washington dc": "DC",
    }
    return mapping.get(state_name_or_abbr.strip().lower())


def _parse_city_spec(item: str, canadian_regions: set[str]) -> CityTarget:
    parts = [p.strip() for p in item.split(",")]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid city spec '{item}'. Use 'City,Region' or 'City,Region,Country'."
        )

    city = parts[0]
    region = parts[1]
    if len(parts) >= 3:
        country = parts[2]
    else:
        country = "Canada" if region in canadian_regions else "United States"

    return CityTarget(city=city, region=region, country=country)


def _default_city_targets(regions: Iterable[str], canada_pool: List[Tuple[str, str]], us_pool: List[Tuple[str, str]]) -> List[CityTarget]:
    out: List[CityTarget] = []
    region_set = {r.strip().lower() for r in regions}

    if "canada" in region_set:
        for city, province in canada_pool:
            out.append(CityTarget(city=city, region=province, country="Canada"))

    if "us" in region_set or "united states" in region_set:
        for city, state in us_pool:
            out.append(CityTarget(city=city, region=state, country="United States"))

    return out


def _apply_city_filters(
    cities: List[CityTarget], include: Iterable[str], exclude: Iterable[str]
) -> List[CityTarget]:
    include_terms = [x.strip().lower() for x in include if x and x.strip()]
    exclude_terms = [x.strip().lower() for x in exclude if x and x.strip()]

    def _matches_any(target: CityTarget, terms: List[str]) -> bool:
        blob = f"{target.city} {target.region} {target.country}".lower()
        return any(t in blob for t in terms)

    filtered = cities
    if include_terms:
        filtered = [c for c in filtered if _matches_any(c, include_terms)]
    if exclude_terms:
        filtered = [c for c in filtered if not _matches_any(c, exclude_terms)]
    return filtered


def _build_city_targets(config: Dict[str, Any]) -> List[CityTarget]:
    city_cfg = config["cities"]
    canada_pool = [tuple(x) for x in city_cfg.get("canada_pool", DEFAULT_CANADIAN_CITIES)]
    us_pool = [tuple(x) for x in city_cfg.get("us_pool", DEFAULT_US_CITIES)]
    canadian_regions = {region for _, region in canada_pool}
    explicit = city_cfg.get("list") or []

    if explicit:
        targets = [_parse_city_spec(item, canadian_regions) for item in explicit]
    else:
        targets = _default_city_targets(city_cfg.get("regions", ["canada", "us"]), canada_pool, us_pool)

    targets = _apply_city_filters(
        targets,
        city_cfg.get("include", []),
        city_cfg.get("exclude", []),
    )

    unique: List[CityTarget] = []
    seen = set()
    for target in targets:
        key = (target.city.lower(), target.region.lower(), target.country.lower())
        if key not in seen:
            seen.add(key)
            unique.append(target)
    return unique


def _slug_city(city: str) -> str:
    return city.strip().replace(" ", "_").lower()


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_json_list_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_csv_from_records(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _expand_source_globs(data_root: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend([p for p in data_root.glob(pattern) if p.is_file()])

    deduped: List[Path] = []
    seen = set()
    for p in files:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.replace("\xa0", " ").split()).strip()
    return value


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        norm_key = str(key).strip()
        out[norm_key] = _normalize_scalar(value)

    # Standard aliases for common fields.
    if "name" not in out and "Name" in out:
        out["name"] = out["Name"]
    if "url" not in out and "URL" in out:
        out["url"] = out["URL"]
    if "city" not in out and "City" in out:
        out["city"] = out["City"]
    if "address" not in out and "Address" in out:
        out["address"] = out["Address"]

    return out


def _record_key(record: Dict[str, Any]) -> str:
    for key in ["url", "URL", "gcid", "GCID"]:
        value = str(record.get(key, "")).strip()
        if value:
            return f"{key}:{value.lower()}"

    name = str(record.get("name", record.get("Name", ""))).strip().lower()
    address = str(record.get("address", record.get("Address", ""))).strip().lower()
    city = str(record.get("city", record.get("City", ""))).strip().lower()
    if name or address or city:
        return f"nac:{name}|{address}|{city}"

    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def _records_from_json(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        if payload.get("type") == "FeatureCollection" and isinstance(payload.get("features"), list):
            out = []
            for feature in payload["features"]:
                if not isinstance(feature, dict):
                    continue
                props = feature.get("properties") or {}
                if isinstance(props, dict):
                    out.append(props)
            return out
        return [payload]

    return []


def _records_from_csv(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    rows.append(dict(row))
                    if isinstance(max_rows, int) and max_rows > 0 and (idx + 1) >= max_rows:
                        return rows
            return rows
        except UnicodeDecodeError:
            continue
        except Exception:
            return []
    return rows


def _trim_leading_data(path_str: str) -> str:
    normalized = path_str.replace("\\", "/").lstrip("/")
    if normalized.startswith("data/"):
        return normalized[5:]
    return normalized


def _download_hf_golf_sources(
    repo_id: str,
    repo_type: str,
    token: Optional[str],
    patterns: Sequence[str],
    local_dir: Path,
    subdir: str = "",
) -> List[Path]:
    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for golf_use_hf. Install with: pip install huggingface_hub"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    subdir_norm = _trim_leading_data(subdir).strip("/")
    normalized_patterns = [_trim_leading_data(p) for p in patterns]

    matches: List[str] = []
    for full_path in repo_files:
        rel = _trim_leading_data(full_path)
        if subdir_norm:
            if rel == subdir_norm:
                rel = ""
            elif rel.startswith(subdir_norm + "/"):
                rel = rel[len(subdir_norm) + 1 :]
            else:
                continue
        if not rel:
            continue
        if any(fnmatch.fnmatch(rel, pat) for pat in normalized_patterns):
            matches.append(full_path)

    downloaded: List[Path] = []
    for filename in matches:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            token=token,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        downloaded.append(Path(local_path))

    # Deduplicate while preserving order.
    deduped: List[Path] = []
    seen = set()
    for p in downloaded:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


ACS5_VARS: Dict[str, str] = {
    "population_total": "B01003_001E",
    "median_age": "B01002_001E",
    "housing_units": "B25001_001E",
    "median_home_value": "B25077_001E",
    "median_gross_rent": "B25064_001E",
    "owner_occupied_units": "B25003_002E",
    "renter_occupied_units": "B25003_003E",
    "median_household_income": "B19013_001E",
    "per_capita_income": "B19301_001E",
    "poverty_rate_total": "B17001_002E",
    "poverty_population_total": "B17001_001E",
    "civilian_labor_force": "B23025_003E",
    "employed_total": "B23025_004E",
    "unemployed_total": "B23025_005E",
    "bachelors_or_higher_25_plus": "B06009_005E",
    "graduate_or_professional_25_plus": "B06009_006E",
    "white_alone": "B02001_002E",
    "black_or_african_american_alone": "B02001_003E",
    "asian_alone": "B02001_005E",
    "two_or_more_races": "B02001_008E",
    "hispanic_or_latino_any_race": "B03003_003E",
    "not_hispanic_or_latino": "B03003_002E",
    "foreign_born": "B05002_013E",
    "native_born": "B05002_002E",
    "renter_gross_rent_35_plus_income": "B25070_010E",
}


US_STATE_FIPS: Dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


DEFAULT_CANCENSUS_API_KEY = "CensusMapper_d918ab7e2b0cb08ac7b24a3990a6cb93"


def _aggregate_acs_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not rows:
        return out

    pop_weights: List[float] = []
    for row in rows:
        try:
            pop_weights.append(float(row.get("B01003_001E", 0) or 0))
        except Exception:
            pop_weights.append(0.0)

    for key, var in ACS5_VARS.items():
        values: List[float] = []
        for row in rows:
            raw = row.get(var)
            try:
                values.append(float(raw))
            except Exception:
                continue
        if not values:
            continue

        if key.startswith("median_"):
            weighted_sum = 0.0
            total_weight = 0.0
            for idx, row in enumerate(rows):
                raw = row.get(var)
                try:
                    val = float(raw)
                except Exception:
                    continue
                weight = pop_weights[idx] if idx < len(pop_weights) else 0.0
                weighted_sum += val * weight
                total_weight += weight
            if total_weight > 0:
                out[key] = weighted_sum / total_weight
            else:
                out[key] = sum(values) / len(values)
        else:
            out[key] = float(sum(values))

    lf = out.get("civilian_labor_force")
    unemp = out.get("unemployed_total")
    if isinstance(lf, (int, float)) and lf > 0 and isinstance(unemp, (int, float)):
        out["unemployment_rate"] = float(unemp) / float(lf)

    pov_num = out.get("poverty_rate_total")
    pov_den = out.get("poverty_population_total")
    if isinstance(pov_num, (int, float)) and isinstance(pov_den, (int, float)) and pov_den > 0:
        out["poverty_rate"] = float(pov_num) / float(pov_den)

    owners = out.get("owner_occupied_units")
    renters = out.get("renter_occupied_units")
    if isinstance(owners, (int, float)) or isinstance(renters, (int, float)):
        total_occ = float(owners or 0) + float(renters or 0)
        if total_occ > 0:
            out["owner_share"] = float(owners or 0) / total_occ
            out["renter_share"] = float(renters or 0) / total_occ

    pop_total = out.get("population_total")
    if isinstance(pop_total, (int, float)) and pop_total > 0:
        for race_key in [
            "white_alone",
            "black_or_african_american_alone",
            "asian_alone",
            "two_or_more_races",
            "hispanic_or_latino_any_race",
            "not_hispanic_or_latino",
        ]:
            val = out.get(race_key)
            if isinstance(val, (int, float)):
                out[f"{race_key}_share"] = float(val) / float(pop_total)

    return out


def _collect_demographics_us(city: str, state_abbr: str, year: int = 2022) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise RuntimeError("requests is required for US demographics. Install with: pip install requests") from exc

    api_key = os.environ.get("CENSUS_API_KEY")

    fips = US_STATE_FIPS.get(state_abbr.upper())
    if not fips:
        raise RuntimeError(f"No FIPS mapping for state abbreviation: {state_abbr}")

    variables = list(ACS5_VARS.values())
    endpoint = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(variables),
        "for": "place:*",
        "in": f"state:{fips}",
    }
    if api_key:
        params["key"] = api_key

    response = requests.get(endpoint, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected ACS response format for {city}, {state_abbr}")

    headers = payload[0]
    rows = payload[1:]
    structured: List[Dict[str, Any]] = [dict(zip(headers, row)) for row in rows]
    filtered = [r for r in structured if city.lower() in str(r.get("NAME", "")).lower()]
    if not filtered:
        raise RuntimeError(f"No ACS place rows matched city '{city}' in {state_abbr}")

    out = _aggregate_acs_rows(filtered)
    out["city"] = city
    out["state"] = state_abbr
    out["country"] = "United States"
    out["year"] = year
    out["data_source"] = f"US Census ACS5 {year}"
    return out


def _collect_demographics_canada(city: str, province: str, year: int = 2021) -> Dict[str, Any]:
    del year
    try:
        import pycancensus as pc  # type: ignore
    except Exception as exc:
        raise RuntimeError("pycancensus is required for Canadian demographics. Install with: pip install pycancensus") from exc

    api_key = os.environ.get("CANCENSUS_API_KEY") or DEFAULT_CANCENSUS_API_KEY
    os.environ["CANCENSUS_API_KEY"] = api_key

    cma_mapping = {
        "Toronto": "Toronto",
        "Vancouver": "Vancouver",
        "Montreal": "Montreal",
        "Calgary": "Calgary",
        "Edmonton": "Edmonton",
        "Ottawa": "Ottawa-Gatineau",
        "Winnipeg": "Winnipeg",
        "Quebec City": "Quebec",
        "Hamilton": "Hamilton",
        "London": "London",
        "Kitchener": "Kitchener-Cambridge-Waterloo",
        "St. Catharines": "St. Catharines-Niagara",
    }
    cma_name = cma_mapping.get(city, city)

    regions_df = pc.search_census_regions(cma_name, dataset="CA21")
    if regions_df.empty:
        raise RuntimeError(f"Could not find census region for '{cma_name}'")

    region_id = regions_df.iloc[0]["region"]
    vectors = [
        "v_CA21_1",
        "v_CA21_389",
        "v_CA21_4237",
        "v_CA21_4238",
        "v_CA21_4239",
        "v_CA21_575",
        "v_CA21_5799",
        "v_CA21_4404",
        "v_CA21_4405",
        "v_CA21_4406",
        "v_CA21_4872",
        "v_CA21_4873",
        "v_CA21_4874",
        "v_CA21_4875",
        "v_CA21_4876",
    ]
    census_df = pc.get_census(
        dataset="CA21",
        regions={"CMA": [region_id]},
        vectors=vectors,
        use_cache=True,
    )

    vector_mapping = {
        "v_CA21_1": "population_total",
        "v_CA21_389": "median_age",
        "v_CA21_4237": "total_households",
        "v_CA21_4238": "owner_occupied_units",
        "v_CA21_4239": "renter_occupied_units",
        "v_CA21_575": "employed_total",
        "v_CA21_5799": "high_school",
        "v_CA21_4404": "total_immigrant_status",
        "v_CA21_4405": "native_born",
        "v_CA21_4406": "foreign_born",
        "v_CA21_4872": "total_visible_minority",
        "v_CA21_4873": "white_alone",
        "v_CA21_4874": "south_asian",
        "v_CA21_4875": "chinese",
        "v_CA21_4876": "black_or_african_american_alone",
    }

    out: Dict[str, Any] = {}
    for col in census_df.columns:
        vec_id = col.split(":")[0].strip() if ":" in col else col
        field_name = vector_mapping.get(vec_id)
        if not field_name:
            continue
        raw = census_df[col].iloc[0]
        try:
            out[field_name] = float(raw)
        except Exception:
            continue

    if not out:
        raise RuntimeError(f"No Canadian census values extracted for {city}, {province}")

    owners = out.get("owner_occupied_units")
    renters = out.get("renter_occupied_units")
    if isinstance(owners, (int, float)) and isinstance(renters, (int, float)):
        total_occ = owners + renters
        if total_occ > 0:
            out["owner_share"] = owners / total_occ
            out["renter_share"] = renters / total_occ

    pop_total = out.get("population_total")
    if isinstance(pop_total, (int, float)) and pop_total > 0:
        for race_key in ["white_alone", "black_or_african_american_alone", "south_asian", "chinese"]:
            race_val = out.get(race_key)
            if isinstance(race_val, (int, float)):
                out[f"{race_key}_share"] = race_val / pop_total

        foreign_born = out.get("foreign_born")
        if isinstance(foreign_born, (int, float)):
            out["foreign_born_share"] = foreign_born / pop_total

    out["city"] = city
    out["province"] = province
    out["country"] = "Canada"
    out["year"] = 2021
    out["data_source"] = "Statistics Canada 2021 via pycancensus"
    return out


def _load_city_boundary(city: str, province: str):
    try:
        import osmnx as ox  # type: ignore
    except Exception as exc:
        raise RuntimeError("osmnx is required for census cache build. Install with: pip install osmnx") from exc

    query = f"{city}, {province}, Canada"
    gdf = ox.geocode_to_gdf(query)
    if gdf is None or len(gdf) == 0:
        raise RuntimeError(f"City boundary not found for '{query}'")
    return gdf.to_crs("EPSG:3347")


def _load_census_tracts(shapefile_dir: str):
    try:
        import geopandas as gpd  # type: ignore
    except Exception as exc:
        raise RuntimeError("geopandas is required for census cache build. Install with: pip install geopandas") from exc

    cache_key = str(Path(shapefile_dir).resolve())
    cached = _CENSUS_TRACTS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    shp_files = [f for f in os.listdir(shapefile_dir) if f.lower().endswith(".shp")]
    if not shp_files:
        raise RuntimeError(f"No shapefile found in {shapefile_dir}")
    shp_path = str(Path(shapefile_dir) / shp_files[0])
    gdf = gpd.read_file(shp_path).to_crs("EPSG:3347")
    if "CTUID" not in gdf.columns:
        raise RuntimeError("Shapefile missing CTUID column")
    gdf["CTUID"] = gdf["CTUID"].astype(str)
    _CENSUS_TRACTS_CACHE[cache_key] = gdf
    return gdf


def _find_english_csv(data_dir: str) -> Path:
    cache_key = str(Path(data_dir).resolve())
    cached = _ENGLISH_CSV_CACHE.get(cache_key)
    if cached is not None and cached.exists():
        return cached

    csv_files = [Path(data_dir) / f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    for p in csv_files:
        if "english_csv_data" in p.name.lower():
            _ENGLISH_CSV_CACHE[cache_key] = p
            return p
    raise FileNotFoundError("Could not find English census profile CSV in census root")


def _build_dguid_to_ctuid_map(data_dir: str, ct_set: set[str]) -> Dict[str, str]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for census cache build. Install with: pip install pandas") from exc

    cache_key = str(Path(data_dir).resolve())
    full_map = _DGUID_MAP_CACHE.get(cache_key)

    if full_map is None:
        geo_map_path = None
        for f in os.listdir(data_dir):
            if f.lower().endswith(".csv") and "geo" in f.lower():
                geo_map_path = Path(data_dir) / f
                break
        if geo_map_path is None:
            return {}

        geo_df = None
        for enc in ["utf-8", "cp1252", "latin1"]:
            try:
                geo_df = pd.read_csv(geo_map_path, low_memory=False, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if geo_df is None or "Geo Code" not in geo_df.columns or "Geo Name" not in geo_df.columns:
            return {}

        geo_df["Geo Code"] = geo_df["Geo Code"].astype(str)
        geo_df["Geo Name"] = geo_df["Geo Name"].astype(str)
        full_map = dict(zip(geo_df["Geo Code"], geo_df["Geo Name"]))
        _DGUID_MAP_CACHE[cache_key] = full_map

    if not ct_set:
        return dict(full_map)

    return {dguid: ctuid for dguid, ctuid in full_map.items() if ctuid in ct_set}


def _normalize_chunk_ctuid(chunk: Any, ct_set: set[str], dguid_to_ctuid: Dict[str, str]):
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for census cache build. Install with: pip install pandas") from exc

    id_col = None
    for cand in ["CTUID", "GEO_CODE (POR)", "DGUID"]:
        if cand in chunk.columns:
            id_col = cand
            break
    if id_col is None:
        return pd.DataFrame()

    chunk = chunk.copy()
    chunk[id_col] = chunk[id_col].astype(str)
    if id_col == "DGUID" and dguid_to_ctuid:
        chunk["CTUID"] = chunk[id_col].map(dguid_to_ctuid)
        chunk = chunk[chunk["CTUID"].notna()]
    elif id_col != "CTUID":
        chunk = chunk.rename(columns={id_col: "CTUID"})

    if "CTUID" in chunk.columns:
        chunk["CTUID"] = chunk["CTUID"].astype(str)
        if ct_set:
            chunk = chunk[chunk["CTUID"].isin(ct_set)]
    return chunk


def _iter_filtered_city_chunks(data_dir: str, ct_set: set[str], chunk_size: int = 100000):
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for census cache build. Install with: pip install pandas") from exc

    english_csv = _find_english_csv(data_dir)
    dguid_to_ctuid = _build_dguid_to_ctuid_map(data_dir, ct_set)

    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            for chunk in pd.read_csv(english_csv, encoding=enc, low_memory=False, chunksize=chunk_size):
                filtered = _normalize_chunk_ctuid(chunk, ct_set, dguid_to_ctuid)
                if not filtered.empty:
                    yield filtered
            break
        except UnicodeDecodeError:
            continue


def _build_filtered_csv_cache(data_dir: str, clipped: Any, city_slug: str) -> str:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for census cache build. Install with: pip install pandas") from exc

    target_dir = Path(data_dir) / city_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    ct_set = set(clipped["CTUID"].astype(str).tolist())
    english_csv = _find_english_csv(data_dir)
    dguid_to_ctuid = _build_dguid_to_ctuid_map(data_dir, ct_set)
    out_path = target_dir / f"{city_slug}_data.csv"

    if not out_path.exists():
        first_chunk = True
        for enc in ["utf-8", "cp1252", "latin1"]:
            try:
                for chunk in pd.read_csv(english_csv, encoding=enc, low_memory=False, chunksize=100000):
                    chunk = _normalize_chunk_ctuid(chunk, ct_set, dguid_to_ctuid)
                    if chunk.empty:
                        continue
                    mode = "w" if first_chunk else "a"
                    header = first_chunk
                    chunk.to_csv(out_path, index=False, mode=mode, header=header)
                    first_chunk = False
                break
            except UnicodeDecodeError:
                continue

    return str(target_dir)


def _detect_value_col(df: Any) -> Optional[str]:
    for cand in ["VALUE", "C1_COUNT_TOTAL", "C10_RATE_TOTAL", "C11_RATE_MEN+", "C12_RATE_WOMEN+", "C2_COUNT_MEN+", "C3_COUNT_WOMEN+"]:
        if cand in df.columns:
            return cand

    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    for col in df.columns:
        if col in ("CTUID", "DGUID", "CHARACTERISTIC_NAME", "DIMENSION", "MEMBER_ID"):
            continue
        try:
            pd.to_numeric(df[col])
            return str(col)
        except Exception:
            continue
    return None


def _process_rows_to_cache(rows: Any, cache: Dict[str, Any], value_col: str):
    if rows.empty or value_col not in rows.columns:
        return

    work = rows[rows["CTUID"].notna()].copy()
    if work.empty:
        return

    member_label_col = None
    for cand in ["MEMBER", "MEMBER_LABEL", "MEMBER_NAME", "Member", "Member Label"]:
        if cand in work.columns:
            member_label_col = cand
            break

    dim_col = "DIMENSION" if "DIMENSION" in work.columns else None
    if "CHARACTERISTIC_ID" in work.columns:
        work.sort_values(["CTUID", "CHARACTERISTIC_ID"], inplace=True)
    else:
        work.sort_values(["CTUID"], inplace=True)

    current_category_by_ct: Dict[str, str] = {}
    for _, row in work.iterrows():
        ctuid = str(row["CTUID"]).strip()
        raw_char = str(row.get("CHARACTERISTIC_NAME", ""))
        leading_spaces = len(raw_char) - len(raw_char.lstrip(" "))
        char_name = raw_char.strip()

        if dim_col:
            category = str(row[dim_col]).strip() or char_name
        else:
            if char_name.startswith("Total - ") and leading_spaces == 0:
                current_category_by_ct[ctuid] = char_name
                category = char_name
            else:
                category = current_category_by_ct.get(ctuid)
                if not category:
                    parts = char_name.split(" - ")
                    category = parts[0] if parts and parts[0] else (char_name or "Other")

        metric = str(row.get(member_label_col, "")).strip() if member_label_col else ""
        if not metric:
            if char_name.startswith("Total - "):
                metric = "Total"
            else:
                if current_category_by_ct.get(ctuid) and leading_spaces > 2:
                    continue
                metric = char_name

        raw_value = row[value_col]
        try:
            value = float(raw_value)
        except Exception:
            try:
                value = float(str(raw_value).replace(",", ""))
            except Exception:
                continue

        ct_entry = cache.setdefault(ctuid, {})
        cat_entry = ct_entry.setdefault(category, {})
        cat_entry[metric] = value


def _build_census_cache(
    data_root: str,
    city: str,
    province: str,
    overwrite: bool = False,
    single_pass: bool = True,
    remove_intermediate_csv: bool = True,
    write_compat_profile_name: bool = True,
):
    del single_pass
    try:
        import geopandas as gpd  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pandas and geopandas are required for census cache build. Install with: pip install pandas geopandas"
        ) from exc

    city_slug = _slug_city(city)
    city_dir = Path(data_root) / city_slug
    city_dir.mkdir(parents=True, exist_ok=True)

    profile_file = city_dir / f"{city_slug}_profile_cache.json"
    profile_compat = city_dir / "profile_cache.json"
    if (profile_file.exists() or profile_compat.exists()) and not overwrite:
        return str(city_dir)

    tracts = _load_census_tracts(data_root)
    boundary = _load_city_boundary(city, province)
    city_poly = boundary.iloc[0].geometry
    clipped = gpd.clip(tracts, city_poly)
    clipped["CTUID"] = clipped["CTUID"].astype(str)
    ct_set = set(clipped["CTUID"].tolist())
    if not ct_set:
        raise RuntimeError(f"No census tracts intersect city boundary for {city}, {province}")

    english_csv = _find_english_csv(data_root)
    dguid_to_ctuid = _build_dguid_to_ctuid_map(data_root, ct_set)
    cache: Dict[str, Any] = {}
    value_col = None
    rows_seen = 0

    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            for chunk in pd.read_csv(english_csv, encoding=enc, low_memory=False, chunksize=100000):
                filtered = _normalize_chunk_ctuid(chunk, ct_set, dguid_to_ctuid)
                if filtered.empty:
                    continue
                rows_seen += len(filtered)
                if value_col is None:
                    value_col = _detect_value_col(filtered)
                    if value_col is None:
                        continue
                _process_rows_to_cache(filtered, cache, value_col)
            break
        except UnicodeDecodeError:
            continue

    if rows_seen == 0 or not cache:
        raise RuntimeError(f"No matching census profile rows found for {city}, {province}")

    _save_json(profile_file, cache)
    if write_compat_profile_name:
        _save_json(profile_compat, cache)

    if remove_intermediate_csv:
        intermediate_csv = city_dir / f"{city_slug}_data.csv"
        if intermediate_csv.exists():
            try:
                intermediate_csv.unlink()
            except OSError:
                pass
    return str(city_dir)


def _build_census_cache_legacy(
    data_root: str,
    city: str,
    province: str,
    overwrite: bool = False,
    single_pass: bool = True,
    remove_intermediate_csv: bool = True,
    write_compat_profile_name: bool = True,
):
    try:
        import geopandas as gpd  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pandas and geopandas are required for census cache build. Install with: pip install pandas geopandas"
        ) from exc

    city_slug = _slug_city(city)
    city_dir = Path(data_root) / city_slug
    city_dir.mkdir(parents=True, exist_ok=True)

    profile_file = city_dir / f"{city_slug}_profile_cache.json"
    profile_compat = city_dir / "profile_cache.json"
    if (profile_file.exists() or profile_compat.exists()) and not overwrite:
        return str(city_dir)

    tracts = _load_census_tracts(data_root)
    boundary = _load_city_boundary(city, province)
    city_poly = boundary.iloc[0].geometry
    clipped = gpd.clip(tracts, city_poly)
    clipped["CTUID"] = clipped["CTUID"].astype(str)

    if single_pass:
        ct_set = set(clipped["CTUID"].tolist())
        cache: Dict[str, Any] = {}
        value_col = None
        chunks_seen = 0

        for chunk in _iter_filtered_city_chunks(data_root, ct_set):
            chunks_seen += 1
            if value_col is None:
                value_col = _detect_value_col(chunk)
                if value_col is None:
                    continue
            _process_rows_to_cache(chunk, cache, value_col)

        if chunks_seen == 0:
            raise RuntimeError(f"No matching census rows found for {city_slug}.")
        if not cache:
            raise RuntimeError(f"No cache records built for {city_slug}.")

        _save_json(profile_file, cache)
    else:
        _build_filtered_csv_cache(data_root, clipped, city_slug)
        data_csv = city_dir / f"{city_slug}_data.csv"
        if not data_csv.exists():
            raise FileNotFoundError(f"Cached data CSV not found: {data_csv}")

        try:
            df = pd.read_csv(data_csv, encoding="cp1252", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(data_csv, encoding="latin1", low_memory=False)

        if "CTUID" not in df.columns:
            raise ValueError("Filtered city CSV missing CTUID after processing; expected CTUID column.")

        value_col = _detect_value_col(df)
        if value_col is None:
            raise ValueError("No usable numeric value column found in cached CSV")

        cache: Dict[str, Any] = {}
        _process_rows_to_cache(df, cache, value_col)
        _save_json(profile_file, cache)

    if write_compat_profile_name:
        _save_json(profile_compat, _load_json_if_exists(profile_file))

    if remove_intermediate_csv:
        intermediate_csv = city_dir / f"{city_slug}_data.csv"
        if intermediate_csv.exists():
            try:
                intermediate_csv.unlink()
            except OSError:
                pass

    return str(city_dir)


class _BuiltinAmenityCounter:
    def __init__(self, output_json: str = "city_amenities.json"):
        self.output_path = Path(output_json)
        self.data = _load_json_if_exists(self.output_path)
        self._boundary_cache: Dict[Tuple[str, str, str, float], Any] = {}

        self._ox = None
        self._geo_import_error = None
        try:
            import osmnx as ox  # type: ignore
            self._ox = ox
        except Exception as exc:
            self._geo_import_error = exc

    def get_city_data(self, city: str) -> Optional[Dict[str, Any]]:
        data = self.data.get(city)
        return data if isinstance(data, dict) else None

    def _save_data(self):
        _save_json(self.output_path, self.data)

    def _get_osm_tags(self, location_type: str) -> Dict[str, str]:
        location_type_lower = location_type.lower()
        tag_mappings = {
            "parks": {"leisure": "park"},
            "park": {"leisure": "park"},
            "pools": {"leisure": "swimming_pool"},
            "pool": {"leisure": "swimming_pool"},
            "swimming pools": {"leisure": "swimming_pool"},
            "hockey rinks": {"sport": "ice_hockey"},
            "hockey rink": {"sport": "ice_hockey"},
            "ice hockey": {"sport": "ice_hockey"},
            "ice rinks": {"sport": "ice_hockey"},
            "ice rink": {"sport": "ice_hockey"},
            "golf courses": {"leisure": "golf_course"},
            "golf course": {"leisure": "golf_course"},
            "playgrounds": {"leisure": "playground"},
            "playground": {"leisure": "playground"},
            "sports centres": {"leisure": "sports_centre"},
            "sports center": {"leisure": "sports_centre"},
            "sports centres": {"leisure": "sports_centre"},
            "gyms": {"leisure": "fitness_centre"},
            "gym": {"leisure": "fitness_centre"},
            "fitness centres": {"leisure": "fitness_centre"},
            "basketball courts": {"sport": "basketball"},
            "basketball": {"sport": "basketball"},
            "tennis courts": {"sport": "tennis"},
            "tennis": {"sport": "tennis"},
            "soccer fields": {"sport": "soccer"},
            "soccer": {"sport": "soccer"},
            "baseball fields": {"sport": "baseball"},
            "baseball": {"sport": "baseball"},
            "stadiums": {"leisure": "stadium"},
            "stadium": {"leisure": "stadium"},
            "libraries": {"amenity": "library"},
            "library": {"amenity": "library"},
            "schools": {"amenity": "school"},
            "school": {"amenity": "school"},
            "hospitals": {"amenity": "hospital"},
            "hospital": {"amenity": "hospital"},
        }
        return tag_mappings.get(location_type_lower, {"leisure": location_type_lower.replace(" ", "_")})

    def get_city_polygon(
        self,
        city: str,
        province: Optional[str] = None,
        country: str = "Canada",
        buffer_km: float = 0.0,
    ):
        if province:
            query = f"{city}, {province}, {country}"
        else:
            query = f"{city}, {country}"

        cache_key = (
            city.strip().lower(),
            (province or "").strip().lower(),
            country.strip().lower(),
            float(buffer_km),
        )
        if cache_key in self._boundary_cache:
            return self._boundary_cache[cache_key]

        boundary = self._ox.geocode_to_gdf(query)
        if boundary is None or len(boundary) == 0:
            return None

        geometry = boundary.geometry
        if buffer_km > 0:
            metric = boundary.to_crs(epsg=3857)
            geometry = metric.geometry.buffer(buffer_km * 1000.0).to_crs(epsg=4326)

        polygon = geometry.iloc[0]
        self._boundary_cache[cache_key] = polygon
        return polygon

    def count_amenities(
        self,
        city: str,
        location_type: str,
        province: Optional[str] = None,
        country: str = "Canada",
        min_area_m2: float = 0.0,
        buffer_km: float = 0.0,
        city_polygon: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if self._ox is None:
            raise RuntimeError(
                "osmnx is not installed; install osmnx/geopandas to collect amenities "
                "or disable amenities source"
            ) from self._geo_import_error

        polygon = city_polygon
        if polygon is None:
            polygon = self.get_city_polygon(
                city=city,
                province=province,
                country=country,
                buffer_km=buffer_km,
            )

        if polygon is None:
            return {"count": 0, "total_area_m2": 0.0}
        tags = self._get_osm_tags(location_type)

        try:
            try:
                gdf = self._ox.features.features_from_polygon(polygon, tags)
            except AttributeError:
                gdf = self._ox.geometries_from_polygon(polygon, tags)
        except Exception:
            return {"count": 0, "total_area_m2": 0.0}

        if gdf.empty:
            return {"count": 0, "total_area_m2": 0.0}

        gdf_projected = gdf.to_crs(epsg=3347)
        gdf_projected["area_m2"] = gdf_projected.geometry.area

        if min_area_m2 > 0:
            gdf_projected = gdf_projected[gdf_projected["area_m2"] >= min_area_m2]

        return {
            "count": int(len(gdf_projected)),
            "total_area_m2": float(gdf_projected["area_m2"].sum()),
        }


class CityCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources = config["sources"]
        self.outputs = config["outputs"]
        self.behavior = config["behavior"]
        self.amenities_cfg = config.get("amenities", {})
        self.validation = config["validation"]
        self.targets = _build_city_targets(config)

        self.force_refresh = bool(self.behavior.get("force_refresh", False))
        self.skip_existing = bool(self.behavior.get("skip_existing", True)) and not self.force_refresh
        self.dry_run = bool(self.behavior.get("dry_run", False))
        self.verbose = bool(self.behavior.get("verbose", False))

        limit = self.behavior.get("small_test_limit")
        if isinstance(limit, int) and limit > 0:
            self.targets = self.targets[:limit]

        self.stats: Dict[str, Dict[str, int]] = {
            "amenities": {"tasks": 0, "updated": 0, "skipped": 0, "errors": 0},
            "demographics": {"updated": 0, "skipped": 0, "errors": 0},
            "census_cache": {"updated": 0, "skipped": 0, "errors": 0},
            "golf": {"files": 0, "fetched": 0, "skipped": 0, "errors": 0},
            "validation": {"checks": 0, "failed": 0},
        }

    def log(self, msg: str):
        print(msg)

    def _should_log_progress(self, current: int, total: int) -> bool:
        if total <= 0:
            return False
        if self.verbose:
            return True

        interval = self.behavior.get("progress_interval", 25)
        if not isinstance(interval, int) or interval <= 0:
            interval = 25

        return current == 1 or current == total or (current % interval == 0)

    def _run_stage(self, stage_num: int, total_stages: int, stage_key: str, stage_label: str, stage_fn):
        if not self.sources.get(stage_key, True):
            self.log(f"\n[STAGE {stage_num}/{total_stages}] {stage_label} - skipped")
            return

        self.log(f"\n[STAGE {stage_num}/{total_stages}] {stage_label} - start")
        start = time.time()
        stage_fn()
        elapsed = time.time() - start
        self.log(f"[STAGE {stage_num}/{total_stages}] {stage_label} - done in {elapsed:.1f}s")

    def run(self):
        total_stages = 4
        self.log("=" * 72)
        self.log("CityCollector: standalone pipeline")
        self.log("=" * 72)
        self.log(f"Targets selected: {len(self.targets)} cities")

        self._run_stage(1, total_stages, "golf", "Golf", self.collect_golf_data)
        self._run_stage(2, total_stages, "amenities", "Amenities", self.collect_amenities)
        self._run_stage(3, total_stages, "demographics", "Demographics", self.collect_demographics)
        self._run_stage(4, total_stages, "census_cache", "Census Cache", self.collect_census_cache)

        if self.validation.get("enabled", True):
            self.run_validation()

        self.print_summary()

    def collect_amenities(self):
        out_path = Path(self.outputs["amenities_json"])
        delay = float(self.behavior.get("amenity_delay_seconds", 0.0))
        save_every = self.behavior.get("amenity_save_every", 25)
        if not isinstance(save_every, int) or save_every <= 0:
            save_every = 25
        buffer_km = float(self.behavior.get("amenity_buffer_km", 0.0))
        amenity_types = self.amenities_cfg.get("types", DEFAULT_AMENITY_TYPES)

        self.log("\n[AMENITIES] Starting amenity collection")
        if self.dry_run:
            self.log(f"[AMENITIES] Dry run: would update {out_path}")
            return

        counter = _BuiltinAmenityCounter(output_json=str(out_path))
        self.log("[AMENITIES] Using provider: builtin")

        total_tasks = len(self.targets) * len(amenity_types)
        processed = 0
        pending_writes = 0
        self.log(f"[AMENITIES] Total tasks: {total_tasks}")

        for target in self.targets:
            city_data = counter.get_city_data(target.city) or {}
            city_polygon = counter.get_city_polygon(
                city=target.city,
                province=target.region,
                country=target.country,
                buffer_km=buffer_km,
            )
            for amenity in amenity_types:
                self.stats["amenities"]["tasks"] += 1
                processed += 1
                if self._should_log_progress(processed, total_tasks):
                    self.log(
                        f"[AMENITIES] Progress {processed}/{total_tasks}: {target.city} - {amenity}"
                    )
                if self.skip_existing and amenity in city_data:
                    self.stats["amenities"]["skipped"] += 1
                    continue

                try:
                    result = counter.count_amenities(
                        city=target.city,
                        location_type=amenity,
                        province=target.region,
                        country=target.country,
                        min_area_m2=0.0,
                        buffer_km=buffer_km,
                        city_polygon=city_polygon,
                    )
                    if target.city not in counter.data:
                        counter.data[target.city] = {}
                    counter.data[target.city][amenity] = result
                    pending_writes += 1
                    if pending_writes >= save_every:
                        counter._save_data()
                        pending_writes = 0
                    self.stats["amenities"]["updated"] += 1
                    if self.verbose:
                        self.log(f"[AMENITIES] {target.city} - {amenity}: {result.get('count', 0)}")
                    if delay > 0:
                        time.sleep(delay)
                except Exception as exc:
                    self.stats["amenities"]["errors"] += 1
                    self.log(f"[AMENITIES] ERROR {target.city} - {amenity}: {exc}")

        if pending_writes > 0:
            counter._save_data()

    def collect_demographics(self):
        out_path = Path(self.outputs["demographics_json"])
        existing = _load_json_if_exists(out_path)

        self.log("\n[DEMOGRAPHICS] Starting demographics collection")
        if self.dry_run:
            self.log(f"[DEMOGRAPHICS] Dry run: would update {out_path}")
            return

        collect_us = _collect_demographics_us
        collect_canada = _collect_demographics_canada
        resolve_state = _fallback_resolve_state_abbr
        province_slug = _fallback_province_slug

        self.log("[DEMOGRAPHICS] Using provider: builtin")

        us_year = 2022
        ca_year = 2021
        total_targets = len(self.targets)
        processed = 0
        self.log(f"[DEMOGRAPHICS] Total targets: {total_targets}")

        for target in self.targets:
            country = target.country.lower()
            processed += 1
            if self._should_log_progress(processed, total_targets):
                self.log(
                    f"[DEMOGRAPHICS] Progress {processed}/{total_targets}: {target.city}, {target.region}"
                )
            try:
                if "united states" in country or country == "us":
                    state_abbr = resolve_state(target.region)
                    if not state_abbr:
                        self.stats["demographics"]["errors"] += 1
                        self.log(f"[DEMOGRAPHICS] Skipping {target.city}: invalid state '{target.region}'")
                        continue

                    city_key = f"{target.city.strip().replace(' ', '_').lower()}_{state_abbr.lower()}"
                    if self.skip_existing and city_key in existing:
                        self.stats["demographics"]["skipped"] += 1
                        continue

                    data = collect_us(target.city, state_abbr, year=us_year)
                    existing[city_key] = data
                    self.stats["demographics"]["updated"] += 1
                    self.log(f"[DEMOGRAPHICS] Fetched US {target.city} ({state_abbr})")
                elif country == "canada":
                    province_code = province_slug(target.region)
                    city_key = f"{target.city.strip().replace(' ', '_').lower()}_{province_code}"
                    if self.skip_existing and city_key in existing:
                        self.stats["demographics"]["skipped"] += 1
                        continue

                    data = collect_canada(target.city, target.region, year=ca_year)
                    existing[city_key] = data
                    self.stats["demographics"]["updated"] += 1
                    self.log(f"[DEMOGRAPHICS] Fetched Canada {target.city} ({target.region})")
                else:
                    self.stats["demographics"]["skipped"] += 1
            except Exception as exc:
                self.stats["demographics"]["errors"] += 1
                self.log(f"[DEMOGRAPHICS] ERROR {target.city}: {exc}")

        _save_json(out_path, existing)

    def collect_census_cache(self):
        census_root = self.outputs.get("census_root", "data/censusShape")
        self.log("\n[CENSUS CACHE] Building per-city profile caches")

        ca_targets = [t for t in self.targets if t.country.lower() == "canada"]
        if not ca_targets:
            self.log("[CENSUS CACHE] No Canadian targets selected.")
            return

        provider = str(self.behavior.get("census_cache_provider", "legacy")).strip().lower()
        if provider in {"legacy", "plugin", "census_cacher"}:
            build_fn = _build_census_cache_legacy
            provider_name = "legacy(embedded)"
        elif provider in {"builtin", "standalone"}:
            build_fn = _build_census_cache
            provider_name = "builtin"
        else:
            self.log(
                f"[CENSUS CACHE] Unknown provider '{provider}', falling back to legacy(embedded)."
            )
            build_fn = _build_census_cache_legacy
            provider_name = "legacy(embedded)"

        self.log(f"[CENSUS CACHE] Using provider: {provider_name}")

        total_targets = len(ca_targets)
        processed = 0
        self.log(f"[CENSUS CACHE] Total Canadian targets: {total_targets}")

        for target in ca_targets:
            processed += 1
            if self._should_log_progress(processed, total_targets):
                self.log(
                    f"[CENSUS CACHE] Progress {processed}/{total_targets}: {target.city}, {target.region}"
                )
            city_slug = _slug_city(target.city)
            city_dir = Path(census_root) / city_slug
            profile_file = city_dir / f"{city_slug}_profile_cache.json"
            if self.skip_existing and profile_file.exists():
                self.stats["census_cache"]["skipped"] += 1
                continue

            if self.dry_run:
                self.log(f"[CENSUS CACHE] Dry run: would build cache for {target.city}")
                continue

            try:
                build_fn(
                    data_root=census_root,
                    city=target.city,
                    province=target.region,
                    overwrite=self.force_refresh,
                    single_pass=True,
                    remove_intermediate_csv=True,
                    write_compat_profile_name=True,
                )
                self.stats["census_cache"]["updated"] += 1
                self.log(f"[CENSUS CACHE] Built cache for {target.city}")
            except Exception as exc:
                self.stats["census_cache"]["errors"] += 1
                self.log(f"[CENSUS CACHE] ERROR {target.city}: {exc}")

    def collect_golf_data(self):
        data_root = Path(self.outputs.get("data_root", "data"))
        golf_json = Path(self.outputs["golf_json"])
        golf_csv = Path(self.outputs["golf_csv"])
        legacy_enabled = bool(self.outputs.get("emit_legacy_golf_files", False))
        legacy_json = Path(self.outputs.get("legacy_golf_json", "data/golf_canada_data_full.json"))
        legacy_csv = Path(self.outputs.get("legacy_golf_csv", "data/golf_canada_data_full.csv"))
        source_globs = self.outputs.get("golf_source_globs") or list(DEFAULT_GOLF_SOURCE_GLOBS)
        use_hf = bool(self.behavior.get("golf_use_hf", True))
        hf_repo_id = str(self.outputs.get("golf_hf_repo_id", "colerjstevenson/GolfGulf"))
        hf_repo_type = str(self.outputs.get("golf_hf_repo_type", "dataset"))
        hf_subdir = str(self.outputs.get("golf_hf_subdir", ""))
        hf_token = (
            self.outputs.get("golf_hf_token")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        hf_cache_dir = data_root / "_hf_golf_cache"

        self.log("\n[GOLF] Starting golf data normalization from local data folder")
        if self.dry_run:
            self.log(f"[GOLF] Dry run: would read from {data_root} and write {golf_json} / {golf_csv}")
            return

        source_files: List[Path] = []
        if data_root.exists():
            self.log("[GOLF] Discovering source files in local data root")
            source_files = _expand_source_globs(data_root, source_globs)
            if source_files:
                self.log(f"[GOLF] Found {len(source_files)} local golf source files; skipping Hugging Face fetch")
        else:
            self.log(f"[GOLF] Local data root not found: {data_root}")

        if not source_files and use_hf:
            self.log(f"[GOLF] No local source files found; pulling from Hugging Face repo: {hf_repo_id} ({hf_repo_type})")
            try:
                source_files = _download_hf_golf_sources(
                    repo_id=hf_repo_id,
                    repo_type=hf_repo_type,
                    token=hf_token,
                    patterns=source_globs,
                    local_dir=hf_cache_dir,
                    subdir=hf_subdir,
                )
                self.log(f"[GOLF] Downloaded/matched {len(source_files)} golf source files from Hugging Face")
            except Exception as exc:
                self.stats["golf"]["errors"] += 1
                self.log(f"[GOLF] Hugging Face source fetch failed: {exc}")

        source_files = [p for p in source_files if p.resolve() not in {golf_json.resolve(), golf_csv.resolve()}]

        file_limit = self.behavior.get("golf_file_limit")
        if isinstance(file_limit, int) and file_limit > 0:
            source_files = source_files[:file_limit]

        self.stats["golf"]["files"] = len(source_files)
        if not source_files:
            self.stats["golf"]["errors"] += 1
            self.log("[GOLF] No source golf files matched configured patterns.")
            return

        self.log(f"[GOLF] Source files to process: {len(source_files)}")

        max_rows = self.behavior.get("max_rows_per_source_file")
        max_rows = max_rows if isinstance(max_rows, int) and max_rows > 0 else None

        by_key: Dict[str, Dict[str, Any]] = {}
        total_files = len(source_files)
        for idx, src in enumerate(source_files, start=1):
            if self._should_log_progress(idx, total_files):
                self.log(f"[GOLF] Progress {idx}/{total_files}: {src}")
            if self.verbose:
                self.log(f"[GOLF] Reading {src}")

            records: List[Dict[str, Any]] = []
            suffix = src.suffix.lower()
            if suffix == ".csv":
                records = _records_from_csv(src, max_rows=max_rows)
            elif suffix == ".json":
                records = _records_from_json(src)
            else:
                self.stats["golf"]["skipped"] += 1
                continue

            if not records:
                self.stats["golf"]["skipped"] += 1
                continue

            for record in records:
                normalized = _normalize_record(record)
                key = _record_key(normalized)
                by_key[key] = normalized
                self.stats["golf"]["fetched"] += 1

        final_records = list(by_key.values())
        _save_json(golf_json, final_records)
        _save_csv_from_records(golf_csv, final_records)

        if legacy_enabled:
            _save_json(legacy_json, final_records)
            _save_csv_from_records(legacy_csv, final_records)

        self.log(f"[GOLF] Wrote {len(final_records)} deduplicated records to {golf_json} and {golf_csv}")

    def run_validation(self):
        self.log("\n[VALIDATION] Running lightweight output checks")

        expected_files = [Path(p) for p in self.validation.get("expected_files", [])]
        if self.sources.get("amenities", True):
            expected_files.append(Path(self.outputs["amenities_json"]))
        if self.sources.get("demographics", True):
            expected_files.append(Path(self.outputs["demographics_json"]))
        if self.sources.get("golf", True):
            expected_files.append(Path(self.outputs["golf_json"]))
            expected_files.append(Path(self.outputs["golf_csv"]))

        deduped: List[Path] = []
        seen = set()
        for path in expected_files:
            key = str(path)
            if key not in seen:
                seen.add(key)
                deduped.append(path)

        for path in deduped:
            self.stats["validation"]["checks"] += 1
            if not path.exists() and not self.dry_run:
                self.stats["validation"]["failed"] += 1
                self.log(f"[VALIDATION] Missing expected file: {path}")

        if not self.dry_run and self.sources.get("amenities", True):
            min_types = int(self.validation.get("min_amenity_types_per_city", 1))
            amenities = _load_json_if_exists(Path(self.outputs["amenities_json"]))
            for target in self.targets:
                city_data = amenities.get(target.city, {})
                self.stats["validation"]["checks"] += 1
                if city_data and len(city_data) < min_types:
                    self.stats["validation"]["failed"] += 1
                    self.log(
                        f"[VALIDATION] {target.city} has {len(city_data)} amenity types (< {min_types})"
                    )

        if not self.dry_run and self.sources.get("demographics", True):
            min_demo = int(self.validation.get("min_demographic_cities", 1))
            demo = _load_json_if_exists(Path(self.outputs["demographics_json"]))
            self.stats["validation"]["checks"] += 1
            if len(demo) < min_demo:
                self.stats["validation"]["failed"] += 1
                self.log(f"[VALIDATION] demographics records={len(demo)} (< {min_demo})")

        if not self.dry_run and self.sources.get("golf", True):
            min_golf = int(self.validation.get("min_golf_records", 1))
            golf = _load_json_list_if_exists(Path(self.outputs["golf_json"]))
            self.stats["validation"]["checks"] += 1
            if len(golf) < min_golf:
                self.stats["validation"]["failed"] += 1
                self.log(f"[VALIDATION] golf records={len(golf)} (< {min_golf})")

    def print_summary(self):
        self.log("\n" + "=" * 72)
        self.log("CityCollector summary")
        self.log("=" * 72)
        for section, values in self.stats.items():
            parts = ", ".join(f"{k}={v}" for k, v in values.items())
            self.log(f"{section}: {parts}")


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace):
    if args.dry_run:
        config["behavior"]["dry_run"] = True
    if args.force:
        config["behavior"]["force_refresh"] = True
        config["behavior"]["skip_existing"] = False
    if args.no_resume:
        config["behavior"]["skip_existing"] = False
    if args.verbose:
        config["behavior"]["verbose"] = True
    if args.small_test_limit is not None:
        config["behavior"]["small_test_limit"] = args.small_test_limit
    if args.golf_file_limit is not None:
        config["behavior"]["golf_file_limit"] = args.golf_file_limit
    if args.max_rows_per_source_file is not None:
        config["behavior"]["max_rows_per_source_file"] = args.max_rows_per_source_file
    if args.progress_interval is not None:
        config["behavior"]["progress_interval"] = args.progress_interval
    if args.census_provider:
        config["behavior"]["census_cache_provider"] = args.census_provider

    if args.sources:
        source_flags = {
            "golf": False,
            "amenities": False,
            "demographics": False,
            "census_cache": False,
        }
        for src in args.sources:
            if src == "census":
                source_flags["census_cache"] = True
            else:
                source_flags[src] = True
        config["sources"].update(source_flags)

    if args.regions:
        config["cities"]["regions"] = args.regions
    if args.city:
        config["cities"]["list"] = args.city
    if args.include:
        config["cities"]["include"] = args.include
    if args.exclude:
        config["cities"]["exclude"] = args.exclude

    if args.amenities_output:
        config["outputs"]["amenities_json"] = args.amenities_output
    if args.demographics_output:
        config["outputs"]["demographics_json"] = args.demographics_output
    if args.golf_json_output:
        config["outputs"]["golf_json"] = args.golf_json_output
    if args.golf_csv_output:
        config["outputs"]["golf_csv"] = args.golf_csv_output
    if args.census_root:
        config["outputs"]["census_root"] = args.census_root
    if args.data_root:
        config["outputs"]["data_root"] = args.data_root
    if args.golf_source_glob:
        config["outputs"]["golf_source_globs"] = args.golf_source_glob


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone city data collection pipeline")
    parser.add_argument("--config", help="Path to YAML or JSON config file")

    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["golf", "amenities", "demographics", "census"],
        help="Select which sources to run",
    )

    parser.add_argument(
        "--regions",
        nargs="+",
        choices=["canada", "us"],
        help="Select default region pools",
    )
    parser.add_argument(
        "--city",
        nargs="+",
        help="Explicit city entries as 'City,Region' or 'City,Region,Country'",
    )
    parser.add_argument("--include", nargs="+", help="Include filters")
    parser.add_argument("--exclude", nargs="+", help="Exclude filters")

    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs")
    parser.add_argument("--force", action="store_true", help="Force refresh existing entries")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing entries")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress logs")
    parser.add_argument("--small-test-limit", type=int, help="Limit number of selected cities")
    parser.add_argument("--golf-file-limit", type=int, help="Limit number of discovered golf source files")
    parser.add_argument("--max-rows-per-source-file", type=int, help="Optional row cap when reading each CSV source")
    parser.add_argument(
        "--progress-interval",
        type=int,
        help="Log progress every N items when not in verbose mode (default: 25)",
    )
    parser.add_argument(
        "--census-provider",
        choices=["legacy", "builtin"],
        help="Census cache provider: legacy (embedded old flow) or builtin",
    )

    parser.add_argument("--amenities-output", help="Override amenities output JSON path")
    parser.add_argument("--demographics-output", help="Override demographics output JSON path")
    parser.add_argument("--golf-json-output", help="Override golf JSON output path")
    parser.add_argument("--golf-csv-output", help="Override golf CSV output path")
    parser.add_argument("--census-root", help="Override census root data folder")
    parser.add_argument("--data-root", help="Override data root used for golf source discovery")
    parser.add_argument(
        "--golf-source-glob",
        nargs="+",
        help="Override golf source glob patterns, relative to data root",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    config = _load_config(args.config)
    _apply_cli_overrides(config, args)

    collector = CityCollector(config)
    collector.run()


if __name__ == "__main__":
    main()
