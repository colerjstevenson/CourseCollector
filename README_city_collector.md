# CityCollector README

## Overview

`CityCollector.py` is the standalone orchestration script for city-level data collection in this repo. It can run up to four data sources in one pipeline:

- golf (local file normalization and deduping)
- amenities (OpenStreetMap-based amenity counts)
- demographics (US ACS + Canadian census via plugin, with fallback behavior)
- census cache (per-city Canadian profile cache files)

The default config is in `city_config.yaml`.

## What It Produces

By default, the collector writes:

- `data/city_amenities.json`
- `data/city_demographics.json`
- `data/canada/golf_canada_full.json`
- `data/canada/golf_canada_full.csv`
- `data/censusShape/<city_slug>/<city_slug>_profile_cache.json` (for Canadian targets)

Optional legacy golf outputs can be enabled in config:

- `data/golf_canada_data_full.json`
- `data/golf_canada_data_full.csv`

## Prerequisites

Install Python dependencies for the sources you want to run.

Minimum useful set:

```powershell
pip install pyyaml pandas
```

For amenities collection (OSM):

```powershell
pip install osmnx geopandas
```

For demographics plugins:

```powershell
pip install pytidycensus us stats_can pycancensus
```

If you run US demographics with the ACS path, set:

- `CENSUS_API_KEY` (US Census API key)

If you run Canadian demographics through `pycancensus`, set:

- `CANCENSUS_API_KEY` (CensusMapper key)

## Quick Start

Run the full default pipeline using `city_config.yaml`:

```powershell
python CityCollector.py --config city_config.yaml
```

Run only amenities and demographics:

```powershell
python CityCollector.py --config city_config.yaml --sources amenities demographics
```

Dry run (no writes):

```powershell
python CityCollector.py --config city_config.yaml --dry-run
```

Force refresh existing records:

```powershell
python CityCollector.py --config city_config.yaml --force
```

## City Selection

### Use default pools

By default, `regions` in config selects from Canada/US pools:

```powershell
python CityCollector.py --config city_config.yaml --regions canada us
```

### Run specific cities only

Pass one or more explicit city specs:

- `City,Region`
- `City,Region,Country`

```powershell
python CityCollector.py --config city_config.yaml --city "Toronto,Ontario" "Seattle,Washington,United States"
```

### Include/Exclude filters

```powershell
python CityCollector.py --config city_config.yaml --include Toronto Vancouver --exclude Ottawa
```

## Useful CLI Flags

- `--sources golf amenities demographics census`
- `--dry-run`
- `--force`
- `--no-resume`
- `--verbose`
- `--small-test-limit <N>`
- `--golf-file-limit <N>`
- `--max-rows-per-source-file <N>`

Output/location overrides:

- `--amenities-output <path>`
- `--demographics-output <path>`
- `--golf-json-output <path>`
- `--golf-csv-output <path>`
- `--census-root <path>`
- `--data-root <path>`
- `--golf-source-glob <glob...>`

## Config Notes (`city_config.yaml`)

Key sections:

- `sources`: toggle pipeline parts (`golf`, `amenities`, `demographics`, `census_cache`)
- `cities`: pools, explicit `list`, `regions`, `include`, `exclude`
- `amenities.types`: amenity list to query from OSM
- `plugins`: external module/class/function hooks for amenities, demographics, census cache
- `outputs`: output file paths and golf source globs
- `behavior`: run controls (dry run, skipping existing, delays, limits)
- `validation`: lightweight post-run checks

## Plugin Behavior

The collector supports plugin overrides. If plugin modules are missing or not configured, built-in fallbacks are used:

- amenities: built-in OSM counter
- demographics: metadata fallback records if plugin functions are unavailable
- census cache: minimal fallback profile cache JSON

To use your custom module, set the corresponding `plugins.<section>.module` in `city_config.yaml`.

## Golf Input Discovery

Golf normalization reads local files under `outputs.data_root` (default: `data`) using configured glob patterns such as:

- `canada/Fully_Matched_Golf_Courses.csv`
- `canada/golf_canada_full.csv`
- `usa/Fully_Matched_Golf_Courses.csv`
- `world/combined.csv`
- `golfLinkData.csv`

Records are normalized and deduplicated (URL/GCID/name-address-city key logic) before writing final outputs.

## Validation

When `validation.enabled: true`, the script checks:

- expected output files exist
- minimum amenity types per city
- minimum demographics city records
- minimum golf record count

Validation thresholds are configurable in `city_config.yaml`.

## Recommended Incremental Runs

Fast smoke test on a few cities:

```powershell
python CityCollector.py --config city_config.yaml --small-test-limit 2 --sources amenities demographics --verbose
```

Rebuild only golf normalized outputs:

```powershell
python CityCollector.py --config city_config.yaml --sources golf --force
```

Build only Canadian census cache profiles:

```powershell
python CityCollector.py --config city_config.yaml --sources census --regions canada
```

## Troubleshooting

- `YAML config requires PyYAML`: install `pyyaml`.
- `osmnx is not installed`: install `osmnx` and `geopandas`, or disable amenities source.
- US demographics skipped/failing: verify `pytidycensus`, `us`, and `CENSUS_API_KEY`.
- Canadian demographics failing via API: verify `pycancensus` and `CANCENSUS_API_KEY`.
- No golf files matched: confirm `outputs.data_root` and `outputs.golf_source_globs`.

## Related Files

- `CityCollector.py`
- `city_config.yaml`
- `city_amenity_counter.py`
- `collect_city_amenities.py`
- `collect_city_demographics.py`
- `census_cacher.py`
