# CourseCollector

CourseCollector is a Python data pipeline for gathering golf course data from multiple sources, normalizing it, enriching missing fields, and generating an interactive map for review and manual cleanup.

This README documents how the repository is organized, what each file does, and the typical end-to-end workflow.

## Unified Orchestrator (Current Workflow)

The recommended path is now the unified CLI entrypoint at the repository root:

```bash
python CourseCollector run --target world --config course_config.yaml
```

This entrypoint routes into the new package under src/course_collector/ and runs the non-map pipeline as a sequence of stages:

- collect: gathers regional OSM-based golf course data
- scrape: optionally enriches records with external metadata from GolfLink, Golf Canada, or Golf Digest
- combine: merges regional outputs into target-specific combined CSV/GeoJSON files
- match: adds postal codes and produces a matched output dataset

Common examples:

```bash
# full default stage sequence (collect -> scrape -> combine -> match)
python CourseCollector run --target usa --config course_config.yaml

# run one stage only
python CourseCollector collect --target world --config course_config.yaml
python CourseCollector scrape --target world --config course_config.yaml
python CourseCollector combine --target world --config course_config.yaml
python CourseCollector match --target world --config course_config.yaml

# dry run with no writes
python CourseCollector run --target world --config course_config.yaml --dry-run

# resume from checkpoints and skip already completed stages
python CourseCollector run --target world --config course_config.yaml --resume

# rerun stages even if checkpoints say they are complete
python CourseCollector run --target world --config course_config.yaml --force

# start later or stop earlier in the stage order
python CourseCollector run --target world --config course_config.yaml --from-stage combine --to-stage match
```

Notes:
- Map generation remains a separate step and is not part of the orchestrator pipeline.
- The main configuration lives in course_config.yaml and controls target profiles, output paths, logging, checkpoints, and scrape toggles.
- Legacy scripts still exist as the underlying implementation for some stages while the new orchestrator package is being finalized.

## What This Repo Produces

- Regional course datasets from OpenStreetMap (CSV + GeoJSON)
- Scraped course metadata from public golf directories (CSV + JSON)
- Combined and normalized regional datasets
- Postal-code based matching between map data and external course metadata
- Interactive HTML map with editing and merge tools

## Current Recommended Workflow

1. Choose a target profile in course_config.yaml (`usa` or `world`) and confirm the regions file for collection.
2. Run the collect stage to pull geospatial golf course polygons for the selected regions.
3. Optionally run the scrape stage to enrich the dataset with external golf-directory metadata.
4. Run the combine stage to build consolidated CSV and GeoJSON outputs for the target.
5. Run the match stage to add postal codes and produce the final matched dataset.
6. Use the map generator separately if you want to review, edit, or merge the matched records interactively.

## Repository Structure

- Data outputs:
  - data/ (large; contains regional outputs and combined files)
  - cache/ and cache_bu/ (HTTP/cache artifacts)
  - images/ (generated PNG maps)
- Core scripts:
  - CourseCollector (single-entry orchestrator launcher)
  - src/course_collector/ (new orchestrator package)
- Legacy scripts:
  - legacy/golfSearchUSA.py, legacy/golfSearchWorld.py, legacy/golfSearch.py, legacy/golfSearchbyRegion.py
  - legacy/cleaner.py
  - legacy/postal_lookup.py
  - legacy/golfLinkScrapper.py, legacy/golfCanadaScrapper.py, legacy/golfCanadaCombiner.py, legacy/golfdigest_urls.py
  - legacy/map_generator.py
  - legacy/nameFiller.py
- Other files:
  - states_list.txt (state/country list for batch runs)
  - golf_course_cache.json (Google Places lookup cache)
  - golf_course_collection.log (pipeline logs)
  - legacy/golfsearchUnitedStates (alternate CLI collector script)
  - legacy/mapMaker.py (currently empty)
  - token (local token file)

## Static Data Site (Landing + World Map)

The repository includes a deploy-ready static site pipeline built around a Hugging Face dataset snapshot.

- Source templates and assets:
  - docs/templates/index.html
  - docs/templates/world-map.html
  - docs/assets/
  - docs/config/site_links.json
- Data preparation and validation scripts:
  - scripts/prepare_dataset.py
  - scripts/validate_site_dataset.py
- Build and dataset scripts:
  - scripts/build_site.py
  - scripts/publish_hf_dataset.py
  - scripts/pull_hf_dataset.py
- Build output:
  - maps/index.html
  - maps/world-map.html
  - maps/assets/
  - maps/data/golf_courses.geojson
  - maps/data/cities.json
  - maps/data/city_demographics.json
  - maps/data/amenities.json
  - maps/data/derived/city_course_counts.json
  - maps/data/landing_summary.json
  - maps/data/site_links.json

### Local Build (from local data)

1. Prepare dataset-shaped artifacts from local collector outputs:

```bash
python scripts/prepare_dataset.py --data-root data --output data/site_dataset
```

2. Validate artifacts:

```bash
python scripts/validate_site_dataset.py --dataset-root data/site_dataset --min-cities 5
```

3. Build static site pages from prepared dataset:

```bash
python scripts/build_site.py --dataset-root data/site_dataset --output maps
```

### Local Build (from Hugging Face dataset)

```bash
python scripts/pull_hf_dataset.py --dataset colerjstevenson/GolfGulf --output .hf_dataset --clean
python scripts/validate_site_dataset.py --dataset-root .hf_dataset --min-cities 5
python scripts/build_site.py --dataset-root .hf_dataset --output maps
```

### GitHub Workflows

- .github/workflows/deploy-pages.yml
  - Pulls dataset from Hugging Face.
  - Works even when data/ is not committed to git.
  - If site-ready files are missing at dataset root, it builds them from raw snapshot files in the dataset.
  - Validates required JSON/GeoJSON files.
  - Builds maps/ static output.
  - Runs smoke checks.
  - Deploys to GitHub Pages.

- .github/workflows/publish-hf-dataset.yml
  - Prepares and validates data/site_dataset artifacts.
  - Publishes JSON/GeoJSON artifacts to the Hugging Face dataset.
  - Uses HF_TOKEN secret and optional HF_DATASET_ID repository variable.
  - If no local data inputs exist on the runner, it safely exits without publishing.

Notes:

- Tract-level city map pages are intentionally deferred.
- The landing page now renders city cards directly from city-level demographics + amenities + course counts.
- The world map still uses point clustering and debounced search for large datasets.
- Local data files are gitignored by design; production deploys are expected to pull from Hugging Face.

## Script-by-Script Guide

## 1) OSM Data Collection

### golfSearchUSA.py
Purpose:
- Main batch collector for USA states listed in states_list.txt.

Behavior:
- Reads states list, skips commented lines.
- For each state, fetches OSM golf_course features.
- Projects geometry, computes area, centroid lat/lon.
- Fills missing names using Google Places helper.
- Writes per-state files to data/usa/:
  - golf_courses_<state>.geojson
  - golf_courses_<state>.csv
- Optionally writes PNG overlays to images/.

Entry point:
- Run directly to process all active lines in states_list.txt.

### golfSearchWorld.py
Purpose:
- Same pattern as USA script, but writes to data/world/ and supports non-US regions in states_list.txt.

Behavior:
- Reads lines from states_list.txt, skips comments.
- Processes each region name as an OSM place query.
- Uses region abbreviation map where available for GCID prefixes.

Outputs:
- data/world/golf_courses_<region>.geojson
- data/world/golf_courses_<region>.csv

### golfSearch.py
Purpose:
- Single-region collector (currently hardcoded to Newfoundland and Labrador).

Behavior:
- Fetches OSM golf courses for one region.
- Computes area and centroids, fills names if missing.
- Writes to data/ as:
  - golf_courses_<region>.geojson
  - golf_courses_<region>.csv
- Optionally writes map image to images/.

Notes:
- Useful for testing one place quickly.

### golfSearchbyRegion.py
Purpose:
- Collect by administrative subregions (admin boundaries) within a province.

Behavior:
- Gets province polygon, then admin level 6 or 7 subregions.
- Fetches golf features per subregion and merges results.
- Writes timestamped outputs in working directory:
  - <province>_<timestamp>.geojson
  - <province>_<timestamp>.csv
  - optional PNG map

Notes:
- Designed for finer-grained region pulls.

### golfsearchUnitedStates
Purpose:
- Newer CLI-style collector with scope controls.

Behavior:
- Supports CLI flags for max states, max subregions, max courses, min area, output directory, pause between states, and map toggles.
- Intended for safer throttling on large runs.

Important:
- This file currently appears incomplete/broken in at least one line and may not run without fixes.

## 2) External Metadata Scrapers

### golfLinkScrapper.py
Purpose:
- Scrape GolfLink course pages from a sitemap.

Behavior:
- Pulls URLs from sitemap XML.
- Scrapes metadata from each course page.
- Cleans HTML text and links.
- Writes:
  - data/golfLinkData.json
  - data/golfLinkData.csv

### golfCanadaScrapper.py
Purpose:
- Scrape Golf Canada course pages from several sitemap files.

Behavior:
- Iterates through sitemap URLs.
- Keeps only English page URLs.
- Extracts course facts and cleans values.
- Appends output to:
  - data/golf_canada_data_full.json
  - data/golf_canada_data_full.csv

Note:
- Save mode is append for both JSON and CSV; repeated runs can duplicate/append data.

### golfCanadaCombiner.py
Purpose:
- Consolidate Golf Canada JSON outputs into normalized full files.

Behavior:
- Reads files matching golf_canada_data* under data/canada.
- Normalizes strings and splits Address field.
- Removes French URL records.
- Drops sparse columns.
- Writes:
  - data/canada/golf_canada_full.csv
  - data/canada/golf_canada_full.json

### golfdigest_urls.py
Purpose:
- Discover and optionally scrape Golf Digest course URLs from sitemap index.

Behavior:
- Fetches sitemap index, resolves nested sitemaps, filters /courses/ URLs.
- Scrapes pages and writes:
  - courses_urls.json
  - courses_data.json
  - courses_data.csv

Note:
- Outputs default to repo root unless changed in script.

## 3) Data Combination, Enrichment, and Matching

### cleaner.py
Purpose:
- Combine OSM regional files into consolidated world-level outputs.

Behavior:
- Scans data/world recursively for golf_courses_*.csv and golf_courses_*.geojson.
- Skips regions already present in existing combined files.
- Normalizes key columns (gcid, name, province).
- Optionally translates non-English text to English.
- Reorders sparse vs dense columns.
- Writes:
  - data/world/combined.csv
  - data/world/combined.geojson

Notes:
- Translation depends on deep-translator or googletrans.
- Script is incremental and merge-oriented.

### postal_lookup.py
Purpose:
- Add postal codes and perform postal-based matching to external metadata.

Behavior:
- add_postal_codes:
  - Reads combined coordinate CSV.
  - Looks up missing postal codes with geopy Nominatim reverse geocoding.
  - Appends results to postal_codes.csv.
- greedy_match_by_postal:
  - Matches combined OSM rows to GolfLink rows by postal code.
  - Falls back to name similarity scoring when postal match is missing/ambiguous.
  - Writes unified table with match metadata.

Default main run target:
- COUNTRY = world
- Inputs:
  - data/world/combined.csv
  - data/world/postal_codes.csv
  - data/golfLinkData.csv
- Output:
  - data/world/Fully_Matched_Golf_Courses.csv

### nameFiller.py
Purpose:
- Fill missing course names using Google Places Nearby Search.

Behavior:
- Called by OSM collectors when a course name is blank.
- Caches lookups in golf_course_cache.json keyed by rounded lat/lon.

Important:
- Uses `GOOGLE_MAPS_API_KEY` environment variable for Google Places lookup.
- If `GOOGLE_MAPS_API_KEY` is not set, missing names are left unresolved instead of calling the API.

## 4) Interactive Review and Manual Editing

### map_generator.py
Purpose:
- Build interactive Folium map from one or many CSV sources.

Behavior:
- Loads and merges multiple CSV files.
- Standardizes schema (lat/lon vs latitude/longitude, name vs CourseName).
- Clusters markers and colors unmatched rows differently.
- Loads polygon overlays from data/*/combined.geojson.
- Injects custom JavaScript for:
  - Row editing and saving back to source CSV
  - External data linking/merge into missing fields
  - GCID search and polygon hover highlighting
- Starts local HTTP server with API endpoint:
  - POST /api/update_row
- Default input files in main:
  - data/canada/Fully_Matched_Golf_Courses.csv
  - data/usa/Fully_Matched_Golf_Courses.csv
  - data/mexico/Fully_Matched_Golf_Courses.csv
  - data/world/combined.csv
- Default output:
  - golf_courses_map.html

### golf_courses_map.html
Purpose:
- Generated map artifact. Can be opened directly, but edit/save features need the local server started by map_generator.py.

## Incomplete or Legacy Files

### CourseCollector
- Active launcher entrypoint that routes commands into `src/course_collector/cli.py`.
- Additional stage internals are still being migrated from legacy scripts.

### mapMaker.py
- Empty file.

## Configuration Guide

The main runtime settings are stored in course_config.yaml. The file is read by the orchestrator via the --config argument and provides a small, explicit configuration surface for the pipeline.

Key sections:
- logging.path: location of the pipeline log file
- checkpoint.path: location of the stage-completion checkpoint file
- targets.usa and targets.world: per-target settings for the regions file, data directory, combined CSV, postal-code CSV, and matched-output CSV
- inputs.golflink_csv: input path used by the matching stage
- scrape.golflink, scrape.golfcanada, scrape.golfdigest: booleans that control which external scrapers run

Example structure:

```yaml
logging:
  path: golf_course_collection.log

checkpoint:
  path: .course_collector/checkpoint.json

targets:
  usa:
    regions_file: states_list.txt
    data_dir: data/usa
    combined_csv: data/usa/combined.csv
    postal_csv: data/usa/postal_codes.csv
    matched_csv: data/usa/Fully_Matched_Golf_Courses.csv
  world:
    regions_file: states_list.txt
    data_dir: data/world
    combined_csv: data/world/combined.csv
    postal_csv: data/world/postal_codes.csv
    matched_csv: data/world/Fully_Matched_Golf_Courses.csv

inputs:
  golflink_csv: data/golfLinkData.csv

scrape:
  golflink: false
  golfcanada: false
  golfdigest: false
```

When you want to keep multiple configuration variants, use course_config.yaml as the main pipeline config and give other files more specific names such as course_config_world.yaml or course_config_debug.yaml.

## CityCollector quick run

`CityCollector.py` has one required package dependency and three optional plugin modules:

- Required package: `PyYAML` (for loading `city_config*.yaml`)
- Optional local plugin modules: `city_amenity_counter.py`, `collect_city_demographics.py`, `census_cacher.py`

Install the required package:

```bash
python -m pip install -r requirements-citycollector.txt
```

Run the standalone collector with the included golf-only config (works without optional plugin modules):

```bash
python CityCollector.py --config city_config.golf_only.yaml
```

If you later add the optional plugin modules, switch back to `city_config.yaml` to enable amenities, demographics, and census cache stages.

## Typical Run Sequences

## A) Build or refresh the world dataset with the orchestrator

1. Edit states_list.txt to include the countries or regions you want to collect.
2. Run the orchestrator for the world target:
   ```bash
   python CourseCollector run --target world --config course_config.yaml
   ```
3. If you want to inspect or rerun stages individually, use collect/combine/match as needed.

## B) Build or refresh the USA dataset with the orchestrator

1. Edit states_list.txt for the US states you want to process.
2. Run:
   ```bash
   python CourseCollector run --target usa --config course_config.yaml
   ```
3. Review the generated matched output in data/usa/Fully_Matched_Golf_Courses.csv.

## C) Refresh external metadata before matching

1. Enable the desired scrapers in course_config.yaml under scrape.
2. Run:
   ```bash
   python CourseCollector scrape --target world --config course_config.yaml
   ```
3. Follow with the combine and match stages to incorporate the new metadata into the final dataset.

## Dependencies (from imports)

Core:
- pandas
- requests
- beautifulsoup4
- lxml

Geospatial:
- osmnx
- geopandas
- shapely (transitive with geopandas)
- pyproj (transitive with geopandas)
- contextily (optional basemaps)
- matplotlib
- folium

Enrichment:
- geopy
- deep-translator or googletrans (optional translation)

## Operational Notes

- The data folder is large; scripts generally process files by pattern and can be incremental.
- Some scripts append output instead of overwriting. Verify duplicates before downstream merges.
- Long batch runs write to golf_course_collection.log.
- API and geocoding services may rate-limit. Consider adding pauses and batching states/regions.

## Suggested Future Cleanup

- Move API keys and tokens to environment variables.
- Consolidate duplicate collectors (golfSearchUSA.py, golfSearchWorld.py, golfsearchUnitedStates).
- Replace incomplete CourseCollector stub with a real orchestrator CLI.
- Add a requirements.txt or pyproject.toml for reproducible setup.
- Add small sample-data test fixtures and smoke tests for each stage.
