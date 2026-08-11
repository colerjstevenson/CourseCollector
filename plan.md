# Census API Fallback Implementation Plan (Option 1)

## Goal
Add an API-backed fallback for census cache generation so the pipeline still builds per-city profile cache files when local census source files are missing.

Preserve current behavior:
- Local file path remains primary.
- Existing output contract remains unchanged.
- Fallback activates only when local inputs are unavailable or explicitly requested.

## Current State Summary
- Census cache stage is orchestrated in CityCollector.py (`collect_census_cache`).
- Current builder (`_build_census_cache`) requires local files in `outputs.census_root`:
  - At least one `.shp` with `CTUID`.
  - A CSV with `english_csv_data` in filename.
  - Optional `geo` CSV for DGUID mapping.
- If these files are missing, run errors per city.

## Option 1 Scope
Implement fallback logic only (no full replacement):
1. Try local-file builder first.
2. If missing-input errors occur, run API builder.
3. Write the same output files:
   - `<census_root>/<city_slug>/<city_slug>_profile_cache.json`
   - optional compat file `profile_cache.json`

## Non-Goals
- No removal of local shapefile/CSV path.
- No schema changes to existing profile cache consumers.
- No broad pipeline redesign.

## Design Overview

### 1) New Config Knobs (safe defaults)
In `DEFAULT_CONFIG["behavior"]` and config file support:
- `census_api_fallback: true`
- `census_prefer_api: false`
- `census_api_year: 2021`
- `census_profile_as_single_ctuid: true`

Meaning:
- `census_api_fallback`: enable fallback when local source files are absent.
- `census_prefer_api`: skip local path and use API builder first.
- `census_api_year`: year for API-backed extraction.
- `census_profile_as_single_ctuid`: write a stable synthetic CTUID key for API-only cache format.

### 2) New API Builder Function
Add function near `_build_census_cache`:
- `_build_census_cache_api(data_root, city, province, overwrite=False, write_compat_profile_name=True, year=2021)`

Behavior:
1. Resolve city directory and skip when cache exists unless overwrite.
2. Fetch city-level Canadian demographics from existing `_collect_demographics_canada`.
3. Transform metrics into profile cache shape compatible with downstream expectations:
   - top-level dict keyed by one synthetic CTUID (e.g., `CITY_<slug>`).
   - category blocks mirroring current cache pattern, for example:
     - `Total - Population and dwelling counts`
     - `Total - Income`
     - `Total - Labour`
     - `Total - Housing`
     - `Total - Immigrant status and period of immigration`
     - `Total - Visible minority and population group`
4. Map known metric keys into those categories; omit unavailable metrics.
5. Save both cache files with existing helper `_save_json`.

Notes:
- This is city-level data, not tract-level detail. Keep this explicit in metadata.
- Include metadata field in each city cache payload, for example category `__meta__`:
  - `source_mode: "api_fallback_city_level"`
  - `data_source`
  - `year`
  - `city`
  - `province`

### 3) Fallback Trigger Logic
Update `collect_census_cache` flow:
1. Determine mode via config flags.
2. If `census_prefer_api` true: use API builder directly.
3. Else attempt local builder.
4. On failure, if `census_api_fallback` true and error indicates missing local inputs, retry with API builder.

Create helper classifier:
- `_is_missing_census_local_inputs_error(exc: Exception) -> bool`

Match messages such as:
- `No shapefile found`
- `Could not find English census profile CSV`
- `Shapefile missing CTUID column`

Do not fallback for unrelated errors (network outages, invalid city boundary, etc.) unless explicitly desired.

### 4) Logging & Stats
Add clear logs in census stage:
- `[CENSUS CACHE] Local mode success: <city>`
- `[CENSUS CACHE] Local inputs missing; switching to API fallback: <city>`
- `[CENSUS CACHE] API fallback success: <city>`
- `[CENSUS CACHE] API fallback failed: <city>: <error>`

Optional stat extension:
- `stats["census_cache"]["api_fallback_used"] += 1`

### 5) Documentation Update
Update README_city_collector.md:
- Explain dual-mode census cache behavior.
- Document new behavior flags.
- Clarify fallback output is city-level approximation when local tract files are absent.

## Data Shape Compatibility Plan
Maintain compatibility by preserving file names and top-level structure (dict -> ctuid -> categories -> metrics).

Proposed synthetic key format:
- `CITY_<city_slug>` (stable, deterministic)

Example structure:
{
  "CITY_toronto": {
    "Total - Population and dwelling counts": {
      "Population, 2021": 12345
    },
    "Total - Housing": {
      "Owner occupied private dwellings": 1000,
      "Renter occupied private dwellings": 900
    },
    "__meta__": {
      "source_mode": "api_fallback_city_level",
      "year": 2021
    }
  }
}

## Implementation Steps

1. Add config defaults and CLI/config plumbing
- Extend `DEFAULT_CONFIG["behavior"]`.
- Ensure `_apply_arg_overrides` can optionally wire future CLI flags if desired.

2. Add API fallback builder
- Implement `_build_census_cache_api`.
- Reuse `_collect_demographics_canada` to avoid duplicate API code.
- Build mapper from demographics keys to cache categories/metric names.

3. Add fallback error classifier
- Implement `_is_missing_census_local_inputs_error`.

4. Patch census stage orchestration
- Modify `collect_census_cache` decision flow.
- Add logs and fallback counter.

5. Update docs
- README_city_collector.md new section: Census Cache Modes.

6. Validate locally
- Run census with empty `data/censusShape` and confirm API fallback writes cache files.
- Run with valid local files and confirm local mode still used.
- Run with fallback disabled to confirm previous strict behavior preserved.

## Test Matrix

### A) Local files missing, fallback enabled
Config:
- `census_api_fallback: true`
- `census_prefer_api: false`
Expected:
- No hard failure for missing `.shp`/CSV.
- Per-city cache files created from API.

### B) Local files present
Config:
- same as A
Expected:
- Local path used.
- Existing behavior/output unchanged.

### C) Force API mode
Config:
- `census_prefer_api: true`
Expected:
- API builder used regardless of local files.

### D) Strict local mode
Config:
- `census_api_fallback: false`
- `census_prefer_api: false`
Expected:
- Existing missing-file error behavior preserved.

## Risks and Mitigations

Risk: Downstream expects real tract CTUID granularity.
- Mitigation: Keep structure compatible; include `source_mode` metadata; document clearly.

Risk: pycancensus dependency/API key issues.
- Mitigation: Reuse current error handling from `_collect_demographics_canada`; log actionable messages.

Risk: Mixed cache provenance across cities.
- Mitigation: Add per-city metadata showing local vs API fallback mode.

## Suggested CLI Additions (Optional)
- `--census-api-fallback / --no-census-api-fallback`
- `--census-prefer-api`
- `--census-api-year <year>`

These can be deferred if config-only control is enough.

## Definition of Done
- Census stage no longer hard-fails solely because local `.shp`/`english_csv_data` are absent.
- Existing local-file path still functions unchanged.
- Output file names and loading contract remain compatible.
- README clearly documents behavior and caveats.
- At least one successful run verified for both local and fallback modes.
