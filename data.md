# Golf course data layout

This document summarizes how the golf course data under the data folder is organized without opening the large data files directly.

## High-level idea

The data is organized by:

- geography (USA, Canada, Mexico, World)
- output stage (raw collection, scraped metadata, combined/normalized output, matched output)
- file type (CSV, JSON, GeoJSON)

## Top-level folders

### data/usa/
Used for USA collection and matching outputs.

Typical files:
- golf_courses_<state>.csv
- golf_courses_<state>.geojson
- combined.csv
- combined.geojson
- Fully_Matched_Golf_Courses.csv
- postal_codes.csv

Conventions:
- Each state gets its own per-state output pair.
- The combined files are merged datasets for the USA target.
- The matched file is the final cleaned/matched result for review.

### data/world/
Used for world-region collection and combined outputs.

Typical files:
- golf_courses_<region>.csv
- golf_courses_<region>.geojson
- combined.csv
- combined.geojson
- postal_codes.csv

Conventions:
- Regions are stored as separate files using region names as part of the filename.
- The combined files represent a merged world-level view.

### data/canada/
Used for Canadian course collection and scraper output.

Typical files:
- golf_canada_data*.csv
- golf_canada_data*.json
- golf_canada_full.csv
- golf_canada_full.json
- golf_courses_<province>.csv
- golf_courses_<province>.geojson
- combined.csv
- combined.geojson
- Fully_Matched_Golf_Courses.csv
- postal_codes.csv

Conventions:
- Raw scraper outputs are stored as numbered or sitemap-based files.
- The full files are normalized Canada-wide outputs.
- Provincial files are separate regional datasets.

### data/mexico/
Used for Mexico collection and matching outputs.

Typical files:
- golf_courses_mexico.csv
- golf_courses_mexico.geojson
- combined.csv
- combined.geojson
- Fully_Matched_Golf_Courses.csv
- postal_codes.csv

## Root-level data files

These files sit directly under data/ and are usually source or cross-region datasets.

- golfLinkData.csv
- golfLinkData.json
- golf_canada_data_golfcourse001.csv
- golf_canada_data_golfcourse001.json

These are generally scraper exports or cross-region reference files that feed into the regional and combined outputs.

## File type guide

- .csv: tabular course data, usually the most convenient form for analysis and matching
- .json: structured scraper or metadata output
- .geojson: geospatial features with coordinates/polygons for mapping

## Naming conventions

Common patterns:
- golf_courses_<name>.<ext> = per-region or per-area collection output
- combined.<ext> = merged output for a region or target group
- Fully_Matched_Golf_Courses.csv = final matched dataset
- postal_codes.csv = postal code lookup results used during matching
- golf_canada_data*.csv/.json = intermediate Golf Canada scraper artifacts
- golfLinkData.* = GolfLink scraper export files

## Typical workflow

The data is generally produced in stages:

1. Collection stage
   - creates per-region files such as golf_courses_<region>.csv/.geojson
2. Scrape stage
   - creates external metadata files such as golfLinkData.* or golf_canada_data*.*
3. Combine stage
   - creates combined.csv and combined.geojson for a target area
4. Match stage
   - produces Fully_Matched_Golf_Courses.csv and postal_codes.csv

## Practical note

The data folder is large and is mostly processed by pattern (for example, all files matching golf_courses_*.csv or all files under a regional folder) rather than by opening each large file individually.
