# Data Site Implementation Plan

## Current Status (2026-08-10)

Overall: Deployment-ready for phase 1. The site now builds from dataset-shaped artifacts, can be pulled from Hugging Face in CI, and includes city-level cards from demographics and amenities without tract-level pages.

Completed in this session:

- Added dataset packaging script (`scripts/prepare_dataset.py`) for world map + city-level files.
- Added dataset validation script (`scripts/validate_site_dataset.py`) for release quality gates.
- Refactored static site build to consume dataset snapshots (`scripts/build_site.py --dataset-root`).
- Updated landing page to render searchable city cards using demographics and amenities data.
- Added GitHub Pages deploy workflow (`.github/workflows/deploy-pages.yml`) using Hugging Face snapshot pull + validation + build + smoke checks.
- Updated Hugging Face publish workflow to run preparation + validation before upload.
- Added README runbook for local and CI release flow.

Not started yet:

- Tract-level city map overlays/pages (`cities/<city_slug>.geojson`) are still pending and intentionally excluded from this phase.

## Goal

Build and ship a static data site that:

- Is hosted on GitHub Pages.
- Uses a public Hugging Face dataset as the source of truth for map and city data.
- Preserves the current user experience (city search, city cards, city maps, visuals, data links).
- Adds reliable publishing so data and pages stay in sync.

## Scope

### In scope

- Landing page and navigation.
- City-level interactive map pages.
- Global golf course map page with hover details and search.
- Visuals section links.
- Data download section.
- Automated build and deploy pipeline.
- Public Hugging Face dataset structure and update workflow.

### Out of scope (phase 1)

- User authentication.
- Server-side APIs.
- Real-time edits in browser.
- Paid/private datasets.

## Target Architecture

### High-level flow

1. Data is collected/updated locally by existing scripts.
2. A publish script validates and transforms data into web-ready artifacts.
3. Artifacts are pushed to a Hugging Face dataset repository.
4. GitHub Actions builds the site using pinned dataset files.
5. Site is deployed to GitHub Pages.

### Why this architecture

- GitHub Pages is static and low-maintenance.
- Hugging Face datasets provide versioned, public, linkable data files.
- GitHub Actions provides repeatable deployment and rollback via git history.

## Data Model for Hugging Face Dataset

Use this Hugging Face dataset repository as the source of truth:

- colerjstevenson/GolfGulf

Recommended files:

- README.md (dataset card, schema, license, update cadence)
- manifest.json (version, build timestamp, commit hash, file checksums)
- cities.json (city metadata for card grid)
- city_demographics.json (city-level census fields)
- amenities.json (or split by category if large)
- golf_courses.geojson (global golf course points)
- cities/<city_slug>.geojson (city map overlays, one file per city)
- derived/city_course_counts.json (precomputed counts for fast landing page render)

Required fields for cities.json:

- city_name
- city_slug
- province
- lat
- lon
- population
- golf_course_count
- city_map_path

Required fields for each golf course feature:

- id
- name
- city_slug
- province
- country
- latitude
- longitude
- holes (nullable)
- course_type (nullable)
- source
- last_updated

## Repository Structure (Proposed)

Within this repository:

- maps/ (deployed static site output)
- site_src/ (templates, JS, CSS source)
- data/ (local generated artifacts prior to upload)
- scripts/
- scripts/build_site.py
- scripts/prepare_dataset.py
- scripts/publish_dataset.py
- .github/workflows/deploy-pages.yml
- .github/workflows/publish-dataset.yml

## Implementation Phases

## Phase 1: Baseline and Inventory

Status: In progress

1. Audit any existing visualization pages and data sources.
2. Inventory all data inputs that will feed the generated site pages.
3. Define canonical schemas for each output file.
4. Add lightweight validation rules (required fields, types, coordinate bounds).
5. Since map HTML files do not exist yet, define the generation strategy and templates for city pages and the global golf course map.

Deliverable:

- Schema document + validation script passing on current data.

## Phase 2: Dataset Packaging for Hugging Face

Status: In progress

1. Build scripts/prepare_dataset.py to normalize and split data.
2. Generate manifest.json with version and checksum metadata.
3. Add scripts/publish_dataset.py to upload to Hugging Face dataset repo.
4. Create dataset README with field definitions and license.

Deliverable:

- Public Hugging Face dataset containing all required files.

## Phase 3: Site Refactor to Data-Driven Rendering

Status: In progress

Progress notes:

- Landing page and global golf course map are implemented from templates.
- Global map uses clustered point rendering for large datasets.
- Current build uses local data/*/combined.csv as interim inputs until Hugging Face + city/census artifacts are wired.

1. Refactor landing page to read cities.json and derived/city_course_counts.json.
2. Keep search/filter fully client-side.
3. Update city cards to link to city map pages using city_slug.
4. Generate city-level map pages and the global golf course map page from templates and dataset-shaped data during the build.
5. Ensure Visuals section and Data section are generated from config.
6. Build the global golf course map page from golf_courses.geojson with:
- hover/click popup details
- city/name search
- optional clustering if point volume is high

Deliverable:

- Fully functional local site using dataset-shaped data.

## Phase 4: GitHub Pages Deployment

Status: In progress

1. Configure Pages deployment target (gh-pages branch or Pages artifact workflow).
2. Build deploy-pages.yml:
- Trigger on push to main and manual dispatch.
- Fetch pinned dataset version from Hugging Face.
- Build static assets into deploy directory.
- Publish to GitHub Pages.
3. Add cache headers strategy and content hash filenames for JS/CSS where practical.

Deliverable:

- Public GitHub Pages URL serving the full site.

## Phase 5: Automation and Operations

Status: Not started

1. Add publish-dataset.yml for scheduled or manual dataset refresh.
2. Add smoke checks:
- required files available
- minimum city count
- map pages load
- JSON parse validation
3. Add rollback procedure:
- redeploy prior Git commit
- or pin site build to previous dataset manifest version

Deliverable:

- Repeatable data and site release process with basic operational safeguards.

## Build and Release Workflow

## Dataset release workflow

1. Run data collection scripts.
2. Run prepare_dataset script.
3. Run validation.
4. Publish to Hugging Face dataset.
5. Tag dataset version in manifest.

## Site release workflow

1. Trigger GitHub Action.
2. Pull manifest and referenced files from Hugging Face.
3. Build static site.
4. Run smoke tests.
5. Deploy to GitHub Pages.

## Environment and Secrets

GitHub repository secrets:

- HF_TOKEN (write access for dataset publish workflow)

Variables/config:

- HF_DATASET_ID=colerjstevenson/GolfGulf
- SITE_BASE_PATH (if using project Pages path)
- DATASET_VERSION_PIN (optional for reproducible deploys)

Local development token handling:

- Read the Hugging Face token from the local file token for local publish scripts.
- Never print the token in logs.
- Never commit token to git.
- Mirror the same credential in GitHub as HF_TOKEN for CI publishing.

Security hardening tasks:

1. Add token to .gitignore if it is not already ignored.
2. Add a startup check in scripts/publish_dataset.py that fails if token is missing.
3. Prefer environment variable HF_TOKEN in automation and only use local token file as a fallback for local runs.

## Quality Gates

Before each release:

- Schema validation passes.
- No missing city links from landing page.
- Golf course map loads with expected feature count.
- Search works for city names and course names.
- Data links return 200 and parse successfully.

## Risks and Mitigations

- Risk: Large GeoJSON hurts performance.
- Mitigation: Split by city, enable clustering, and lazy-load on demand.

- Risk: Dataset/schema drift breaks site rendering.
- Mitigation: Strict schema validation + manifest version pinning.

- Risk: CORS or transient fetch issues from remote dataset.
- Mitigation: Build-time fetch into site artifacts and deploy static copies.

- Risk: Inconsistent updates between data and pages.
- Mitigation: Single pipeline that promotes dataset and pages together.

## Milestones and Timeline

- Milestone 1 (1-2 days): Schema + validation + data inventory complete.
- Milestone 2 (1-2 days): Hugging Face dataset published with manifest.
- Milestone 3 (2-4 days): Data-driven site pages complete.
- Milestone 4 (1 day): GitHub Pages deployment workflow live.
- Milestone 5 (1 day): Smoke checks, docs, and rollback playbook complete.

## Definition of Done

The project is complete when:

- A public GitHub Pages URL serves the landing page, city pages, visuals links, and global golf map.
- The site consumes data produced from the Hugging Face dataset pipeline.
- Data and site publish workflows are automated in GitHub Actions.
- README documentation explains how to refresh data, publish dataset, and deploy site.

## Immediate Next Actions

1. Publish a fresh dataset snapshot to Hugging Face from the prepared `data/site_dataset` output.
2. Enable GitHub Pages in repository settings (Actions source) so `deploy-pages.yml` can publish.
3. Add tract-level `cities/<city_slug>.geojson` when collection is ready and wire optional city map pages.
4. Add rollback note in README for pinning `dataset_revision` to prior manifest commits.
