from __future__ import annotations

import importlib
from pathlib import Path


def run_scrape(enable_golflink: bool, enable_golfcanada: bool, enable_golfdigest: bool) -> dict:
    result = {"golflink": False, "golfcanada": False, "golfdigest": False}

    if enable_golflink:
        gl = importlib.import_module("legacy.golfLinkScrapper")
        all_rows = []
        for sitemap in gl.SITEMAP_URLS:
            print(f"Processing GolfLink sitemap: {sitemap}")
            all_rows.extend(gl.scrape_all(sitemap))
        gl.save_results(all_rows, json_path="data/golfLinkData.json", csv_path="data/golfLinkData.csv")
        result["golflink"] = True

    if enable_golfcanada:
        gc = importlib.import_module("legacy.golfCanadaScrapper")
        all_rows = []
        for sitemap in gc.SITEMAP_URLS:
            print(f"Processing Golf Canada sitemap: {sitemap}")
            all_rows.extend(gc.scrape_all(sitemap))
        # Keep historical behavior for now (append mode in module writer).
        gc.save_results(all_rows, json_path="data/golf_canada_data_full.json", csv_path="data/golf_canada_data_full.csv")
        result["golfcanada"] = True

    if enable_golfdigest:
        gd = importlib.import_module("legacy.golfdigest_urls")
        gd.main()
        result["golfdigest"] = True

    return result
