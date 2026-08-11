from __future__ import annotations

from pathlib import Path
import importlib


def run_match(combined_csv: Path, postal_csv: Path, golflink_csv: Path, matched_csv: Path) -> dict:
    module = importlib.import_module("legacy.postal_lookup")
    lookup = module.PostalCodeLookup()

    # Ensure postal code file exists with header for append mode.
    postal_csv.parent.mkdir(parents=True, exist_ok=True)
    if not postal_csv.exists():
        postal_csv.write_text("gcid,postal_code\n", encoding="utf-8")

    lookup.add_postal_codes(str(combined_csv), str(postal_csv))
    lookup.greedy_match_by_postal(str(combined_csv), str(golflink_csv), str(matched_csv))

    return {"matched_csv": str(matched_csv)}
