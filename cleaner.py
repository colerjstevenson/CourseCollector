from pathlib import Path
import sys
import json
import pandas as pd
import geopandas as gpd

#!/usr/bin/env python3
"""
cleaner.py

Combine all CSV and GeoJSON files from data/ into a single CSV and a single GeoJSON.
- Ensures first column is "gcid" and second is "name" (attempts to map common variants).
- Drops columns that are mostly empty after combining (default threshold: 50% empty).
- Outputs: data/combined.csv and data/combined.geojson
"""



# geopandas is optional but preferred for GeoJSON; fall back to manual merge if absent
try:
    has_gpd = True
except Exception:
    has_gpd = False

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_OUTPUT = DATA_DIR / "combined.csv"
GEOJSON_OUTPUT = DATA_DIR / "combined.geojson"

# Column name candidates for mapping to gcid and name
GCID_CANDIDATES = ["gcid", "id", "gid", "uuid", "global_id", "objectid"]
NAME_CANDIDATES = ["name", "title", "label", "placename", "site_name", "display_name"]


def normalize_cols(columns):
    """Return normalized column names (strip, lower, replace spaces with underscores)."""
    def norm(c):
        if c is None:
            return ""
        c = str(c).strip().lower()
        c = c.replace(" ", "_").replace("-", "_")
        return c
    return [norm(c) for c in columns]


def find_best_col(cols, candidates):
    """Return the first matching column name from candidates in cols, or None."""
    lc = {c: c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
    # try fuzzy: remove underscores
    simplified = {c.replace("_", ""): c for c in cols}
    for cand in candidates:
        key = cand.replace("_", "")
        if key in simplified:
            return simplified[key]
    return None


def drop_sparse_columns(df, thresh=0.2):
    """
    Instead of dropping mostly-empty columns, reorder columns so that
    "sparse" columns (those with < thresh fraction of non-empty values)
    appear first, followed by the remaining columns sorted
    alphabetically. The geometry column (if present) is preserved at the end.
    thresh: fraction of values that must be non-empty to be considered "dense"
    For string columns treat empty string and whitespace as missing.
    """
    n = len(df)
    if n == 0:
        return df

    geom_name = df.geometry.name if hasattr(df, "geometry") else None
    sparse_cols = []
    dense_cols = []

    for col in df.columns:
        # skip geometry for now; append it at the end unchanged
        if geom_name and col == geom_name:
            continue
        s = df[col]
        if pd.api.types.is_string_dtype(s.dtype):
            non_empty = s.dropna().map(lambda v: str(v).strip() != "").sum()
        else:
            non_empty = s.count()  # non-null
        if non_empty / n >= thresh:
            dense_cols.append(col)
        else:
            sparse_cols.append(col)

    # keep sparse columns in their original order, sort dense columns alphabetically
    dense_cols_sorted = sorted(dense_cols, key=lambda x: x.lower())
    new_order = sparse_cols + dense_cols_sorted
    if geom_name and geom_name in df.columns:
        new_order.append(geom_name)

    # Reindex to the new column order
    return df.loc[:, new_order]


def combine_csvs(data_dir, out_path, sparsity_threshold=0.2):
    csv_files = list(Path(data_dir).glob("**/*.csv"))
    if not csv_files:
        print("No CSV files found in", data_dir)
        return

    dfs = []
    for p in csv_files:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
        except Exception as e:
            print(f"Failed to read {p}: {e}", file=sys.stderr)
            continue
        # normalize columns
        orig_cols = list(df.columns)
        norm = normalize_cols(orig_cols)
        col_map = {o: n for o, n in zip(orig_cols, norm)}
        df = df.rename(columns=col_map)
        cols = list(df.columns)

        # Ensure gcid and name columns exist; try to map common variants
        gcid_col = find_best_col(cols, GCID_CANDIDATES)
        name_col = find_best_col(cols, NAME_CANDIDATES)

        if gcid_col and gcid_col != "gcid":
            df = df.rename(columns={gcid_col: "gcid"})
        elif not gcid_col:
            # create empty gcid to keep shape; will be NaN
            df["gcid"] = pd.NA

        if name_col and name_col != "name":
            df = df.rename(columns={name_col: "name"})
        elif not name_col:
            df["name"] = pd.NA

        # Move gcid and name to front
        cols_now = [c for c in df.columns if c not in ("gcid", "name", "province", 'lat', 'lon', 'area_m2')]
        new_order = ["gcid", "name", "province", 'lat', 'lon', 'area_m2'] + sorted(cols_now)
        df = df.reindex(columns=new_order)

        dfs.append(df)

    if not dfs:
        print("No CSV content to combine.", file=sys.stderr)
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # Normalize gcid and name to string
    combined["gcid"] = combined["gcid"].astype(str).replace({"nan": pd.NA})
    combined["name"] = combined["name"].astype(str).replace({"nan": pd.NA})


    # Ensure output dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Wrote combined CSV to {out_path} ({combined.shape[0]} rows, {combined.shape[1]} cols)")


def read_geojson_features(path):
    """Read a geojson file and return list of features (as dicts)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "features" in data:
        return data["features"]
    # If it's a single Feature, wrap it
    if data.get("type", "").lower() == "feature":
        return [data]
    raise ValueError("Unsupported GeoJSON structure in " + str(path))


def combine_geojsons(data_dir, out_path, sparsity_threshold=0.2):
    geo_files = list(Path(data_dir).glob("**/*.geojson")) + list(Path(data_dir).glob("**/*.json"))
    geo_files = [p for p in geo_files if p.is_file()]
    if not geo_files:
        print("No GeoJSON files found in", data_dir)
        return

    # Prefer geopandas if available
    if has_gpd:
        gdfs = []
        for p in geo_files:
            try:
                gdf = gpd.read_file(p)
            except Exception as e:
                print(f"geopandas failed to read {p}: {e}", file=sys.stderr)
                # fallback to manual read
                try:
                    feats = read_geojson_features(p)
                    tmp = gpd.GeoDataFrame.from_features(feats)
                    gdfs.append(tmp)
                    continue
                except Exception as e2:
                    print(f"Failed to parse {p}: {e2}", file=sys.stderr)
                    continue
            # normalize columns
            orig_cols = list(gdf.columns)
            # keep geometry name separate
            geom_name = gdf.geometry.name if hasattr(gdf, "geometry") else None
            cols_non_geom = [c for c in orig_cols if c != geom_name]
            norm = normalize_cols(cols_non_geom)
            col_map = {o: n for o, n in zip(cols_non_geom, norm)}
            gdf = gdf.rename(columns=col_map)
            # map gcid/name
            cols = list(gdf.columns)
            gcid_col = find_best_col(cols, GCID_CANDIDATES)
            name_col = find_best_col(cols, NAME_CANDIDATES)
            if gcid_col and gcid_col != "gcid":
                gdf = gdf.rename(columns={gcid_col: "gcid"})
            elif not gcid_col:
                gdf["gcid"] = pd.NA
            if name_col and name_col != "name":
                gdf = gdf.rename(columns={name_col: "name"})
            elif not name_col:
                gdf["name"] = pd.NA
            # reorder: gcid, name, others..., geometry at end
            cols_now = [c for c in gdf.columns if c not in ("gcid", "name")]
            # ensure geometry remains geometry column
            if geom_name and geom_name in cols_now:
                cols_now.remove(geom_name)
                new_order = ["gcid", "name"] + cols_now + [geom_name]
            else:
                new_order = ["gcid", "name"] + cols_now
            # some columns may be missing - reindex safely
            for c in new_order:
                if c not in gdf.columns:
                    gdf[c] = pd.NA
            gdf = gdf.reindex(columns=new_order)
            gdfs.append(gdf)

        if not gdfs:
            print("No GeoDataFrames to combine.", file=sys.stderr)
            return

        combined_gdf = pd.concat(gdfs, ignore_index=True, sort=False)
        # ensure geometry column
        if hasattr(combined_gdf, "geometry"):
            # Drop sparse columns excluding geometry
            combined_gdf = drop_sparse_columns(combined_gdf, thresh=sparsity_threshold)
            # Ensure geometry column exists and is named "geometry"
            if "geometry" not in combined_gdf.columns:
                # try to find a geometry-like column
                geoms = [c for c in combined_gdf.columns if combined_gdf[c].dtype.name == "geometry"]
                if geoms:
                    combined_gdf = combined_gdf.set_geometry(geoms[0])
            # write
            out_path.parent.mkdir(parents=True, exist_ok=True)
            combined_gdf.to_file(out_path, driver="GeoJSON")
            print(f"Wrote combined GeoJSON to {out_path} ({len(combined_gdf)} features, {len(combined_gdf.columns)} props)")
            return
        else:
            # fallback to manual feature merge below
            pass

    # Manual GeoJSON merge (no geopandas)
    all_features = []
    prop_keys = set()
    for p in geo_files:
        try:
            feats = read_geojson_features(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}", file=sys.stderr)
            continue
        for f in feats:
            props = f.get("properties", {}) or {}
            # normalize property keys
            new_props = {}
            for k, v in props.items():
                nk = k.strip().lower().replace(" ", "_").replace("-", "_")
                new_props[nk] = v
                prop_keys.add(nk)
            # attempt to map gcid/name
            if "gcid" not in new_props:
                gcid_k = find_best_col(list(new_props.keys()), GCID_CANDIDATES)
                if gcid_k:
                    new_props["gcid"] = new_props.pop(gcid_k)
                else:
                    new_props.setdefault("gcid", None)
            if "name" not in new_props:
                name_k = find_best_col(list(new_props.keys()), NAME_CANDIDATES)
                if name_k:
                    new_props["name"] = new_props.pop(name_k)
                else:
                    new_props.setdefault("name", None)
            f["properties"] = new_props
            all_features.append(f)

    if not all_features:
        print("No features collected.", file=sys.stderr)
        return

    # Build properties dataframe to compute sparsity
    props_list = [f.get("properties", {}) for f in all_features]
    props_df = pd.DataFrame(props_list)
    props_df = props_df.astype(object)
    # Use drop_sparse_columns to compute desired ordering (sparse-first, dense alpha)
    props_df = drop_sparse_columns(props_df, thresh=sparsity_threshold)

    # Keep ordered list of keys to preserve column ordering in output
    keep_keys = list(props_df.columns)
    combined = {"type": "FeatureCollection", "features": []}
    for i, f in enumerate(all_features):
        geom = f.get("geometry")
        props = f.get("properties", {})
        # Preserve ordering: iterate keep_keys and pick from props if present
        filtered_props = {k: props.get(k) for k in keep_keys if k in props}
        combined["features"].append({"type": "Feature", "geometry": geom, "properties": filtered_props})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Wrote combined GeoJSON to {out_path} ({len(combined['features'])} features, {len(keep_keys)} props)")


def main():
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f"Data directory not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    combine_csvs(DATA_DIR, CSV_OUTPUT)
    combine_geojsons(DATA_DIR, GEOJSON_OUTPUT)


if __name__ == "__main__":
    main()