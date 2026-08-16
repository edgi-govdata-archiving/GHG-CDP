"""
Generates TopoJSON (state, county) and GeoJSON (facilities) from GHG emissions
output files. Congressional District output is stubbed for v2.

Output structure:
    output_topojson/
    ├── ghg_state.topojson           all years; includes company_violations_2023 (stub)
    ├── ghg_county.topojson          all years
    ├── ghg_cd.topojson              STUB - v2
    ├── parent_company_2023.json     2023 only; violations stub
    └── facilities/
        └── ghg_facilities_{year}.geojson

Dependencies:
    pip install pandas geopandas topojson
"""

import os
import json
import pandas as pd
import geopandas as gpd
import topojson as tp

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# Outputs
OUTPUT_DIR = "output_topojson"
FACILITIES_DIR = os.path.join(OUTPUT_DIR, "facilities")

# Emissions CSVs
EMISSIONS_STATE_CSV = os.path.join("output_files", "aggregated_emissions_by_state.csv")
EMISSIONS_COUNTY_CSV = os.path.join(
    "output_files", "aggregated_emissions_by_county.csv"
)
EMISSIONS_CD_CSV = os.path.join("output_files", "aggregated_emissions_by_cd.csv")
EMISSIONS_FACILITY_CSV = os.path.join(
    "output_files", "aggregated_emissions_by_facility_with_metadata.csv"
)
PARENT_COMPANY_CSV = os.path.join(
    "output_files", "aggregated_emissions_by_parent_company.csv"
)
FACILITY_FRSID_CSV = os.path.join("output_files", "facility_frsid_metadata.csv")

# Geometry
STATE_GEOJSON = os.path.join("input_data", "tl_2025_us_state.json")
COUNTY_GEOJSON = os.path.join("input_data", "tl_2025_us_county_small.json")
CD_GEOJSON = os.path.join("input_data", "cds_119th.json")

# Violation data
# See GHGRP_CAAviolations_2023.ipynb for the join logic.
VIOLATIONS_CSV = None  # e.g. "input_data/ICIS-AIR_VIOLATION_HISTORY-2023.csv"
FRS_LINKS_CSV = None  # Data that links FRS IDs to the ICIS data

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COMPANY_DATA_YEAR = 2023  # Parent company attribution only reliable for this year

# Facility properties to include in GeoJSON output
FACILITY_KEEP_COLS = [
    "Facility Id",
    "Facility Name",
    "Latitude",
    "Longitude",
    "Total Direct Emissions",
    "Total Supplier Emissions",
    "Latest Parent Company",
    "Latest Parent Ownership %",
    "County_FIPS",
    "DISTRICTID",
    "Effective State",
    "FRS Id",
    "Primary NAICS Code",
    "Is_Direct_Emitter",
    "Is_Supplier",
    "Year",
]

# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────


def ensure_output_dirs():
    os.makedirs(FACILITIES_DIR, exist_ok=True)


def pivot_emissions_to_nested(df, id_col):
    """
    Convert long-format emissions data to a nested dict:
        { id_value: { "2010": { total_direct: X, total_supplier: Y }, ... } }

    Only Total Direct Emissions and Total Supplier Emissions are included.
    'Total Emissions' (Direct + Supplier) is intentionally excluded — it is
    an incorrect combined measure for our purposes.
    """
    result = {}
    for _, row in df.iterrows():
        key = str(row[id_col])
        year = str(int(row["Year"]))
        result.setdefault(key, {})[year] = {
            "total_direct": _clean_float(row.get("Total Direct Emissions")),
            "total_supplier": _clean_float(row.get("Total Supplier Emissions")),
        }
    return result


def _clean_float(value):
    """Return float or None; guards against NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def build_geojson_feature(geometry, properties, feature_id=None):
    """Build a single GeoJSON Feature dict."""
    feat = {"type": "Feature", "geometry": geometry, "properties": properties}
    if feature_id is not None:
        feat["id"] = feature_id
    return feat


def write_topojson(geojson_dict, output_path):
    """Convert a GeoJSON FeatureCollection dict to TopoJSON and write to disk."""
    topo = tp.Topology(geojson_dict, prequantize=True)
    topo_json = topo.to_dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(topo_json, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Written: {output_path}  ({size_kb:,.0f} KB)")


def write_geojson(features, output_path):
    """Write a list of GeoJSON Feature dicts to a FeatureCollection file."""
    fc = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Written: {output_path}  ({size_kb:,.0f} KB, {len(features)} features)")


# ─────────────────────────────────────────────────────────────────────────────
# Violation data loading + company aggregation
# ─────────────────────────────────────────────────────────────────────────────


def load_violations_2023():
    """
    Load 2023 ECHO CAA violation data and join to GHGRP facility IDs.

    Returns a DataFrame with columns [Facility Id, frv_count, hpv_count],
    or None if violation data paths are not yet configured.

    TODO: Set VIOLATIONS_CSV and FRS_LINKS_CSV at the top of this file.
          See GHGRP_CAAviolations_2023.ipynb for the full join logic.
    """
    if VIOLATIONS_CSV is None or FRS_LINKS_CSV is None:
        print(
            "  [STUB] VIOLATIONS_CSV / FRS_LINKS_CSV not set — skipping violation data."
        )
        return None

    violations = pd.read_csv(VIOLATIONS_CSV)
    frs_links = pd.read_csv(FRS_LINKS_CSV)
    frsid_meta = pd.read_csv(FACILITY_FRSID_CSV)

    # Join violation events → FRS registry IDs
    merged = violations.merge(
        frs_links[["PGM_SYS_ID", "REGISTRY_ID"]], on="PGM_SYS_ID", how="left"
    )

    # Filter to GHGRP facilities only
    ghgrp_ids = set(frsid_meta["FRS Id"].dropna().unique())
    merged = merged[merged["REGISTRY_ID"].isin(ghgrp_ids)].copy()

    # Count FRV and HPV events per REGISTRY_ID
    counts = (
        merged.groupby(["REGISTRY_ID", "ENF_RESPONSE_POLICY_CODE"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts.rename(columns={"FRV": "frv_count", "HPV": "hpv_count"}, inplace=True)
    for col in ["frv_count", "hpv_count"]:
        if col not in counts.columns:
            counts[col] = 0

    # Map REGISTRY_ID → Facility Id
    id_map = frsid_meta[["FRS Id", "Facility Id"]].rename(
        columns={"FRS Id": "REGISTRY_ID"}
    )
    result = counts.merge(id_map, on="REGISTRY_ID", how="left")

    return result[["Facility Id", "frv_count", "hpv_count"]]


def compute_company_violations_by_state(violations_df, facility_meta_df):
    """
    Aggregate 2023 violation counts per parent company per state, split by
    ownership tier (full = 100%, partial = <100%; null ownership is excluded).

    Violation fields per company entry:
        full_frv_count, partial_frv_count     FRV events by ownership tier
        full_hpv_count, partial_hpv_count     HPV events by ownership tier
        total_violations_full                  full_frv + full_hpv (convenience)
        total_violations_partial               partial_frv + partial_hpv (convenience)

    Returns: { state_abbr: [ {company: str, full_frv_count: int, ...}, ... ] }
             Empty dict if violations_df is None.
    """
    if violations_df is None:
        return {}

    ownership = (
        facility_meta_df[facility_meta_df["Year"] == COMPANY_DATA_YEAR][
            [
                "Facility Id",
                "Latest Parent Company",
                "Latest Parent Ownership %",
                "Effective State",
            ]
        ]
        .dropna(subset=["Latest Parent Ownership %", "Latest Parent Company"])
        .copy()
    )

    merged = ownership.merge(violations_df, on="Facility Id", how="left")
    merged["frv_count"] = merged["frv_count"].fillna(0).astype(int)
    merged["hpv_count"] = merged["hpv_count"].fillna(0).astype(int)

    # Ownership tier classification
    merged["tier"] = merged["Latest Parent Ownership %"].apply(
        lambda x: "full" if float(x) == 100.0 else "partial"
    )

    agg = (
        merged.groupby(["Effective State", "Latest Parent Company", "tier"])[
            ["frv_count", "hpv_count"]
        ]
        .sum()
        .reset_index()
    )

    full = (
        agg[agg["tier"] == "full"]
        .rename(columns={"frv_count": "full_frv_count", "hpv_count": "full_hpv_count"})
        .drop(columns="tier")
    )
    partial = (
        agg[agg["tier"] == "partial"]
        .rename(
            columns={"frv_count": "partial_frv_count", "hpv_count": "partial_hpv_count"}
        )
        .drop(columns="tier")
    )

    company_agg = full.merge(
        partial, on=["Effective State", "Latest Parent Company"], how="outer"
    ).fillna(0)

    int_cols = [
        "full_frv_count",
        "full_hpv_count",
        "partial_frv_count",
        "partial_hpv_count",
    ]
    for col in int_cols:
        if col not in company_agg.columns:
            company_agg[col] = 0
        company_agg[col] = company_agg[col].astype(int)

    company_agg["total_violations_full"] = (
        company_agg["full_frv_count"] + company_agg["full_hpv_count"]
    )
    company_agg["total_violations_partial"] = (
        company_agg["partial_frv_count"] + company_agg["partial_hpv_count"]
    )

    result = {}
    for state, group in company_agg.groupby("Effective State"):
        rows = group.drop(columns="Effective State").rename(
            columns={"Latest Parent Company": "company"}
        )
        result[state] = rows.to_dict(orient="records")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. State TopoJSON
# ─────────────────────────────────────────────────────────────────────────────


def build_state_topojson():
    print("\n[1/5] Building state TopoJSON ...")

    # Emissions
    df = pd.read_csv(EMISSIONS_STATE_CSV)
    emissions = pivot_emissions_to_nested(df, "Effective State")

    # Violation + company data (2023 only)
    facility_meta = pd.read_csv(EMISSIONS_FACILITY_CSV, low_memory=False)
    violations_df = load_violations_2023()
    company_violations = compute_company_violations_by_state(
        violations_df, facility_meta
    )

    # Geometry
    geo = gpd.read_file(STATE_GEOJSON)[["STUSPS", "NAME", "geometry"]]

    features = []
    for _, row in geo.iterrows():
        abbr = row["STUSPS"]
        props = {
            "state_abbr": abbr,
            "state_name": row["NAME"],
            "emissions": emissions.get(abbr, {}),
            # 2023-only. Null until VIOLATIONS_CSV is configured.
            "company_violations_2023": company_violations.get(abbr, None),
            # TODO (v2): "ejscreen": ejscreen_by_state.get(abbr, None)
            # See EJSCREEN_Climate_Data_Merge.ipynb for aggregation method.
        }
        features.append(
            build_geojson_feature(
                geometry=row["geometry"].__geo_interface__,
                properties=props,
                feature_id=abbr,
            )
        )

    write_topojson(
        {"type": "FeatureCollection", "features": features},
        os.path.join(OUTPUT_DIR, "ghg_state.topojson"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. County TopoJSON
# ─────────────────────────────────────────────────────────────────────────────


def build_county_topojson():
    print("\n[2/5] Building county TopoJSON ...")

    df = pd.read_csv(EMISSIONS_COUNTY_CSV)
    df["County_FIPS"] = df["County_FIPS"].astype(str).str.zfill(5)
    emissions = pivot_emissions_to_nested(df, "County_FIPS")

    # Name lookup: one row per FIPS
    name_lookup = (
        df[["County_FIPS", "COUNTYNAME_cleaned", "STATE"]]
        .drop_duplicates("County_FIPS")
        .set_index("County_FIPS")
    )

    geo = gpd.read_file(COUNTY_GEOJSON)[["GEOID", "NAME", "STATEFP", "geometry"]]

    features = []
    for _, row in geo.iterrows():
        fips = row["GEOID"]
        if fips in name_lookup.index:
            county_name = name_lookup.loc[fips, "COUNTYNAME_cleaned"]
            state = name_lookup.loc[fips, "STATE"]
        else:
            county_name = row["NAME"]
            state = None

        props = {
            "county_fips": fips,
            "county_name": county_name,
            "state": state,
            "emissions": emissions.get(fips, {}),
            # TODO (v2): "ejscreen": ejscreen_by_county.get(fips, None)
            # See EJSCREEN_Climate_Data_Merge.ipynb — county-level data exists natively.
        }
        features.append(
            build_geojson_feature(
                geometry=row["geometry"].__geo_interface__,
                properties=props,
                feature_id=fips,
            )
        )

    write_topojson(
        {"type": "FeatureCollection", "features": features},
        os.path.join(OUTPUT_DIR, "ghg_county.topojson"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. CD TopoJSON — STUB (v2)
# ─────────────────────────────────────────────────────────────────────────────


def build_cd_topojson():
    """
    TODO (v2): Congressional district TopoJSON.

    Implementation notes:
    - Source:    aggregated_emissions_by_cd.csv, join key: DISTRICTID (zero-pad to 4)
    - Geometry:  input_data/cds_119th.json — DISTRICTID already stored as "0101" format
    - Extras:    Representative_Name, Representative_Party from emissions CSV
    - EJScreen:  reuse aggregation from EJSCREEN_Climate_Data_CD.ipynb
    - Boundary caveat: 119th Congress geometry is used for all years. Document
      this limitation in the output file's top-level metadata object.
    """
    print("\n[3/5] CD TopoJSON — STUB (v2, skipping)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Facility GeoJSON (one file per year)
# ─────────────────────────────────────────────────────────────────────────────


def build_facility_geojson():
    print("\n[4/5] Building facility GeoJSON files ...")

    # Read the FIPS/district codes as strings. The CSV stores them zero-padded
    # (e.g. "06039"), but if pandas is left to infer the dtype it reads the whole
    # column as float64 — because missing offshore rows introduce NaN — which
    # strips leading zeros and appends ".0" (06039 -> "6039.0").
    df = pd.read_csv(
        EMISSIONS_FACILITY_CSV,
        low_memory=False,
        dtype={"County_FIPS": str, "DISTRICTID": str},
    )

    # Keep only columns that exist in this version of the file
    cols = [c for c in FACILITY_KEEP_COLS if c in df.columns]
    df = df[cols].dropna(subset=["Latitude", "Longitude"]).copy()

    # Normalise ID fields (safety net; values are already zero-padded strings).
    if "County_FIPS" in df.columns:
        df["County_FIPS"] = df["County_FIPS"].str.zfill(5)
    if "DISTRICTID" in df.columns:
        df["DISTRICTID"] = df["DISTRICTID"].str.zfill(4)

    for year in sorted(df["Year"].unique()):
        year_df = df[df["Year"] == year]
        features = []
        for _, row in year_df.iterrows():
            props = {
                k: (None if (isinstance(v, float) and pd.isna(v)) else v)
                for k, v in row.drop(["Latitude", "Longitude"]).items()
            }
            # TODO: Add per-facility violation counts for this year.
            # Requires multi-year ECHO ICIS-AIR bulk data (one CSV per year).
            # When available, join on FRS Id → REGISTRY_ID, count FRV/HPV rows.
            # props["frv_count"] = None
            # props["hpv_count"] = None

            features.append(
                build_geojson_feature(
                    geometry={
                        "type": "Point",
                        "coordinates": [
                            float(row["Longitude"]),
                            float(row["Latitude"]),
                        ],
                    },
                    properties=props,
                )
            )

        write_geojson(
            features,
            os.path.join(FACILITIES_DIR, f"ghg_facilities_{int(year)}.geojson"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parent Company JSON (2023 only)
# ─────────────────────────────────────────────────────────────────────────────


def build_parent_company_json():
    """
    Flat JSON of parent company emissions and violation data for 2023.

    Only 2023 is included because parent company ownership changes over time
    make historical attribution unreliable. The 'violations' field is a stub
    until VIOLATIONS_CSV is configured.
    """
    print("\n[5/5] Building parent company JSON (2023 only) ...")

    df = pd.read_csv(PARENT_COMPANY_CSV)
    df_2023 = df[df["Year"] == COMPANY_DATA_YEAR].copy()

    # Build violation summaries across all states per company
    facility_meta = pd.read_csv(EMISSIONS_FACILITY_CSV, low_memory=False)
    violations_df = load_violations_2023()
    company_violations_by_state = compute_company_violations_by_state(
        violations_df, facility_meta
    )

    # Flatten per-state company violations → per-company totals
    company_violation_totals = {}
    for _state, companies in company_violations_by_state.items():
        for entry in companies:
            name = entry["company"]
            if name not in company_violation_totals:
                company_violation_totals[name] = {
                    "full_frv_count": 0,
                    "full_hpv_count": 0,
                    "partial_frv_count": 0,
                    "partial_hpv_count": 0,
                }
            for field in company_violation_totals[name]:
                company_violation_totals[name][field] += entry.get(field, 0)

    # Add convenience totals
    for name, v in company_violation_totals.items():
        v["total_violations_full"] = v["full_frv_count"] + v["full_hpv_count"]
        v["total_violations_partial"] = v["partial_frv_count"] + v["partial_hpv_count"]

    records = []
    for _, row in df_2023.iterrows():
        company = row["Parent Company"]
        records.append(
            {
                "parent_company": company,
                "year": COMPANY_DATA_YEAR,
                "total_direct_emissions": _clean_float(
                    row.get("Total Direct Emissions")
                ),
                "total_supplier_emissions": _clean_float(
                    row.get("Total Supplier Emissions")
                ),
                "facility_count": int(row["Facility Count"])
                if pd.notna(row.get("Facility Count"))
                else None,
                # Violation data: populated when VIOLATIONS_CSV is configured
                "violations": company_violation_totals.get(company, None),
            }
        )

    output_path = os.path.join(OUTPUT_DIR, "parent_company_2023.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Written: {output_path}  ({size_kb:,.0f} KB, {len(records)} companies)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_output_dirs()
    build_state_topojson()
    build_county_topojson()
    build_cd_topojson()
    build_facility_geojson()
    build_parent_company_json()
    print("\nAll done.")
