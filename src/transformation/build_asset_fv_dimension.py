"""
Build the analytical EBA Asset Fair Value dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5I — Analytical Asset Fair Value Dimension

Inputs:
    data/staging/reference/dim_asset_fv.parquet
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/dimensions/dim_asset_fv.parquet

Purpose:
    Build a reusable analytical Asset Fair Value dimension from the
    validated staged EBA lookup.

Validated source characteristics:
    - 4 staged Asset Fair Value rows;
    - assets_fv is unique;
    - no NULL assets_fv keys;
    - all 4 Asset Fair Value codes are currently used by the
      other-template fact;
    - no fact Asset Fair Value codes are missing from the lookup.

This script preserves every staged lookup member and adds current
fact-usage metadata.

This script DOES NOT:
    - modify staging data;
    - modify fact data;
    - infer fair-value classifications;
    - calculate valuation risk metrics;
    - calculate banking KPIs;
    - join the dimension to fact tables.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_ASSET_FV_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_asset_fv.parquet"
)

OTHER_FACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
    / "stg_other_templates.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dim_asset_fv.parquet"
)


# =============================================================================
# Expected staged structure
# =============================================================================

EXPECTED_COLUMNS = [
    "assets_fv",
    "label",
]


# =============================================================================
# Loading
# =============================================================================

def load_staged_asset_fv() -> pl.DataFrame:
    """
    Load the validated staged Asset Fair Value lookup.
    """

    if not STAGED_ASSET_FV_FILE.is_file():

        raise FileNotFoundError(
            f"Staged Asset Fair Value lookup not found: "
            f"{STAGED_ASSET_FV_FILE}"
        )

    df = pl.read_parquet(
        STAGED_ASSET_FV_FILE
    )

    if df.columns != EXPECTED_COLUMNS:

        raise ValueError(
            "Unexpected staged Asset Fair Value schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_other_asset_fv_codes() -> set[int]:
    """
    Return distinct non-null assets_fv codes used by
    stg_other_templates.
    """

    if not OTHER_FACT_FILE.is_file():

        raise FileNotFoundError(
            f"Other-template staging file not found: "
            f"{OTHER_FACT_FILE}"
        )

    lazy_df = pl.scan_parquet(
        OTHER_FACT_FILE
    )

    schema = lazy_df.collect_schema()

    if "assets_fv" not in schema.names():

        raise KeyError(
            "stg_other_templates does not contain 'assets_fv'."
        )

    values = (
        lazy_df
        .select(
            pl.col("assets_fv")
            .drop_nulls()
            .unique()
        )
        .collect()["assets_fv"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_asset_fv(
    df: pl.DataFrame,
) -> None:
    """
    Validate staged Asset Fair Value lookup.
    """

    print()
    print("=" * 100)
    print("STAGED ASSET FAIR VALUE SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows : {df.height:,}"
    )

    if df.height != 4:

        raise ValueError(
            f"Expected 4 staged Asset Fair Value rows, "
            f"found {df.height:,}."
        )

    print(
        "[PASS] Staged row count matches validated profile: 4."
    )

    null_keys = df[
        "assets_fv"
    ].null_count()

    if null_keys != 0:

        raise ValueError(
            f"Asset Fair Value lookup contains "
            f"{null_keys:,} NULL key value(s)."
        )

    print(
        "[PASS] assets_fv contains 0 NULL values."
    )

    distinct_keys = df[
        "assets_fv"
    ].n_unique()

    if distinct_keys != df.height:

        raise ValueError(
            "Asset Fair Value lookup key is not unique."
        )

    print(
        f"[PASS] assets_fv is unique: "
        f"{distinct_keys:,} distinct code(s)."
    )

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:

        raise ValueError(
            f"Asset Fair Value lookup contains "
            f"{null_labels:,} NULL label(s)."
        )

    print(
        "[PASS] label contains 0 NULL values."
    )


# =============================================================================
# Build analytical dimension
# =============================================================================

def build_analytical_dimension(
    staged: pl.DataFrame,
    other_codes: set[int],
) -> pl.DataFrame:
    """
    Preserve all Asset Fair Value members and add current fact-usage
    metadata.
    """

    return (
        staged
        .with_columns(
            pl.col("assets_fv")
            .is_in(
                list(
                    other_codes
                )
            )
            .alias(
                "used_in_other_templates"
            )
        )
        .with_columns(
            pl.when(
                pl.col(
                    "used_in_other_templates"
                )
            )
            .then(
                pl.lit(
                    "other_templates"
                )
            )
            .otherwise(
                pl.lit(
                    "not_used_in_current_facts"
                )
            )
            .alias(
                "fact_usage"
            )
        )
        .sort(
            "assets_fv"
        )
    )


# =============================================================================
# Analytical validation
# =============================================================================

def validate_analytical_dimension(
    df: pl.DataFrame,
    staged: pl.DataFrame,
    other_codes: set[int],
) -> None:
    """
    Validate analytical Asset Fair Value dimension.
    """

    print()
    print("=" * 100)
    print("ANALYTICAL ASSET FAIR VALUE DIMENSION VALIDATION")
    print("=" * 100)

    if df.height != staged.height:

        raise ValueError(
            f"Analytical Asset Fair Value row count mismatch. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} Asset Fair Value lookup members preserved."
    )

    if df[
        "assets_fv"
    ].n_unique() != df.height:

        raise ValueError(
            "Analytical Asset Fair Value key is not unique."
        )

    print(
        "[PASS] assets_fv remains unique."
    )

    if df[
        "assets_fv"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Asset Fair Value contains NULL keys."
        )

    print(
        "[PASS] assets_fv contains 0 NULL values."
    )

    if df[
        "label"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Asset Fair Value contains NULL labels."
        )

    print(
        "[PASS] label contains 0 NULL values."
    )

    dimension_codes = {
        int(value)
        for value in df[
            "assets_fv"
        ].to_list()
    }

    missing_other = sorted(
        other_codes
        - dimension_codes
    )

    if missing_other:

        raise ValueError(
            "Other-template Asset Fair Value coverage failed. "
            f"Missing: {missing_other}"
        )

    print(
        f"[PASS] All {len(other_codes):,} other-template "
        "Asset Fair Value codes are represented."
    )

    flagged_codes = {
        int(value)
        for value in (
            df
            .filter(
                pl.col(
                    "used_in_other_templates"
                )
            )[
                "assets_fv"
            ]
            .to_list()
        )
    }

    if flagged_codes != other_codes:

        raise ValueError(
            "Asset Fair Value usage flags do not reconcile "
            "to other-template fact codes."
        )

    print(
        "[PASS] Other-template usage flags reconcile."
    )

    used_count = (
        df
        .filter(
            pl.col(
                "used_in_other_templates"
            )
        )
        .height
    )

    unused_count = (
        df.height
        - used_count
    )

    if used_count != 4:

        raise ValueError(
            f"Expected all 4 Asset Fair Value codes to be currently used, "
            f"found {used_count:,}."
        )

    if unused_count != 0:

        raise ValueError(
            f"Expected 0 unused Asset Fair Value codes, "
            f"found {unused_count:,}."
        )

    if dimension_codes != other_codes:

        raise ValueError(
            "Asset Fair Value dimension population does not exactly match "
            "the current other-template fact domain."
        )

    print(
        "[PASS] Full Asset Fair Value domain reconciliation: "
        "4 used / 0 unused."
    )

    print(
        "[PASS] Analytical Asset Fair Value population exactly matches "
        "the current other-template fact domain."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save analytical Asset Fair Value dimension.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    print()
    print(
        "Saved analytical Asset Fair Value dimension:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


# =============================================================================
# Post-save validation
# =============================================================================

def validate_saved_dimension(
    expected_rows: int,
) -> None:
    """
    Re-read persisted analytical Asset Fair Value dimension.
    """

    if not OUTPUT_FILE.is_file():

        raise FileNotFoundError(
            f"Expected output not found: "
            f"{OUTPUT_FILE}"
        )

    saved = pl.read_parquet(
        OUTPUT_FILE
    )

    if saved.height != expected_rows:

        raise ValueError(
            f"Saved Asset Fair Value row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "assets_fv"
    ].n_unique() != saved.height:

        raise ValueError(
            "Saved analytical Asset Fair Value key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows                     : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Asset Fair Value codes: "
        f"{saved['assets_fv'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "asset_fv_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Asset Fair Value usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Asset Fair Value dimension:"
    )

    print(
        saved
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build analytical Asset Fair Value dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL ASSET FAIR VALUE DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Asset Fair Value : "
        f"{STAGED_ASSET_FV_FILE}"
    )

    print(
        f"Other fact              : "
        f"{OTHER_FACT_FILE}"
    )

    print(
        f"Output                  : "
        f"{OUTPUT_FILE}"
    )

    staged = (
        load_staged_asset_fv()
    )

    other_codes = (
        load_other_asset_fv_codes()
    )

    validate_staged_asset_fv(
        staged
    )

    analytical = (
        build_analytical_dimension(
            staged=staged,
            other_codes=other_codes,
        )
    )

    validate_analytical_dimension(
        df=analytical,
        staged=staged,
        other_codes=other_codes,
    )

    save_dimension(
        analytical
    )

    validate_saved_dimension(
        expected_rows=staged.height
    )

    print()
    print("=" * 100)
    print(
        "ANALYTICAL ASSET FAIR VALUE DIMENSION BUILD "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
