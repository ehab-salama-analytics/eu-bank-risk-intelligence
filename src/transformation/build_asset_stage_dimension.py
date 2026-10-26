"""
Build the analytical EBA Asset Stage dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5H — Analytical Asset Stage Dimension

Inputs:
    data/staging/reference/dim_asset_stage.parquet
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/dimensions/dim_asset_stage.parquet

Purpose:
    Build a reusable analytical Asset Stage dimension from the validated
    staged EBA lookup.

Validated source characteristics:
    - 4 staged Asset Stage rows;
    - assets_stages is unique;
    - no NULL assets_stages keys;
    - all 4 Asset Stage codes are currently used by the other-template fact;
    - no fact Asset Stage codes are missing from the lookup.

This script preserves every staged lookup member and adds current
fact-usage metadata.

This script DOES NOT:
    - modify staging data;
    - modify fact data;
    - infer IFRS 9 classifications;
    - calculate Stage migration;
    - calculate impairment ratios;
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

STAGED_ASSET_STAGE_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_asset_stage.parquet"
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
    / "dim_asset_stage.parquet"
)


# =============================================================================
# Expected staged structure
# =============================================================================

EXPECTED_COLUMNS = [
    "assets_stages",
    "label",
]


# =============================================================================
# Loading
# =============================================================================

def load_staged_asset_stage() -> pl.DataFrame:
    """
    Load the validated staged Asset Stage lookup.
    """

    if not STAGED_ASSET_STAGE_FILE.is_file():

        raise FileNotFoundError(
            f"Staged Asset Stage lookup not found: "
            f"{STAGED_ASSET_STAGE_FILE}"
        )

    df = pl.read_parquet(
        STAGED_ASSET_STAGE_FILE
    )

    if df.columns != EXPECTED_COLUMNS:

        raise ValueError(
            "Unexpected staged Asset Stage schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_other_asset_stage_codes() -> set[int]:
    """
    Return distinct non-null assets_stages codes used by
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

    if "assets_stages" not in schema.names():

        raise KeyError(
            "stg_other_templates does not contain 'assets_stages'."
        )

    values = (
        lazy_df
        .select(
            pl.col("assets_stages")
            .drop_nulls()
            .unique()
        )
        .collect()["assets_stages"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_asset_stage(
    df: pl.DataFrame,
) -> None:
    """
    Validate staged Asset Stage lookup.
    """

    print()
    print("=" * 100)
    print("STAGED ASSET STAGE SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows : {df.height:,}"
    )

    # -------------------------------------------------------------------------
    # Expected row count
    # -------------------------------------------------------------------------

    if df.height != 4:

        raise ValueError(
            f"Expected 4 staged Asset Stage rows, "
            f"found {df.height:,}."
        )

    print(
        "[PASS] Staged row count matches validated profile: 4."
    )

    # -------------------------------------------------------------------------
    # Key completeness
    # -------------------------------------------------------------------------

    null_keys = df[
        "assets_stages"
    ].null_count()

    if null_keys != 0:

        raise ValueError(
            f"Asset Stage lookup contains "
            f"{null_keys:,} NULL key value(s)."
        )

    print(
        "[PASS] assets_stages contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Key uniqueness
    # -------------------------------------------------------------------------

    distinct_keys = df[
        "assets_stages"
    ].n_unique()

    if distinct_keys != df.height:

        raise ValueError(
            "Asset Stage lookup key is not unique."
        )

    print(
        f"[PASS] assets_stages is unique: "
        f"{distinct_keys:,} distinct code(s)."
    )

    # -------------------------------------------------------------------------
    # Label completeness
    # -------------------------------------------------------------------------

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:

        raise ValueError(
            f"Asset Stage lookup contains "
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
    Preserve all Asset Stage members and add current fact-usage metadata.
    """

    return (
        staged
        .with_columns(
            pl.col("assets_stages")
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
            "assets_stages"
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
    Validate analytical Asset Stage dimension.
    """

    print()
    print("=" * 100)
    print("ANALYTICAL ASSET STAGE DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    # -------------------------------------------------------------------------

    if df.height != staged.height:

        raise ValueError(
            f"Analytical Asset Stage row count mismatch. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} Asset Stage lookup members preserved."
    )

    # -------------------------------------------------------------------------
    # Key integrity
    # -------------------------------------------------------------------------

    if df[
        "assets_stages"
    ].n_unique() != df.height:

        raise ValueError(
            "Analytical Asset Stage key is not unique."
        )

    print(
        "[PASS] assets_stages remains unique."
    )

    if df[
        "assets_stages"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Asset Stage contains NULL keys."
        )

    print(
        "[PASS] assets_stages contains 0 NULL values."
    )

    if df[
        "label"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Asset Stage contains NULL labels."
        )

    print(
        "[PASS] label contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Fact coverage
    # -------------------------------------------------------------------------

    dimension_codes = {
        int(value)
        for value in df[
            "assets_stages"
        ].to_list()
    }

    missing_other = sorted(
        other_codes
        - dimension_codes
    )

    if missing_other:

        raise ValueError(
            "Other-template Asset Stage coverage failed. "
            f"Missing: {missing_other}"
        )

    print(
        f"[PASS] All {len(other_codes):,} other-template "
        "Asset Stage codes are represented."
    )

    # -------------------------------------------------------------------------
    # Usage reconciliation
    # -------------------------------------------------------------------------

    flagged_codes = {
        int(value)
        for value in (
            df
            .filter(
                pl.col(
                    "used_in_other_templates"
                )
            )[
                "assets_stages"
            ]
            .to_list()
        )
    }

    if flagged_codes != other_codes:

        raise ValueError(
            "Asset Stage usage flags do not reconcile "
            "to other-template fact codes."
        )

    print(
        "[PASS] Other-template usage flags reconcile."
    )

    # -------------------------------------------------------------------------
    # Full-domain reconciliation
    # -------------------------------------------------------------------------

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
            f"Expected all 4 Asset Stage codes to be currently used, "
            f"found {used_count:,}."
        )

    if unused_count != 0:

        raise ValueError(
            f"Expected 0 unused Asset Stage codes, "
            f"found {unused_count:,}."
        )

    if dimension_codes != other_codes:

        raise ValueError(
            "Asset Stage dimension population does not exactly match "
            "the current other-template fact domain."
        )

    print(
        "[PASS] Full Asset Stage domain reconciliation: "
        "4 used / 0 unused."
    )

    print(
        "[PASS] Analytical Asset Stage population exactly matches "
        "the current other-template fact domain."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save analytical Asset Stage dimension.
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
        "Saved analytical Asset Stage dimension:"
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
    Re-read persisted analytical Asset Stage dimension.
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
            f"Saved Asset Stage row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "assets_stages"
    ].n_unique() != saved.height:

        raise ValueError(
            "Saved analytical Asset Stage key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows                : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Asset Stage codes: "
        f"{saved['assets_stages'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "asset_stage_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Asset Stage usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Asset Stage dimension:"
    )

    print(
        saved
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build analytical Asset Stage dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL ASSET STAGE DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Asset Stage : "
        f"{STAGED_ASSET_STAGE_FILE}"
    )

    print(
        f"Other fact         : "
        f"{OTHER_FACT_FILE}"
    )

    print(
        f"Output             : "
        f"{OUTPUT_FILE}"
    )

    staged = (
        load_staged_asset_stage()
    )

    other_codes = (
        load_other_asset_stage_codes()
    )

    validate_staged_asset_stage(
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
        "ANALYTICAL ASSET STAGE DIMENSION BUILD "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
