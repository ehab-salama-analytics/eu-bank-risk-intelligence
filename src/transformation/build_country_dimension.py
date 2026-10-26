"""
Build the analytical EBA Country dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5C — Analytical Country Dimension

Inputs:
    data/staging/reference/dim_country.parquet
    data/staging/facts/stg_credit_risk.parquet

Output:
    data/analytical/dimensions/dim_country.parquet

Purpose:
    Build a reusable analytical Country dimension from the validated
    staged EBA country lookup.

Validated staged structure:
    - 359 rows;
    - country is the unique lookup key;
    - columns are country, label, iso_code;
    - 70 country codes are currently used by the credit-risk facts;
    - no credit-risk country codes are missing from the lookup.

The analytical dimension preserves all staged lookup members and adds
a usage flag for the current credit-risk fact population.

This script DOES NOT:
    - modify staged reference data;
    - modify staged fact data;
    - remove unused countries;
    - calculate banking KPIs;
    - join the dimension to fact tables;
    - create surrogate warehouse keys.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_COUNTRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_country.parquet"
)

CREDIT_FACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
    / "stg_credit_risk.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dim_country.parquet"
)


# =============================================================================
# Expected source structure
# =============================================================================

EXPECTED_COLUMNS = [
    "country",
    "label",
    "iso_code",
]


# =============================================================================
# Load sources
# =============================================================================

def load_staged_country() -> pl.DataFrame:
    """
    Load validated staged Country lookup.
    """

    if not STAGED_COUNTRY_FILE.is_file():

        raise FileNotFoundError(
            f"Staged Country lookup not found: "
            f"{STAGED_COUNTRY_FILE}"
        )

    df = pl.read_parquet(
        STAGED_COUNTRY_FILE
    )

    if df.columns != EXPECTED_COLUMNS:

        raise ValueError(
            "Unexpected staged Country schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_credit_country_codes() -> set[int]:
    """
    Return distinct non-null Country codes used by the staged
    credit-risk fact table.
    """

    if not CREDIT_FACT_FILE.is_file():

        raise FileNotFoundError(
            f"Credit-risk staging file not found: "
            f"{CREDIT_FACT_FILE}"
        )

    lazy_df = pl.scan_parquet(
        CREDIT_FACT_FILE
    )

    schema = lazy_df.collect_schema()

    if "country" not in schema.names():

        raise KeyError(
            "stg_credit_risk does not contain 'country'."
        )

    values = (
        lazy_df
        .select(
            pl.col("country")
            .drop_nulls()
            .unique()
        )
        .collect()["country"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_country(
    df: pl.DataFrame,
) -> None:
    """
    Reconfirm the staged Country lookup before analytical build.
    """

    print()
    print("=" * 100)
    print("STAGED COUNTRY SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows : "
        f"{df.height:,}"
    )

    # -------------------------------------------------------------------------
    # Expected row count from validated profiling
    # -------------------------------------------------------------------------

    if df.height != 359:

        raise ValueError(
            f"Expected 359 staged Country rows, "
            f"found {df.height:,}."
        )

    print(
        "[PASS] Staged row count matches validated profile: 359."
    )

    # -------------------------------------------------------------------------
    # Key completeness
    # -------------------------------------------------------------------------

    null_country = df[
        "country"
    ].null_count()

    if null_country != 0:

        raise ValueError(
            f"Country lookup contains "
            f"{null_country:,} NULL key value(s)."
        )

    print(
        "[PASS] country contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Key uniqueness
    # -------------------------------------------------------------------------

    distinct_country = df[
        "country"
    ].n_unique()

    if distinct_country != df.height:

        raise ValueError(
            "Country lookup key is not unique."
        )

    print(
        f"[PASS] country is unique: "
        f"{distinct_country:,} distinct code(s)."
    )

    # -------------------------------------------------------------------------
    # Label completeness
    # -------------------------------------------------------------------------

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:

        raise ValueError(
            f"Country lookup contains "
            f"{null_labels:,} NULL label(s)."
        )

    print(
        "[PASS] label contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # ISO-code information
    #
    # We report missing ISO codes rather than failing automatically,
    # because special EBA categories may legitimately not map to a
    # conventional ISO country code.
    # -------------------------------------------------------------------------

    null_iso = df[
        "iso_code"
    ].null_count()

    blank_iso = (
        df
        .filter(
            pl.col("iso_code")
            .is_not_null()
            &
            (
                pl.col("iso_code")
                .str.strip_chars()
                == ""
            )
        )
        .height
    )

    print(
        f"[INFO] NULL iso_code values  : "
        f"{null_iso:,}"
    )

    print(
        f"[INFO] Blank iso_code values : "
        f"{blank_iso:,}"
    )


# =============================================================================
# Build analytical dimension
# =============================================================================

def build_analytical_dimension(
    staged: pl.DataFrame,
    credit_codes: set[int],
) -> pl.DataFrame:
    """
    Preserve all staged country lookup members and add current-fact usage
    metadata.
    """

    analytical = (
        staged
        .with_columns(
            pl.col("country")
            .is_in(
                list(
                    credit_codes
                )
            )
            .alias(
                "used_in_credit_risk"
            )
        )
        .with_columns(
            pl.when(
                pl.col(
                    "used_in_credit_risk"
                )
            )
            .then(
                pl.lit(
                    "credit_risk"
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
            "country"
        )
    )

    return analytical


# =============================================================================
# Analytical validation
# =============================================================================

def validate_analytical_dimension(
    df: pl.DataFrame,
    staged: pl.DataFrame,
    credit_codes: set[int],
) -> None:
    """
    Validate analytical Country dimension before save.
    """

    print()
    print("=" * 100)
    print("ANALYTICAL COUNTRY DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    # -------------------------------------------------------------------------

    if df.height != staged.height:

        raise ValueError(
            f"Analytical Country row count mismatch. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} Country lookup members preserved."
    )

    # -------------------------------------------------------------------------
    # Key uniqueness
    # -------------------------------------------------------------------------

    if df[
        "country"
    ].n_unique() != df.height:

        raise ValueError(
            "Analytical Country key is not unique."
        )

    print(
        "[PASS] country remains unique."
    )

    # -------------------------------------------------------------------------
    # Key completeness
    # -------------------------------------------------------------------------

    if df[
        "country"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Country dimension contains NULL keys."
        )

    print(
        "[PASS] country contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Fact coverage
    # -------------------------------------------------------------------------

    dimension_codes = {
        int(value)
        for value in df[
            "country"
        ].to_list()
    }

    missing_credit = sorted(
        credit_codes
        - dimension_codes
    )

    if missing_credit:

        raise ValueError(
            "Credit-risk Country coverage failed. "
            f"Missing: {missing_credit}"
        )

    print(
        f"[PASS] All {len(credit_codes):,} credit-risk "
        "Country codes are represented."
    )

    # -------------------------------------------------------------------------
    # Usage flag reconciliation
    # -------------------------------------------------------------------------

    flagged_credit = {
        int(value)
        for value in (
            df
            .filter(
                pl.col(
                    "used_in_credit_risk"
                )
            )[
                "country"
            ]
            .to_list()
        )
    }

    if flagged_credit != credit_codes:

        raise ValueError(
            "Credit-risk Country usage flags do not reconcile "
            "to fact codes."
        )

    print(
        "[PASS] Credit-risk usage flags reconcile."
    )

    # -------------------------------------------------------------------------
    # Expected used/unused counts from profiling
    # -------------------------------------------------------------------------

    used_count = (
        df
        .filter(
            pl.col(
                "used_in_credit_risk"
            )
        )
        .height
    )

    unused_count = (
        df.height
        - used_count
    )

    if used_count != 70:

        raise ValueError(
            f"Expected 70 Country codes currently used "
            f"by credit risk, found {used_count:,}."
        )

    if unused_count != 289:

        raise ValueError(
            f"Expected 289 staged Country codes currently unused, "
            f"found {unused_count:,}."
        )

    print(
        "[PASS] Current usage population reconciles: "
        "70 used / 289 unused."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save analytical Country dimension.
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
        "Saved analytical Country dimension:"
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
    Re-open persisted analytical Country dimension.
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
            f"Saved Country row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "country"
    ].n_unique() != saved.height:

        raise ValueError(
            "Saved analytical Country key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows             : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Country codes : "
        f"{saved['country'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "country_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Country usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Country sample:"
    )

    print(
        saved
        .head(
            15
        )
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build analytical Country dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL COUNTRY DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Country : "
        f"{STAGED_COUNTRY_FILE}"
    )

    print(
        f"Credit fact    : "
        f"{CREDIT_FACT_FILE}"
    )

    print(
        f"Output         : "
        f"{OUTPUT_FILE}"
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    staged = (
        load_staged_country()
    )

    credit_codes = (
        load_credit_country_codes()
    )

    # -------------------------------------------------------------------------
    # Validate source
    # -------------------------------------------------------------------------

    validate_staged_country(
        staged
    )

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    analytical = (
        build_analytical_dimension(
            staged=staged,
            credit_codes=credit_codes,
        )
    )

    # -------------------------------------------------------------------------
    # Validate analytical result
    # -------------------------------------------------------------------------

    validate_analytical_dimension(
        df=analytical,
        staged=staged,
        credit_codes=credit_codes,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_dimension(
        analytical
    )

    # -------------------------------------------------------------------------
    # Post-save validation
    # -------------------------------------------------------------------------

    validate_saved_dimension(
        expected_rows=staged.height
    )

    print()
    print("=" * 100)
    print(
        "ANALYTICAL COUNTRY DIMENSION BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
