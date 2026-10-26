"""
Build the analytical EBA Status dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5E — Analytical Status Dimension

Inputs:
    data/staging/reference/dim_status.parquet
    data/staging/facts/stg_credit_risk.parquet

Output:
    data/analytical/dimensions/dim_status.parquet

Purpose:
    Build a reusable analytical Status dimension from the validated
    staged EBA lookup.

Validated source characteristics:
    - 5 staged Status rows;
    - status is unique;
    - no NULL status keys;
    - 2 Status codes are currently used by the credit-risk fact;
    - no fact Status codes are missing from the lookup.

Important:
    status is analytically meaningful even though previous grain-minimality
    testing showed that it is not required for technical uniqueness in the
    current 2025 credit-risk staging dataset.

This script DOES NOT:
    - modify staging data;
    - remove unused lookup members;
    - join the dimension to facts;
    - calculate risk KPIs;
    - create surrogate warehouse keys.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_STATUS_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_status.parquet"
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
    / "dim_status.parquet"
)


# =============================================================================
# Expected staged structure
# =============================================================================

EXPECTED_COLUMNS = [
    "status",
    "label",
]


# =============================================================================
# Loading
# =============================================================================

def load_staged_status() -> pl.DataFrame:
    """
    Load the validated staged Status lookup.
    """

    if not STAGED_STATUS_FILE.is_file():

        raise FileNotFoundError(
            f"Staged Status lookup not found: "
            f"{STAGED_STATUS_FILE}"
        )

    df = pl.read_parquet(
        STAGED_STATUS_FILE
    )

    if df.columns != EXPECTED_COLUMNS:

        raise ValueError(
            "Unexpected staged Status schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_credit_status_codes() -> set[int]:
    """
    Return distinct non-null Status codes used by stg_credit_risk.
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

    if "status" not in schema.names():

        raise KeyError(
            "stg_credit_risk does not contain 'status'."
        )

    values = (
        lazy_df
        .select(
            pl.col("status")
            .drop_nulls()
            .unique()
        )
        .collect()["status"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_status(
    df: pl.DataFrame,
) -> None:
    """
    Validate staged Status lookup.
    """

    print()
    print("=" * 100)
    print("STAGED STATUS SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows : "
        f"{df.height:,}"
    )

    if df.height != 5:

        raise ValueError(
            f"Expected 5 staged Status rows, "
            f"found {df.height:,}."
        )

    print(
        "[PASS] Staged row count matches validated profile: 5."
    )

    # -------------------------------------------------------------------------
    # Key completeness
    # -------------------------------------------------------------------------

    null_keys = df[
        "status"
    ].null_count()

    if null_keys != 0:

        raise ValueError(
            f"Status lookup contains "
            f"{null_keys:,} NULL key value(s)."
        )

    print(
        "[PASS] status contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Key uniqueness
    # -------------------------------------------------------------------------

    distinct_keys = df[
        "status"
    ].n_unique()

    if distinct_keys != df.height:

        raise ValueError(
            "Status lookup key is not unique."
        )

    print(
        f"[PASS] status is unique: "
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
            f"Status lookup contains "
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
    credit_codes: set[int],
) -> pl.DataFrame:
    """
    Preserve all Status members and add current fact-usage metadata.
    """

    return (
        staged
        .with_columns(
            pl.col("status")
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
            "status"
        )
    )


# =============================================================================
# Analytical validation
# =============================================================================

def validate_analytical_dimension(
    df: pl.DataFrame,
    staged: pl.DataFrame,
    credit_codes: set[int],
) -> None:
    """
    Validate analytical Status dimension.
    """

    print()
    print("=" * 100)
    print("ANALYTICAL STATUS DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    # -------------------------------------------------------------------------

    if df.height != staged.height:

        raise ValueError(
            f"Analytical Status row count mismatch. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} Status lookup members preserved."
    )

    # -------------------------------------------------------------------------
    # Key integrity
    # -------------------------------------------------------------------------

    if df[
        "status"
    ].n_unique() != df.height:

        raise ValueError(
            "Analytical Status key is not unique."
        )

    print(
        "[PASS] status remains unique."
    )

    if df[
        "status"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Status contains NULL keys."
        )

    print(
        "[PASS] status contains 0 NULL values."
    )

    if df[
        "label"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Status contains NULL labels."
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
            "status"
        ].to_list()
    }

    missing_credit = sorted(
        credit_codes
        - dimension_codes
    )

    if missing_credit:

        raise ValueError(
            "Credit-risk Status coverage failed. "
            f"Missing: {missing_credit}"
        )

    print(
        f"[PASS] All {len(credit_codes):,} credit-risk "
        "Status codes are represented."
    )

    # -------------------------------------------------------------------------
    # Usage flag reconciliation
    # -------------------------------------------------------------------------

    flagged_codes = {
        int(value)
        for value in (
            df
            .filter(
                pl.col(
                    "used_in_credit_risk"
                )
            )[
                "status"
            ]
            .to_list()
        )
    }

    if flagged_codes != credit_codes:

        raise ValueError(
            "Status usage flags do not reconcile "
            "to credit-risk fact codes."
        )

    print(
        "[PASS] Credit-risk usage flags reconcile."
    )

    # -------------------------------------------------------------------------
    # Expected usage counts
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

    if used_count != 2:

        raise ValueError(
            f"Expected 2 currently used Status codes, "
            f"found {used_count:,}."
        )

    if unused_count != 3:

        raise ValueError(
            f"Expected 3 currently unused Status codes, "
            f"found {unused_count:,}."
        )

    print(
        "[PASS] Current usage population reconciles: "
        "2 used / 3 unused."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save analytical Status dimension.
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
        "Saved analytical Status dimension:"
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
    Re-read saved Status dimension.
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
            f"Saved Status row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "status"
    ].n_unique() != saved.height:

        raise ValueError(
            "Saved analytical Status key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows          : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Status codes: "
        f"{saved['status'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "status_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Status usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Status dimension:"
    )

    print(
        saved
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build analytical Status dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL STATUS DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Status : {STAGED_STATUS_FILE}"
    )

    print(
        f"Credit fact   : {CREDIT_FACT_FILE}"
    )

    print(
        f"Output        : {OUTPUT_FILE}"
    )

    staged = (
        load_staged_status()
    )

    credit_codes = (
        load_credit_status_codes()
    )

    validate_staged_status(
        staged
    )

    analytical = (
        build_analytical_dimension(
            staged=staged,
            credit_codes=credit_codes,
        )
    )

    validate_analytical_dimension(
        df=analytical,
        staged=staged,
        credit_codes=credit_codes,
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
        "ANALYTICAL STATUS DIMENSION BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
