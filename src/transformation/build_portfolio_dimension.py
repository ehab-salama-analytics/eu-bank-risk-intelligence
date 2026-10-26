"""
Build the analytical EBA Portfolio dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5D — Analytical Portfolio Dimension

Inputs:
    data/staging/reference/dim_portfolio.parquet
    data/staging/facts/stg_credit_risk.parquet

Output:
    data/analytical/dimensions/dim_portfolio.parquet

Purpose:
    Build a reusable analytical Portfolio dimension from the validated
    staged EBA lookup.

Validated source characteristics:
    - 8 staged Portfolio rows;
    - portfolio is unique;
    - no NULL portfolio keys;
    - 3 Portfolio codes are currently used by the credit-risk fact;
    - no fact Portfolio codes are missing from the lookup.

This script preserves every staged lookup member.

This script DOES NOT:
    - modify staging data;
    - remove unused lookup members;
    - join the dimension to facts;
    - calculate KPIs;
    - create surrogate warehouse keys.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_PORTFOLIO_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_portfolio.parquet"
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
    / "dim_portfolio.parquet"
)


# =============================================================================
# Expected staged structure
# =============================================================================

EXPECTED_COLUMNS = [
    "portfolio",
    "label",
]


# =============================================================================
# Loading
# =============================================================================

def load_staged_portfolio() -> pl.DataFrame:
    """Load validated staged Portfolio lookup."""

    if not STAGED_PORTFOLIO_FILE.is_file():

        raise FileNotFoundError(
            f"Staged Portfolio lookup not found: "
            f"{STAGED_PORTFOLIO_FILE}"
        )

    df = pl.read_parquet(
        STAGED_PORTFOLIO_FILE
    )

    if df.columns != EXPECTED_COLUMNS:

        raise ValueError(
            "Unexpected staged Portfolio schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_credit_portfolio_codes() -> set[int]:
    """Return distinct Portfolio codes used by stg_credit_risk."""

    if not CREDIT_FACT_FILE.is_file():

        raise FileNotFoundError(
            f"Credit-risk staging file not found: "
            f"{CREDIT_FACT_FILE}"
        )

    lazy_df = pl.scan_parquet(
        CREDIT_FACT_FILE
    )

    schema = lazy_df.collect_schema()

    if "portfolio" not in schema.names():

        raise KeyError(
            "stg_credit_risk does not contain 'portfolio'."
        )

    values = (
        lazy_df
        .select(
            pl.col("portfolio")
            .drop_nulls()
            .unique()
        )
        .collect()["portfolio"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_portfolio(
    df: pl.DataFrame,
) -> None:
    """Validate staged Portfolio lookup."""

    print()
    print("=" * 100)
    print("STAGED PORTFOLIO SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows : {df.height:,}"
    )

    if df.height != 8:

        raise ValueError(
            f"Expected 8 staged Portfolio rows, "
            f"found {df.height:,}."
        )

    print(
        "[PASS] Staged row count matches validated profile: 8."
    )

    null_keys = df[
        "portfolio"
    ].null_count()

    if null_keys != 0:

        raise ValueError(
            f"Portfolio lookup contains "
            f"{null_keys:,} NULL key value(s)."
        )

    print(
        "[PASS] portfolio contains 0 NULL values."
    )

    distinct_keys = df[
        "portfolio"
    ].n_unique()

    if distinct_keys != df.height:

        raise ValueError(
            "Portfolio lookup key is not unique."
        )

    print(
        f"[PASS] portfolio is unique: "
        f"{distinct_keys:,} distinct code(s)."
    )

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:

        raise ValueError(
            f"Portfolio lookup contains "
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
    Preserve all Portfolio members and add current fact-usage metadata.
    """

    return (
        staged
        .with_columns(
            pl.col("portfolio")
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
            "portfolio"
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
    """Validate analytical Portfolio dimension."""

    print()
    print("=" * 100)
    print("ANALYTICAL PORTFOLIO DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    # -------------------------------------------------------------------------

    if df.height != staged.height:

        raise ValueError(
            f"Analytical Portfolio row count mismatch. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} Portfolio lookup members preserved."
    )

    # -------------------------------------------------------------------------
    # Key integrity
    # -------------------------------------------------------------------------

    if df[
        "portfolio"
    ].n_unique() != df.height:

        raise ValueError(
            "Analytical Portfolio key is not unique."
        )

    print(
        "[PASS] portfolio remains unique."
    )

    if df[
        "portfolio"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Portfolio contains NULL keys."
        )

    print(
        "[PASS] portfolio contains 0 NULL values."
    )

    if df[
        "label"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Portfolio contains NULL labels."
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
            "portfolio"
        ].to_list()
    }

    missing_credit = sorted(
        credit_codes
        - dimension_codes
    )

    if missing_credit:

        raise ValueError(
            "Credit-risk Portfolio coverage failed. "
            f"Missing: {missing_credit}"
        )

    print(
        f"[PASS] All {len(credit_codes):,} credit-risk "
        "Portfolio codes are represented."
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
                "portfolio"
            ]
            .to_list()
        )
    }

    if flagged_codes != credit_codes:

        raise ValueError(
            "Portfolio usage flags do not reconcile "
            "to credit-risk fact codes."
        )

    print(
        "[PASS] Credit-risk usage flags reconcile."
    )

    # -------------------------------------------------------------------------
    # Validated usage counts
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

    if used_count != 3:

        raise ValueError(
            f"Expected 3 currently used Portfolio codes, "
            f"found {used_count:,}."
        )

    if unused_count != 5:

        raise ValueError(
            f"Expected 5 currently unused Portfolio codes, "
            f"found {unused_count:,}."
        )

    print(
        "[PASS] Current usage population reconciles: "
        "3 used / 5 unused."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """Save analytical Portfolio dimension."""

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
        "Saved analytical Portfolio dimension:"
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
    """Re-read persisted Portfolio dimension."""

    if not OUTPUT_FILE.is_file():

        raise FileNotFoundError(
            f"Expected output not found: {OUTPUT_FILE}"
        )

    saved = pl.read_parquet(
        OUTPUT_FILE
    )

    if saved.height != expected_rows:

        raise ValueError(
            f"Saved Portfolio row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "portfolio"
    ].n_unique() != saved.height:

        raise ValueError(
            "Saved analytical Portfolio key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows              : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Portfolio codes: "
        f"{saved['portfolio'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "portfolio_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Portfolio usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Portfolio dimension:"
    )

    print(
        saved
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Build analytical Portfolio dimension."""

    print("=" * 100)
    print("BUILDING ANALYTICAL PORTFOLIO DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Portfolio : {STAGED_PORTFOLIO_FILE}"
    )

    print(
        f"Credit fact      : {CREDIT_FACT_FILE}"
    )

    print(
        f"Output           : {OUTPUT_FILE}"
    )

    staged = (
        load_staged_portfolio()
    )

    credit_codes = (
        load_credit_portfolio_codes()
    )

    validate_staged_portfolio(
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
        "ANALYTICAL PORTFOLIO DIMENSION BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
