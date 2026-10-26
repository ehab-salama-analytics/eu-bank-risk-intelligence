"""
Build the analytical EBA NACE dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5G — Analytical NACE Dimension

Inputs:
    data/staging/reference/dim_nace.parquet
    data/staging/facts/stg_credit_risk.parquet

Output:
    data/analytical/dimensions/dim_nace.parquet

Purpose:
    Build a reusable analytical NACE dimension from the validated staged
    EBA lookup.

Validated source characteristics:
    - 20 staged NACE rows;
    - nace_codes is unique;
    - no NULL nace_codes;
    - all 20 NACE codes are currently used by the credit-risk fact;
    - no credit-risk NACE codes are missing from the lookup.

This script preserves every staged lookup member and adds current
fact-usage metadata.

This script DOES NOT:
    - modify staging data;
    - modify fact data;
    - infer or replace NACE definitions;
    - calculate sector concentrations;
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

STAGED_NACE_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_nace.parquet"
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
    / "dim_nace.parquet"
)


# =============================================================================
# Expected staged structure
# =============================================================================

EXPECTED_COLUMNS = [
    "nace_codes",
    "label",
]


# =============================================================================
# Loading
# =============================================================================

def load_staged_nace() -> pl.DataFrame:
    """Load the validated staged NACE lookup."""

    if not STAGED_NACE_FILE.is_file():
        raise FileNotFoundError(
            f"Staged NACE lookup not found: {STAGED_NACE_FILE}"
        )

    df = pl.read_parquet(
        STAGED_NACE_FILE
    )

    if df.columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected staged NACE schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_credit_nace_codes() -> set[int]:
    """Return distinct non-null NACE codes used by stg_credit_risk."""

    if not CREDIT_FACT_FILE.is_file():
        raise FileNotFoundError(
            f"Credit-risk staging file not found: {CREDIT_FACT_FILE}"
        )

    lazy_df = pl.scan_parquet(
        CREDIT_FACT_FILE
    )

    schema = lazy_df.collect_schema()

    if "nace_codes" not in schema.names():
        raise KeyError(
            "stg_credit_risk does not contain 'nace_codes'."
        )

    values = (
        lazy_df
        .select(
            pl.col("nace_codes")
            .drop_nulls()
            .unique()
        )
        .collect()["nace_codes"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_nace(
    df: pl.DataFrame,
) -> None:
    """Validate staged NACE lookup."""

    print()
    print("=" * 100)
    print("STAGED NACE SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows : {df.height:,}"
    )

    if df.height != 20:
        raise ValueError(
            f"Expected 20 staged NACE rows, found {df.height:,}."
        )

    print(
        "[PASS] Staged row count matches validated profile: 20."
    )

    null_keys = df[
        "nace_codes"
    ].null_count()

    if null_keys != 0:
        raise ValueError(
            f"NACE lookup contains {null_keys:,} NULL key value(s)."
        )

    print(
        "[PASS] nace_codes contains 0 NULL values."
    )

    distinct_keys = df[
        "nace_codes"
    ].n_unique()

    if distinct_keys != df.height:
        raise ValueError(
            "NACE lookup key is not unique."
        )

    print(
        f"[PASS] nace_codes is unique: "
        f"{distinct_keys:,} distinct code(s)."
    )

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:
        raise ValueError(
            f"NACE lookup contains {null_labels:,} NULL label(s)."
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
    Preserve all NACE lookup members and add current fact-usage metadata.
    """

    return (
        staged
        .with_columns(
            pl.col("nace_codes")
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
            "nace_codes"
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
    """Validate analytical NACE dimension."""

    print()
    print("=" * 100)
    print("ANALYTICAL NACE DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    # -------------------------------------------------------------------------

    if df.height != staged.height:
        raise ValueError(
            f"Analytical NACE row count mismatch. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} NACE lookup members preserved."
    )

    # -------------------------------------------------------------------------
    # Key integrity
    # -------------------------------------------------------------------------

    if df[
        "nace_codes"
    ].n_unique() != df.height:
        raise ValueError(
            "Analytical NACE key is not unique."
        )

    print(
        "[PASS] nace_codes remains unique."
    )

    if df[
        "nace_codes"
    ].null_count() != 0:
        raise ValueError(
            "Analytical NACE contains NULL keys."
        )

    print(
        "[PASS] nace_codes contains 0 NULL values."
    )

    if df[
        "label"
    ].null_count() != 0:
        raise ValueError(
            "Analytical NACE contains NULL labels."
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
            "nace_codes"
        ].to_list()
    }

    missing_credit = sorted(
        credit_codes
        - dimension_codes
    )

    if missing_credit:
        raise ValueError(
            "Credit-risk NACE coverage failed. "
            f"Missing: {missing_credit}"
        )

    print(
        f"[PASS] All {len(credit_codes):,} credit-risk "
        "NACE codes are represented."
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
                    "used_in_credit_risk"
                )
            )[
                "nace_codes"
            ]
            .to_list()
        )
    }

    if flagged_codes != credit_codes:
        raise ValueError(
            "NACE usage flags do not reconcile "
            "to credit-risk fact codes."
        )

    print(
        "[PASS] Credit-risk usage flags reconcile."
    )

    # -------------------------------------------------------------------------
    # Full-domain usage check
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

    if used_count != 20:
        raise ValueError(
            f"Expected all 20 NACE codes to be currently used, "
            f"found {used_count:,}."
        )

    if unused_count != 0:
        raise ValueError(
            f"Expected 0 unused NACE codes, "
            f"found {unused_count:,}."
        )

    if dimension_codes != credit_codes:
        raise ValueError(
            "NACE dimension population does not exactly match "
            "the current credit-risk fact NACE domain."
        )

    print(
        "[PASS] Full NACE domain reconciliation: "
        "20 used / 0 unused."
    )

    print(
        "[PASS] Analytical NACE population exactly matches "
        "the current credit-risk NACE domain."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """Save analytical NACE dimension."""

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
        "Saved analytical NACE dimension:"
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
    """Re-read persisted analytical NACE dimension."""

    if not OUTPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Expected output not found: {OUTPUT_FILE}"
        )

    saved = pl.read_parquet(
        OUTPUT_FILE
    )

    if saved.height != expected_rows:
        raise ValueError(
            f"Saved NACE row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "nace_codes"
    ].n_unique() != saved.height:
        raise ValueError(
            "Saved analytical NACE key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows         : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct NACE codes: "
        f"{saved['nace_codes'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "nace_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "NACE usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical NACE dimension:"
    )

    print(
        saved
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Build analytical NACE dimension."""

    print("=" * 100)
    print("BUILDING ANALYTICAL NACE DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged NACE : {STAGED_NACE_FILE}"
    )

    print(
        f"Credit fact : {CREDIT_FACT_FILE}"
    )

    print(
        f"Output      : {OUTPUT_FILE}"
    )

    staged = (
        load_staged_nace()
    )

    credit_codes = (
        load_credit_nace_codes()
    )

    validate_staged_nace(
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
        "ANALYTICAL NACE DIMENSION BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
