"""
Build the analytical EBA Financial Instrument dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5J — Analytical Financial Instrument Dimension

Inputs:
    data/staging/reference/dim_financial_instrument.parquet
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/dimensions/dim_financial_instrument.parquet

Purpose:
    Build a reusable analytical Financial Instrument dimension from the
    validated staged EBA lookup.

Validated source characteristics:
    - 9 staged Financial Instrument rows;
    - financial_instruments is unique;
    - no NULL financial_instruments keys;
    - all 9 Financial Instrument codes are currently used by the
      other-template fact;
    - no fact Financial Instrument codes are missing from the lookup.

This script preserves every staged lookup member and adds current
fact-usage metadata.

This script DOES NOT:
    - modify staging data;
    - modify fact data;
    - infer financial-instrument classifications;
    - calculate banking KPIs;
    - calculate valuation or liquidity metrics;
    - join the dimension to fact tables.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_FINANCIAL_INSTRUMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_financial_instrument.parquet"
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
    / "dim_financial_instrument.parquet"
)


# =============================================================================
# Expected staged structure
# =============================================================================

EXPECTED_COLUMNS = [
    "financial_instruments",
    "label",
]


# =============================================================================
# Loading
# =============================================================================

def load_staged_financial_instrument() -> pl.DataFrame:
    """
    Load the validated staged Financial Instrument lookup.
    """

    if not STAGED_FINANCIAL_INSTRUMENT_FILE.is_file():
        raise FileNotFoundError(
            f"Staged Financial Instrument lookup not found: "
            f"{STAGED_FINANCIAL_INSTRUMENT_FILE}"
        )

    df = pl.read_parquet(
        STAGED_FINANCIAL_INSTRUMENT_FILE
    )

    if df.columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected staged Financial Instrument schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_other_financial_instrument_codes() -> set[int]:
    """
    Return distinct non-null financial_instruments codes used by
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

    if "financial_instruments" not in schema.names():
        raise KeyError(
            "stg_other_templates does not contain "
            "'financial_instruments'."
        )

    values = (
        lazy_df
        .select(
            pl.col("financial_instruments")
            .drop_nulls()
            .unique()
        )
        .collect()["financial_instruments"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_financial_instrument(
    df: pl.DataFrame,
) -> None:
    """
    Validate staged Financial Instrument lookup.
    """

    print()
    print("=" * 100)
    print("STAGED FINANCIAL INSTRUMENT SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows : {df.height:,}"
    )

    if df.height != 9:
        raise ValueError(
            f"Expected 9 staged Financial Instrument rows, "
            f"found {df.height:,}."
        )

    print(
        "[PASS] Staged row count matches validated profile: 9."
    )

    null_keys = df[
        "financial_instruments"
    ].null_count()

    if null_keys != 0:
        raise ValueError(
            f"Financial Instrument lookup contains "
            f"{null_keys:,} NULL key value(s)."
        )

    print(
        "[PASS] financial_instruments contains 0 NULL values."
    )

    distinct_keys = df[
        "financial_instruments"
    ].n_unique()

    if distinct_keys != df.height:
        raise ValueError(
            "Financial Instrument lookup key is not unique."
        )

    print(
        f"[PASS] financial_instruments is unique: "
        f"{distinct_keys:,} distinct code(s)."
    )

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:
        raise ValueError(
            f"Financial Instrument lookup contains "
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
    Preserve all Financial Instrument members and add current
    fact-usage metadata.
    """

    return (
        staged
        .with_columns(
            pl.col("financial_instruments")
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
            "financial_instruments"
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
    Validate analytical Financial Instrument dimension.
    """

    print()
    print("=" * 100)
    print("ANALYTICAL FINANCIAL INSTRUMENT DIMENSION VALIDATION")
    print("=" * 100)

    if df.height != staged.height:
        raise ValueError(
            f"Analytical Financial Instrument row count mismatch. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} Financial Instrument lookup members preserved."
    )

    if df[
        "financial_instruments"
    ].n_unique() != df.height:
        raise ValueError(
            "Analytical Financial Instrument key is not unique."
        )

    print(
        "[PASS] financial_instruments remains unique."
    )

    if df[
        "financial_instruments"
    ].null_count() != 0:
        raise ValueError(
            "Analytical Financial Instrument contains NULL keys."
        )

    print(
        "[PASS] financial_instruments contains 0 NULL values."
    )

    if df[
        "label"
    ].null_count() != 0:
        raise ValueError(
            "Analytical Financial Instrument contains NULL labels."
        )

    print(
        "[PASS] label contains 0 NULL values."
    )

    dimension_codes = {
        int(value)
        for value in df[
            "financial_instruments"
        ].to_list()
    }

    missing_other = sorted(
        other_codes
        - dimension_codes
    )

    if missing_other:
        raise ValueError(
            "Other-template Financial Instrument coverage failed. "
            f"Missing: {missing_other}"
        )

    print(
        f"[PASS] All {len(other_codes):,} other-template "
        "Financial Instrument codes are represented."
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
                "financial_instruments"
            ]
            .to_list()
        )
    }

    if flagged_codes != other_codes:
        raise ValueError(
            "Financial Instrument usage flags do not reconcile "
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

    if used_count != 9:
        raise ValueError(
            f"Expected all 9 Financial Instrument codes to be "
            f"currently used, found {used_count:,}."
        )

    if unused_count != 0:
        raise ValueError(
            f"Expected 0 unused Financial Instrument codes, "
            f"found {unused_count:,}."
        )

    if dimension_codes != other_codes:
        raise ValueError(
            "Financial Instrument dimension population does not "
            "exactly match the current other-template fact domain."
        )

    print(
        "[PASS] Full Financial Instrument domain reconciliation: "
        "9 used / 0 unused."
    )

    print(
        "[PASS] Analytical Financial Instrument population exactly "
        "matches the current other-template fact domain."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save analytical Financial Instrument dimension.
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
        "Saved analytical Financial Instrument dimension:"
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
    Re-read persisted analytical Financial Instrument dimension.
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
            f"Saved Financial Instrument row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "financial_instruments"
    ].n_unique() != saved.height:
        raise ValueError(
            "Saved analytical Financial Instrument key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows                         : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Financial Instrument codes: "
        f"{saved['financial_instruments'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "financial_instrument_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Financial Instrument usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Financial Instrument dimension:"
    )

    print(
        saved
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build analytical Financial Instrument dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL FINANCIAL INSTRUMENT DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Financial Instrument : "
        f"{STAGED_FINANCIAL_INSTRUMENT_FILE}"
    )

    print(
        f"Other fact                  : "
        f"{OTHER_FACT_FILE}"
    )

    print(
        f"Output                      : "
        f"{OUTPUT_FILE}"
    )

    staged = (
        load_staged_financial_instrument()
    )

    other_codes = (
        load_other_financial_instrument_codes()
    )

    validate_staged_financial_instrument(
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
        "ANALYTICAL FINANCIAL INSTRUMENT DIMENSION BUILD "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
