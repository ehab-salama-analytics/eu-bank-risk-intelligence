"""
Build the analytical reporting-period dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-2 — Analytical Period Dimension

Inputs:
    data/staging/facts/stg_credit_risk.parquet
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/dimensions/dim_period.parquet

Purpose:
    Build a deterministic analytical period dimension from reporting
    periods actually present in the validated staging fact tables.

This script:
    - reads distinct reporting periods from both staged fact tables;
    - checks that both datasets use compatible period values;
    - validates YYYYMM structure;
    - validates quarter-end months;
    - derives calendar attributes;
    - writes one row per reporting period.

This script DOES NOT:
    - modify staging tables;
    - modify raw data;
    - calculate banking KPIs;
    - join dimensions to facts.
"""

from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGING_FACT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
)

CREDIT_FILE = (
    STAGING_FACT_DIR
    / "stg_credit_risk.parquet"
)

OTHER_FILE = (
    STAGING_FACT_DIR
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
    / "dim_period.parquet"
)


# =============================================================================
# Expected reporting convention
# =============================================================================

QUARTER_END_MONTHS = {
    3: 1,
    6: 2,
    9: 3,
    12: 4,
}


# =============================================================================
# Source-period extraction
# =============================================================================

def load_distinct_periods(
    file_path: Path,
    dataset_name: str,
) -> set[int]:
    """
    Read the distinct non-null Period values from one staged fact table.
    """

    if not file_path.is_file():

        raise FileNotFoundError(
            f"{dataset_name} staging file not found: {file_path}"
        )

    lazy_df = pl.scan_parquet(
        file_path
    )

    schema = lazy_df.collect_schema()

    if "period" not in schema.names():

        raise KeyError(
            f"{dataset_name} does not contain a 'period' column."
        )

    null_count = (
        lazy_df
        .select(
            pl.col("period")
            .null_count()
            .alias("null_count")
        )
        .collect()["null_count"][0]
    )

    if int(null_count) != 0:

        raise ValueError(
            f"{dataset_name} contains "
            f"{int(null_count):,} NULL period value(s)."
        )

    periods = (
        lazy_df
        .select(
            pl.col("period")
            .unique()
            .sort()
        )
        .collect()["period"]
        .to_list()
    )

    return {
        int(period)
        for period in periods
    }


# =============================================================================
# Period parsing
# =============================================================================

def parse_period(
    period_code: int,
) -> dict[str, object]:
    """
    Convert YYYYMM reporting code into analytical period attributes.

    Example:

        202409
        ->
        period_end_date = 2024-09-30
        year            = 2024
        quarter         = 3
        month           = 9
        year_quarter    = 2024 Q3
        period_key      = 20240930
    """

    period_text = str(
        period_code
    )

    if len(period_text) != 6:

        raise ValueError(
            f"Invalid period '{period_code}'. "
            "Expected six-digit YYYYMM format."
        )

    if not period_text.isdigit():

        raise ValueError(
            f"Invalid period '{period_code}'. "
            "Period must contain digits only."
        )

    year = int(
        period_text[:4]
    )

    month = int(
        period_text[4:]
    )

    if month not in QUARTER_END_MONTHS:

        raise ValueError(
            f"Period '{period_code}' uses month {month:02d}. "
            "Expected a quarter-end month: 03, 06, 09 or 12."
        )

    quarter = QUARTER_END_MONTHS[
        month
    ]

    last_day = calendar.monthrange(
        year,
        month,
    )[1]

    period_end_date = date(
        year,
        month,
        last_day,
    )

    # Stable integer date key in YYYYMMDD form.
    period_key = int(
        period_end_date.strftime(
            "%Y%m%d"
        )
    )

    return {
        "period_key": period_key,
        "period_code": period_code,
        "period_end_date": period_end_date,
        "year": year,
        "quarter": quarter,
        "month": month,
        "year_quarter": f"{year} Q{quarter}",
        "is_year_end": month == 12,
    }


# =============================================================================
# Build dimension
# =============================================================================

def build_period_dimension(
    periods: set[int],
) -> pl.DataFrame:
    """
    Build one analytical dimension row per reporting period.
    """

    records = [
        parse_period(
            period
        )
        for period in sorted(
            periods
        )
    ]

    df = pl.DataFrame(
        records,
        schema={
            "period_key": pl.Int32,
            "period_code": pl.Int32,
            "period_end_date": pl.Date,
            "year": pl.Int32,
            "quarter": pl.Int8,
            "month": pl.Int8,
            "year_quarter": pl.String,
            "is_year_end": pl.Boolean,
        },
    )

    return df


# =============================================================================
# Validation
# =============================================================================

def validate_dimension(
    df: pl.DataFrame,
    source_periods: set[int],
) -> None:
    """
    Validate the completed dimension before writing it.
    """

    print()
    print("=" * 100)
    print("PERIOD DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------------------

    if df.height != len(
        source_periods
    ):

        raise ValueError(
            "Period dimension row count does not reconcile "
            "to the distinct source periods."
        )

    print(
        f"[PASS] Row count: "
        f"{df.height} period row(s)."
    )

    # -------------------------------------------------------------------------
    # period_code uniqueness
    # -------------------------------------------------------------------------

    duplicate_period_codes = (
        df["period_code"]
        .is_duplicated()
        .sum()
    )

    if duplicate_period_codes != 0:

        raise ValueError(
            "Duplicate period_code values detected."
        )

    print(
        "[PASS] period_code is unique."
    )

    # -------------------------------------------------------------------------
    # period_key uniqueness
    # -------------------------------------------------------------------------

    duplicate_period_keys = (
        df["period_key"]
        .is_duplicated()
        .sum()
    )

    if duplicate_period_keys != 0:

        raise ValueError(
            "Duplicate period_key values detected."
        )

    print(
        "[PASS] period_key is unique."
    )

    # -------------------------------------------------------------------------
    # Reconcile source values
    # -------------------------------------------------------------------------

    dimension_periods = set(
        df["period_code"]
        .to_list()
    )

    if dimension_periods != source_periods:

        raise ValueError(
            "Period codes in dimension do not match "
            "the validated source periods."
        )

    print(
        "[PASS] All staged fact periods are represented."
    )

    # -------------------------------------------------------------------------
    # Quarter validation
    # -------------------------------------------------------------------------

    invalid_quarters = (
        df
        .filter(
            ~pl.col("quarter")
            .is_in(
                [1, 2, 3, 4]
            )
        )
        .height
    )

    if invalid_quarters != 0:

        raise ValueError(
            "Invalid quarter values detected."
        )

    print(
        "[PASS] Quarter values are valid."
    )

    # -------------------------------------------------------------------------
    # Year-end validation
    # -------------------------------------------------------------------------

    invalid_year_end = (
        df
        .filter(
            pl.col("is_year_end")
            != (
                pl.col("month")
                == 12
            )
        )
        .height
    )

    if invalid_year_end != 0:

        raise ValueError(
            "is_year_end logic failed validation."
        )

    print(
        "[PASS] Year-end flag is consistent."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save analytical period dimension to Parquet.
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
        "Saved analytical period dimension:"
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
    Re-read the saved Parquet file and validate persisted output.
    """

    if not OUTPUT_FILE.is_file():

        raise FileNotFoundError(
            f"Expected output file not found: {OUTPUT_FILE}"
        )

    saved = pl.read_parquet(
        OUTPUT_FILE
    )

    if saved.height != expected_rows:

        raise ValueError(
            "Saved period dimension row count failed reconciliation."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows: "
        f"{saved.height}"
    )

    print()
    print(
        saved
        .sort(
            "period_code"
        )
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build the analytical period dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL PERIOD DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Credit staging : {CREDIT_FILE}"
    )

    print(
        f"Other staging  : {OTHER_FILE}"
    )

    print(
        f"Output         : {OUTPUT_FILE}"
    )

    # -------------------------------------------------------------------------
    # Extract reporting periods
    # -------------------------------------------------------------------------

    credit_periods = load_distinct_periods(
        CREDIT_FILE,
        "stg_credit_risk",
    )

    other_periods = load_distinct_periods(
        OTHER_FILE,
        "stg_other_templates",
    )

    print()
    print(
        f"Credit-risk periods    : "
        f"{sorted(credit_periods)}"
    )

    print(
        f"Other-template periods : "
        f"{sorted(other_periods)}"
    )

    # -------------------------------------------------------------------------
    # Cross-dataset consistency
    # -------------------------------------------------------------------------

    if credit_periods != other_periods:

        raise ValueError(
            "Reporting periods differ between the two staged fact datasets.\n"
            f"Credit risk: {sorted(credit_periods)}\n"
            f"Other:       {sorted(other_periods)}"
        )

    print()
    print(
        "[PASS] Both staged fact datasets contain "
        "the same reporting periods."
    )

    source_periods = (
        credit_periods
        | other_periods
    )

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    period_dimension = (
        build_period_dimension(
            source_periods
        )
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_dimension(
        period_dimension,
        source_periods,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_dimension(
        period_dimension
    )

    # -------------------------------------------------------------------------
    # Validate saved output
    # -------------------------------------------------------------------------

    validate_saved_dimension(
        expected_rows=len(
            source_periods
        )
    )

    print()
    print("=" * 100)
    print(
        "ANALYTICAL PERIOD DIMENSION BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No staging or raw data was modified."
    )


if __name__ == "__main__":
    main()
