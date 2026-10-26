"""
Build the staged EBA credit-risk fact dataset.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 2C-3A — Credit-Risk Fact Staging

Source:
    data/raw/eba_2025/tr_cre.csv

Output:
    data/staging/facts/stg_credit_risk.parquet

Purpose:
    Perform controlled structural standardisation before analytical modelling.

Transformations:
    - preserve every source record;
    - standardise column names to snake_case;
    - convert non-blank Amount values to Float64;
    - preserve blank Amount values as NULL;
    - write the result to Parquet.

This script DOES NOT:
    - modify the raw CSV;
    - remove records;
    - deduplicate records;
    - replace missing Amount with zero;
    - join reference tables;
    - calculate banking KPIs;
    - create the final analytical fact table.
"""

from __future__ import annotations

from pathlib import Path
import re

import polars as pl


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "eba_2025"
    / "tr_cre.csv"
)

STAGING_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
)

OUTPUT_FILE = (
    STAGING_DIR
    / "stg_credit_risk.parquet"
)


# =============================================================================
# Expected raw schema
# =============================================================================

EXPECTED_RAW_COLUMNS = [
    "LEI_Code",
    "NSA",
    "Period",
    "Item",
    "Label",
    "Portfolio",
    "Country",
    "Country_rank",
    "Exposure",
    "Status",
    "Perf_Status",
    "NACE_codes",
    "Amount",
    "Row",
    "Column",
    "Sheet",
]


# =============================================================================
# Column standardisation
# =============================================================================

def to_snake_case(
    column_name: str,
) -> str:
    """
    Convert a column name to lowercase snake_case.

    Examples:
        LEI_Code     -> lei_code
        Perf_Status  -> perf_status
        NACE_codes   -> nace_codes
    """

    column_name = str(
        column_name
    ).strip()

    column_name = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        column_name,
    )

    column_name = re.sub(
        r"_+",
        "_",
        column_name,
    )

    return (
        column_name
        .strip("_")
        .lower()
    )


def build_column_mapping(
    columns: list[str],
) -> dict[str, str]:
    """
    Build the raw-to-staging column-name mapping.
    """

    return {
        column: to_snake_case(
            column
        )
        for column in columns
    }


# =============================================================================
# Source loading and validation
# =============================================================================

def scan_raw_credit_risk() -> pl.LazyFrame:
    """
    Lazily scan the raw EBA credit-risk CSV.

    Amount is deliberately loaded as String because blank values need
    explicit treatment before numeric conversion.
    """

    if not RAW_FILE.is_file():

        raise FileNotFoundError(
            f"Raw credit-risk file not found: {RAW_FILE}"
        )

    return pl.scan_csv(
        RAW_FILE,
        separator=",",
        has_header=True,
        infer_schema_length=10_000,
        schema_overrides={
            "Amount": pl.String,
        },
        ignore_errors=False,
    )


def validate_raw_schema(
    lazy_df: pl.LazyFrame,
) -> None:
    """
    Confirm that the source columns still match the validated EBA schema.

    This protects the staging pipeline from silent upstream schema changes.
    """

    actual_columns = (
        lazy_df
        .collect_schema()
        .names()
    )

    if actual_columns != EXPECTED_RAW_COLUMNS:

        missing = [
            column
            for column in EXPECTED_RAW_COLUMNS
            if column not in actual_columns
        ]

        unexpected = [
            column
            for column in actual_columns
            if column not in EXPECTED_RAW_COLUMNS
        ]

        raise ValueError(
            "Raw credit-risk schema does not match the "
            "validated expected schema.\n"
            f"Missing columns: {missing}\n"
            f"Unexpected columns: {unexpected}\n"
            f"Actual columns: {actual_columns}"
        )

    print(
        "[PASS] Raw schema matches the expected "
        "16-column credit-risk structure."
    )


# =============================================================================
# Transformation
# =============================================================================

def build_staging_frame(
    lazy_df: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Apply staging-only transformations.

    Important Amount logic:

        non-blank string
            -> Float64

        blank / whitespace-only string
            -> NULL

    No blank value is converted to zero.
    """

    column_mapping = build_column_mapping(
        EXPECTED_RAW_COLUMNS
    )

    staged = lazy_df.rename(
        column_mapping
    )

    stripped_amount = (
        pl.col("amount")
        .str.strip_chars()
    )

    amount_numeric = (
        pl.when(
            pl.col("amount").is_null()
            |
            stripped_amount.eq("")
        )
        .then(
            None
        )
        .otherwise(
            stripped_amount.cast(
                pl.Float64,
                strict=True,
            )
        )
        .cast(
            pl.Float64
        )
        .alias(
            "amount"
        )
    )

    staged = staged.with_columns(
        amount_numeric
    )

    return staged


# =============================================================================
# Validation helpers
# =============================================================================

def get_row_count(
    lazy_df: pl.LazyFrame,
) -> int:
    """Return row count."""

    result = (
        lazy_df
        .select(
            pl.len().alias(
                "row_count"
            )
        )
        .collect()
    )

    return int(
        result["row_count"][0]
    )


def get_blank_amount_count(
    lazy_df: pl.LazyFrame,
) -> int:
    """
    Count blank / whitespace-only Amount strings in the raw source.
    """

    result = (
        lazy_df
        .select(
            (
                pl.col("Amount")
                .is_not_null()
                &
                pl.col("Amount")
                .str.strip_chars()
                .eq("")
            )
            .sum()
            .alias(
                "blank_count"
            )
        )
        .collect()
    )

    return int(
        result["blank_count"][0]
    )


def validate_staged_dataframe(
    df: pl.DataFrame,
    source_row_count: int,
    source_blank_amount_count: int,
) -> None:
    """
    Validate the materialised staged DataFrame before saving it.
    """

    print()
    print("=" * 100)
    print("PRE-SAVE STAGING VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    # -------------------------------------------------------------------------

    staged_row_count = df.height

    if staged_row_count != source_row_count:

        raise ValueError(
            "Row-count reconciliation failed. "
            f"Source rows={source_row_count:,}, "
            f"staged rows={staged_row_count:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{staged_row_count:,} rows preserved."
    )

    # -------------------------------------------------------------------------
    # Column count
    # -------------------------------------------------------------------------

    expected_staged_columns = [
        to_snake_case(
            column
        )
        for column in EXPECTED_RAW_COLUMNS
    ]

    if df.columns != expected_staged_columns:

        raise ValueError(
            "Staged column structure does not match expectation.\n"
            f"Expected: {expected_staged_columns}\n"
            f"Actual:   {df.columns}"
        )

    print(
        f"[PASS] Column structure: "
        f"{len(df.columns)} expected staging columns."
    )

    # -------------------------------------------------------------------------
    # Amount type
    # -------------------------------------------------------------------------

    amount_dtype = df.schema[
        "amount"
    ]

    if amount_dtype != pl.Float64:

        raise TypeError(
            "Staged Amount is not Float64. "
            f"Actual type: {amount_dtype}"
        )

    print(
        "[PASS] amount dtype: Float64."
    )

    # -------------------------------------------------------------------------
    # Missing Amount reconciliation
    # -------------------------------------------------------------------------

    staged_null_amount_count = int(
        df[
            "amount"
        ]
        .null_count()
    )

    if (
        staged_null_amount_count
        != source_blank_amount_count
    ):

        raise ValueError(
            "Amount missing-value reconciliation failed. "
            f"Raw blank Amounts={source_blank_amount_count:,}, "
            f"staged NULL Amounts={staged_null_amount_count:,}"
        )

    print(
        f"[PASS] Amount missingness preserved: "
        f"{staged_null_amount_count:,} NULL value(s)."
    )

    # -------------------------------------------------------------------------
    # Critical identifiers
    # -------------------------------------------------------------------------

    critical_columns = [
        "lei_code",
        "period",
        "item",
    ]

    for column in critical_columns:

        null_count = int(
            df[column]
            .null_count()
        )

        if null_count != 0:

            raise ValueError(
                f"Unexpected null values in '{column}': "
                f"{null_count:,}"
            )

        print(
            f"[PASS] {column}: "
            "0 NULL value(s)."
        )


# =============================================================================
# Save and post-save validation
# =============================================================================

def save_staging_table(
    df: pl.DataFrame,
) -> None:
    """
    Save staged credit-risk data to Parquet.
    """

    STAGING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    print()
    print(
        f"Saved staging table:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


def validate_saved_file(
    expected_rows: int,
    expected_null_amounts: int,
) -> None:
    """
    Re-read the saved Parquet metadata/data and reconcile key controls.
    """

    if not OUTPUT_FILE.is_file():

        raise FileNotFoundError(
            f"Expected staging output was not created: {OUTPUT_FILE}"
        )

    staged = pl.scan_parquet(
        OUTPUT_FILE
    )

    result = (
        staged
        .select(
            [
                pl.len().alias(
                    "row_count"
                ),

                pl.col("amount")
                .null_count()
                .alias(
                    "null_amount_count"
                ),

                pl.col("lei_code")
                .n_unique()
                .alias(
                    "distinct_lei"
                ),

                pl.col("item")
                .n_unique()
                .alias(
                    "distinct_item"
                ),

                pl.col("period")
                .n_unique()
                .alias(
                    "distinct_period"
                ),
            ]
        )
        .collect()
    )

    saved_rows = int(
        result["row_count"][0]
    )

    saved_null_amounts = int(
        result["null_amount_count"][0]
    )

    if saved_rows != expected_rows:

        raise ValueError(
            "Post-save row reconciliation failed. "
            f"Expected={expected_rows:,}, "
            f"saved={saved_rows:,}"
        )

    if (
        saved_null_amounts
        != expected_null_amounts
    ):

        raise ValueError(
            "Post-save Amount reconciliation failed. "
            f"Expected NULLs={expected_null_amounts:,}, "
            f"saved NULLs={saved_null_amounts:,}"
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows          : "
        f"{saved_rows:,}"
    )

    print(
        f"[PASS] NULL Amount values  : "
        f"{saved_null_amounts:,}"
    )

    print(
        f"[INFO] Distinct LEI codes  : "
        f"{int(result['distinct_lei'][0]):,}"
    )

    print(
        f"[INFO] Distinct Item codes : "
        f"{int(result['distinct_item'][0]):,}"
    )

    print(
        f"[INFO] Distinct periods    : "
        f"{int(result['distinct_period'][0]):,}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build and validate the staged credit-risk dataset.
    """

    print("=" * 100)
    print("BUILDING EBA CREDIT-RISK STAGING TABLE")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Raw source : {RAW_FILE}"
    )

    print(
        f"Output     : {OUTPUT_FILE}"
    )

    print()

    print(
        "STAGING RULES:"
    )

    print(
        "- preserve every raw record"
    )

    print(
        "- standardise column names"
    )

    print(
        "- convert non-blank Amount to Float64"
    )

    print(
        "- preserve blank Amount as NULL"
    )

    print(
        "- do not replace missing values with zero"
    )

    print(
        "- do not join dimensions"
    )

    print()

    # -------------------------------------------------------------------------
    # Load source
    # -------------------------------------------------------------------------

    raw_lazy = scan_raw_credit_risk()

    validate_raw_schema(
        raw_lazy
    )

    # -------------------------------------------------------------------------
    # Baseline reconciliation statistics
    # -------------------------------------------------------------------------

    source_row_count = get_row_count(
        raw_lazy
    )

    source_blank_amount_count = (
        get_blank_amount_count(
            raw_lazy
        )
    )

    print(
        f"[INFO] Raw rows            : "
        f"{source_row_count:,}"
    )

    print(
        f"[INFO] Raw blank Amounts   : "
        f"{source_blank_amount_count:,}"
    )

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------

    staged_lazy = build_staging_frame(
        raw_lazy
    )

    print()
    print(
        "Materialising staged credit-risk data..."
    )

    staged_df = staged_lazy.collect(
        engine="streaming"
    )

    # -------------------------------------------------------------------------
    # Validate before writing
    # -------------------------------------------------------------------------

    validate_staged_dataframe(
        df=staged_df,
        source_row_count=source_row_count,
        source_blank_amount_count=source_blank_amount_count,
    )

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    save_staging_table(
        staged_df
    )

    # -------------------------------------------------------------------------
    # Validate written Parquet
    # -------------------------------------------------------------------------

    validate_saved_file(
        expected_rows=source_row_count,
        expected_null_amounts=source_blank_amount_count,
    )

    print()
    print("=" * 100)
    print(
        "CREDIT-RISK STAGING BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "Raw source data was not modified."
    )


if __name__ == "__main__":
    main()
