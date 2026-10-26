"""
Validate the staged EBA credit-risk dataset.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 2C-3A — Credit-Risk Staging Validation

Input:
    data/staging/facts/stg_credit_risk.parquet

Purpose:
    Independently validate the persisted credit-risk staging table.

This script is READ ONLY.

It does not:
    - modify staged data;
    - modify raw data;
    - fill missing values;
    - deduplicate records;
    - join dimensions;
    - calculate banking KPIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGING_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
    / "stg_credit_risk.parquet"
)


# =============================================================================
# Validated expectations
# =============================================================================

EXPECTED_COLUMNS = [
    "lei_code",
    "nsa",
    "period",
    "item",
    "label",
    "portfolio",
    "country",
    "country_rank",
    "exposure",
    "status",
    "perf_status",
    "nace_codes",
    "amount",
    "row",
    "column",
    "sheet",
]

EXPECTED_ROW_COUNT = 648_951
EXPECTED_NULL_AMOUNT_COUNT = 1_637
EXPECTED_DISTINCT_LEI_COUNT = 120
EXPECTED_DISTINCT_ITEM_COUNT = 89
EXPECTED_PERIODS = [
    202409,
    202412,
    202503,
    202506,
]

ALL_OTHER_BANKS_LEI = "XXXXXXXXXXXXXXXXXXXX"


# =============================================================================
# Result tracking
# =============================================================================

@dataclass
class ValidationCounts:
    passed: int = 0
    warnings: int = 0
    failed: int = 0

    def add(
        self,
        status: str,
    ) -> None:

        if status == "PASS":
            self.passed += 1

        elif status == "WARNING":
            self.warnings += 1

        elif status == "FAIL":
            self.failed += 1


COUNTS = ValidationCounts()


def report(
    status: str,
    check: str,
    message: str,
) -> None:
    """Print and count one validation result."""

    COUNTS.add(
        status
    )

    print(
        f"[{status:<7}] "
        f"{check:<30} | "
        f"{message}"
    )


# =============================================================================
# File and schema validation
# =============================================================================

def validate_file_exists() -> bool:
    """Check that the staged Parquet file exists."""

    if STAGING_FILE.is_file():

        report(
            "PASS",
            "file_exists",
            f"Found {STAGING_FILE.name}",
        )

        return True

    report(
        "FAIL",
        "file_exists",
        f"File not found: {STAGING_FILE}",
    )

    return False


def load_staging() -> pl.LazyFrame:
    """Open staged Parquet lazily."""

    return pl.scan_parquet(
        STAGING_FILE
    )


def validate_schema(
    lazy_df: pl.LazyFrame,
) -> None:
    """Validate column names and key data types."""

    schema = lazy_df.collect_schema()

    actual_columns = schema.names()

    if actual_columns == EXPECTED_COLUMNS:

        report(
            "PASS",
            "column_structure",
            "All 16 expected staging columns are present in the expected order",
        )

    else:

        missing = [
            column
            for column in EXPECTED_COLUMNS
            if column not in actual_columns
        ]

        unexpected = [
            column
            for column in actual_columns
            if column not in EXPECTED_COLUMNS
        ]

        report(
            "FAIL",
            "column_structure",
            (
                f"Schema mismatch. "
                f"Missing={missing}; "
                f"Unexpected={unexpected}"
            ),
        )

    # -------------------------------------------------------------------------
    # Amount
    # -------------------------------------------------------------------------

    amount_dtype = schema.get(
        "amount"
    )

    if amount_dtype == pl.Float64:

        report(
            "PASS",
            "amount_dtype",
            "amount is Float64",
        )

    else:

        report(
            "FAIL",
            "amount_dtype",
            f"Expected Float64, found {amount_dtype}",
        )

    # -------------------------------------------------------------------------
    # Identifier types
    # -------------------------------------------------------------------------

    if schema.get("lei_code") == pl.String:

        report(
            "PASS",
            "lei_code_dtype",
            "lei_code is String",
        )

    else:

        report(
            "FAIL",
            "lei_code_dtype",
            f"Unexpected dtype: {schema.get('lei_code')}",
        )


# =============================================================================
# Population reconciliation
# =============================================================================

def validate_population(
    lazy_df: pl.LazyFrame,
) -> None:
    """Validate row counts and key cardinalities."""

    result = (
        lazy_df
        .select(
            [
                pl.len().alias(
                    "row_count"
                ),

                pl.col("amount")
                .null_count()
                .alias(
                    "null_amounts"
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

    row_count = int(
        result["row_count"][0]
    )

    null_amounts = int(
        result["null_amounts"][0]
    )

    distinct_lei = int(
        result["distinct_lei"][0]
    )

    distinct_item = int(
        result["distinct_item"][0]
    )

    distinct_period = int(
        result["distinct_period"][0]
    )

    # -------------------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------------------

    if row_count == EXPECTED_ROW_COUNT:

        report(
            "PASS",
            "row_count",
            f"{row_count:,} rows",
        )

    else:

        report(
            "FAIL",
            "row_count",
            (
                f"Expected {EXPECTED_ROW_COUNT:,}, "
                f"found {row_count:,}"
            ),
        )

    # -------------------------------------------------------------------------
    # Amount missingness
    # -------------------------------------------------------------------------

    if (
        null_amounts
        == EXPECTED_NULL_AMOUNT_COUNT
    ):

        report(
            "WARNING",
            "amount_nulls",
            (
                f"{null_amounts:,} NULL Amount values preserved "
                "from known raw source blanks"
            ),
        )

    else:

        report(
            "FAIL",
            "amount_nulls",
            (
                f"Expected {EXPECTED_NULL_AMOUNT_COUNT:,}, "
                f"found {null_amounts:,}"
            ),
        )

    # -------------------------------------------------------------------------
    # LEI cardinality
    # -------------------------------------------------------------------------

    if (
        distinct_lei
        == EXPECTED_DISTINCT_LEI_COUNT
    ):

        report(
            "PASS",
            "distinct_lei",
            f"{distinct_lei:,} distinct LEI codes",
        )

    else:

        report(
            "FAIL",
            "distinct_lei",
            (
                f"Expected {EXPECTED_DISTINCT_LEI_COUNT}, "
                f"found {distinct_lei}"
            ),
        )

    # -------------------------------------------------------------------------
    # Item cardinality
    # -------------------------------------------------------------------------

    if (
        distinct_item
        == EXPECTED_DISTINCT_ITEM_COUNT
    ):

        report(
            "PASS",
            "distinct_item",
            f"{distinct_item:,} distinct Item codes",
        )

    else:

        report(
            "FAIL",
            "distinct_item",
            (
                f"Expected {EXPECTED_DISTINCT_ITEM_COUNT}, "
                f"found {distinct_item}"
            ),
        )

    # -------------------------------------------------------------------------
    # Period cardinality
    # -------------------------------------------------------------------------

    if (
        distinct_period
        == len(EXPECTED_PERIODS)
    ):

        report(
            "PASS",
            "distinct_period_count",
            f"{distinct_period} distinct periods",
        )

    else:

        report(
            "FAIL",
            "distinct_period_count",
            (
                f"Expected {len(EXPECTED_PERIODS)}, "
                f"found {distinct_period}"
            ),
        )


# =============================================================================
# Period validation
# =============================================================================

def validate_period_values(
    lazy_df: pl.LazyFrame,
) -> None:
    """Confirm the exact validated reporting periods."""

    actual_periods = (
        lazy_df
        .select(
            pl.col("period")
            .unique()
            .sort()
        )
        .collect()[
            "period"
        ]
        .to_list()
    )

    if actual_periods == EXPECTED_PERIODS:

        report(
            "PASS",
            "period_values",
            f"{actual_periods}",
        )

    else:

        report(
            "FAIL",
            "period_values",
            (
                f"Expected {EXPECTED_PERIODS}, "
                f"found {actual_periods}"
            ),
        )


# =============================================================================
# Critical identifier completeness
# =============================================================================

def validate_critical_identifiers(
    lazy_df: pl.LazyFrame,
) -> None:
    """Check critical identifiers for NULL values."""

    critical_columns = [
        "lei_code",
        "period",
        "item",
    ]

    result = (
        lazy_df
        .select(
            [
                pl.col(column)
                .null_count()
                .alias(column)
                for column in critical_columns
            ]
        )
        .collect()
    )

    for column in critical_columns:

        null_count = int(
            result[column][0]
        )

        if null_count == 0:

            report(
                "PASS",
                f"null_{column}",
                "0 NULL value(s)",
            )

        else:

            report(
                "FAIL",
                f"null_{column}",
                f"{null_count:,} NULL value(s)",
            )


# =============================================================================
# All-other-banks aggregate
# =============================================================================

def validate_all_other_banks(
    lazy_df: pl.LazyFrame,
) -> None:
    """
    Confirm the recognised EBA aggregate identifier remains present.

    The aggregate is preserved rather than treated as an invalid institution.
    """

    result = (
        lazy_df
        .filter(
            pl.col("lei_code")
            == ALL_OTHER_BANKS_LEI
        )
        .select(
            pl.len().alias(
                "rows"
            )
        )
        .collect()
    )

    row_count = int(
        result["rows"][0]
    )

    if row_count > 0:

        report(
            "PASS",
            "all_other_banks",
            (
                f"Recognised EBA aggregate "
                f"{ALL_OTHER_BANKS_LEI} preserved "
                f"({row_count:,} row(s))"
            ),
        )

    else:

        report(
            "FAIL",
            "all_other_banks",
            (
                "Recognised EBA 'All other banks' "
                "aggregate is missing"
            ),
        )


# =============================================================================
# Traceability validation
# =============================================================================

def validate_traceability(
    lazy_df: pl.LazyFrame,
) -> None:
    """
    Validate preservation of source traceability fields.
    """

    traceability_columns = [
        "row",
        "column",
        "sheet",
    ]

    schema_columns = set(
        lazy_df
        .collect_schema()
        .names()
    )

    missing = [
        column
        for column in traceability_columns
        if column not in schema_columns
    ]

    if missing:

        report(
            "FAIL",
            "traceability_columns",
            (
                "Missing traceability column(s): "
                + ", ".join(missing)
            ),
        )

    else:

        report(
            "PASS",
            "traceability_columns",
            "row, column and sheet preserved",
        )


# =============================================================================
# Amount profile
# =============================================================================

def validate_amount_profile(
    lazy_df: pl.LazyFrame,
) -> None:
    """
    Report numerical Amount characteristics.

    Negative and zero values are informational, not failures.
    """

    result = (
        lazy_df
        .select(
            [
                pl.col("amount")
                .min()
                .alias(
                    "minimum"
                ),

                pl.col("amount")
                .max()
                .alias(
                    "maximum"
                ),

                (pl.col("amount") < 0)
                .sum()
                .alias(
                    "negative_count"
                ),

                (pl.col("amount") == 0)
                .sum()
                .alias(
                    "zero_count"
                ),

                (pl.col("amount") > 0)
                .sum()
                .alias(
                    "positive_count"
                ),
            ]
        )
        .collect()
    )

    print()
    print("AMOUNT PROFILE")
    print("-" * 100)

    print(
        f"Minimum         : "
        f"{result['minimum'][0]}"
    )

    print(
        f"Maximum         : "
        f"{result['maximum'][0]}"
    )

    print(
        f"Negative values : "
        f"{int(result['negative_count'][0]):,}"
    )

    print(
        f"Zero values     : "
        f"{int(result['zero_count'][0]):,}"
    )

    print(
        f"Positive values : "
        f"{int(result['positive_count'][0]):,}"
    )


# =============================================================================
# Summary
# =============================================================================

def print_summary() -> int:
    """Print final validation outcome."""

    total = (
        COUNTS.passed
        + COUNTS.warnings
        + COUNTS.failed
    )

    print()
    print("=" * 100)
    print("CREDIT-RISK STAGING VALIDATION SUMMARY")
    print("=" * 100)

    print(
        f"Total checks : {total}"
    )

    print(
        f"PASS         : {COUNTS.passed}"
    )

    print(
        f"WARNING      : {COUNTS.warnings}"
    )

    print(
        f"FAIL         : {COUNTS.failed}"
    )

    print("-" * 100)

    if COUNTS.failed > 0:

        print(
            "OVERALL STATUS : FAIL"
        )

        print(
            "Meaning        : Staged credit-risk data "
            "requires investigation before proceeding."
        )

        return 1

    if COUNTS.warnings > 0:

        print(
            "OVERALL STATUS : WARNING"
        )

        print(
            "Meaning        : Staging structure is valid. "
            "Known source missingness is preserved and documented."
        )

        return 0

    print(
        "OVERALL STATUS : PASS"
    )

    return 0


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """Run independent validation of the staged credit-risk table."""

    print("=" * 100)
    print("EBA CREDIT-RISK STAGING VALIDATION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staging file: {STAGING_FILE}"
    )

    print()

    print(
        "VALIDATION MODE: READ ONLY"
    )

    print(
        "No staged or raw data will be modified."
    )

    if not validate_file_exists():

        return print_summary()

    try:

        lazy_df = load_staging()

        lazy_df.collect_schema()

    except Exception as exc:

        report(
            "FAIL",
            "parquet_read",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

        return print_summary()

    report(
        "PASS",
        "parquet_read",
        "Staged Parquet opened successfully",
    )

    validate_schema(
        lazy_df
    )

    validate_population(
        lazy_df
    )

    validate_period_values(
        lazy_df
    )

    validate_critical_identifiers(
        lazy_df
    )

    validate_all_other_banks(
        lazy_df
    )

    validate_traceability(
        lazy_df
    )

    validate_amount_profile(
        lazy_df
    )

    return print_summary()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
