"""
Independent validation of the analytical credit-risk fact table.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Validated fact:
    data/analytical/facts/fact_credit_risk.parquet

Reconciliation source:
    data/staging/facts/stg_credit_risk.parquet

Analytical dimensions:
    data/analytical/dimensions/

Important:
    READ ONLY.
    This script does not modify, clean, deduplicate, aggregate,
    overwrite, or otherwise transform any project data.
"""

from pathlib import Path
import sys

import polars as pl


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "facts"
    / "fact_credit_risk.parquet"
)

STAGING_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
    / "stg_credit_risk.parquet"
)

DIMENSION_DIR = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
)


# =============================================================================
# APPROVED FACT DESIGN
# =============================================================================

EXPECTED_FACT_COLUMNS = [
    "lei_code",
    "period",
    "item",
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

BUSINESS_GRAIN = [
    "lei_code",
    "period",
    "item",
    "portfolio",
    "country",
    "exposure",
    "status",
    "perf_status",
    "nace_codes",
]

TRACEABILITY_COLUMNS = [
    "row",
    "column",
    "sheet",
]

EXPECTED_DTYPES = {
    "lei_code": pl.String,
    "period": pl.Int64,
    "item": pl.Int64,
    "portfolio": pl.Int64,
    "country": pl.Int64,
    "country_rank": pl.Int64,
    "exposure": pl.Int64,
    "status": pl.Int64,
    "perf_status": pl.Int64,
    "nace_codes": pl.Int64,
    "amount": pl.Float64,
    "row": pl.Int64,
    "column": pl.Int64,
    "sheet": pl.String,
}


# =============================================================================
# VALIDATED BASELINE EXPECTATIONS
# =============================================================================

EXPECTED_ROW_COUNT = 648_951
EXPECTED_COLUMN_COUNT = 14
EXPECTED_NULL_AMOUNT_COUNT = 1_637
EXPECTED_DISTINCT_LEI_COUNT = 120
EXPECTED_DISTINCT_ITEM_COUNT = 89
EXPECTED_DISTINCT_PERIOD_COUNT = 4

EXPECTED_PERIODS = [
    202409,
    202412,
    202503,
    202506,
]

ALL_OTHER_BANKS_LEI = "XXXXXXXXXXXXXXXXXXXX"
EXPECTED_ALL_OTHER_BANKS_ROWS = 8_676


# =============================================================================
# DIMENSION COVERAGE CONFIGURATION
# =============================================================================

DIMENSION_CHECKS = {
    "lei_code": {
        "file": "dim_institution.parquet",
        "dimension_column": "lei_code",
    },
    "period": {
        "file": "dim_period.parquet",
        "dimension_column": "period_code",
    },
    "item": {
        "file": "dim_item_analytic.parquet",
        "dimension_column": "item",
    },
    "portfolio": {
        "file": "dim_portfolio.parquet",
        "dimension_column": "portfolio",
    },
    "country": {
        "file": "dim_country.parquet",
        "dimension_column": "country",
    },
    "exposure": {
        "file": "dim_exposure.parquet",
        "dimension_column": "exposure",
    },
    "status": {
        "file": "dim_status.parquet",
        "dimension_column": "status",
    },
    "perf_status": {
        "file": "dim_perf_status.parquet",
        "dimension_column": "perf_status",
    },
    "nace_codes": {
        "file": "dim_nace.parquet",
        "dimension_column": "nace_codes",
    },
}


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

LINE = "=" * 100
SUBLINE = "-" * 100

results = []


def section(title: str) -> None:
    """Print a formatted section heading."""
    print()
    print(LINE)
    print(title)
    print(LINE)


def record(
    status: str,
    area: str,
    check: str,
    message: str,
) -> None:
    """Record and print a validation result."""
    results.append(
        {
            "status": status,
            "area": area,
            "check": check,
            "message": message,
        }
    )

    print(
        f"[{status:<7}] "
        f"{area:<20} | "
        f"{check:<28} | "
        f"{message}"
    )


def passed(area: str, check: str, message: str) -> None:
    record("PASS", area, check, message)


def warning(area: str, check: str, message: str) -> None:
    record("WARNING", area, check, message)


def failed(area: str, check: str, message: str) -> None:
    record("FAIL", area, check, message)


def require(
    condition: bool,
    area: str,
    check: str,
    success_message: str,
    failure_message: str,
) -> None:
    """Record a PASS or FAIL without modifying data."""
    if condition:
        passed(area, check, success_message)
    else:
        failed(area, check, failure_message)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def load_parquet(
    path: Path,
    area: str,
) -> pl.DataFrame | None:
    """Read a Parquet file safely for validation."""
    if not path.exists():
        failed(
            area,
            "file_exists",
            f"File not found: {path}",
        )
        return None

    passed(
        area,
        "file_exists",
        f"Found {path.name}",
    )

    try:
        df = pl.read_parquet(path)
    except Exception as exc:
        failed(
            area,
            "parquet_read",
            f"Could not read Parquet file: {exc}",
        )
        return None

    passed(
        area,
        "parquet_read",
        "Parquet file opened successfully",
    )

    return df


def get_amount_profile(df: pl.DataFrame) -> dict:
    """Return Amount control statistics."""
    return df.select(
        [
            pl.col("amount").null_count().alias("null_count"),
            (pl.col("amount") < 0).sum().alias("negative_count"),
            (pl.col("amount") == 0).sum().alias("zero_count"),
            (pl.col("amount") > 0).sum().alias("positive_count"),
            pl.col("amount").min().alias("minimum"),
            pl.col("amount").max().alias("maximum"),
        ]
    ).row(0, named=True)


def sorted_unique_values(
    df: pl.DataFrame,
    column_name: str,
) -> list:
    """Return sorted unique non-null values from one column."""
    return (
        df
        .select(column_name)
        .drop_nulls()
        .unique()
        .sort(column_name)
        .get_column(column_name)
        .to_list()
    )


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def main() -> None:
    print(LINE)
    print("EBA ANALYTICAL CREDIT-RISK FACT VALIDATION")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    print(LINE)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Analytical fact     : {FACT_FILE}")
    print(f"Staging comparator  : {STAGING_FILE}")
    print(f"Dimension directory : {DIMENSION_DIR}")
    print()
    print("VALIDATION MODE: READ ONLY")
    print("No analytical, staging, reference, or raw data will be modified.")

    # =========================================================================
    # 1. LOAD FACT
    # =========================================================================

    section("1. ANALYTICAL FACT FILE")

    fact = load_parquet(
        FACT_FILE,
        "fact_credit_risk",
    )

    if fact is None:
        print_summary()
        sys.exit(1)

    # =========================================================================
    # 2. STRUCTURAL VALIDATION
    # =========================================================================

    section("2. STRUCTURAL VALIDATION")

    require(
        fact.height == EXPECTED_ROW_COUNT,
        "fact_credit_risk",
        "row_count",
        f"{fact.height:,} rows match approved baseline",
        (
            f"Expected {EXPECTED_ROW_COUNT:,} rows, "
            f"found {fact.height:,}"
        ),
    )

    require(
        fact.width == EXPECTED_COLUMN_COUNT,
        "fact_credit_risk",
        "column_count",
        f"{fact.width} columns match approved design",
        (
            f"Expected {EXPECTED_COLUMN_COUNT} columns, "
            f"found {fact.width}"
        ),
    )

    require(
        fact.columns == EXPECTED_FACT_COLUMNS,
        "fact_credit_risk",
        "column_order",
        "All approved fact columns are present in the expected order",
        (
            "Fact schema differs from approved design. "
            f"Actual columns: {fact.columns}"
        ),
    )

    duplicate_column_names = (
        len(fact.columns) - len(set(fact.columns))
    )

    require(
        duplicate_column_names == 0,
        "fact_credit_risk",
        "duplicate_columns",
        "Column names are unique",
        f"{duplicate_column_names} duplicate column name(s) detected",
    )

    # =========================================================================
    # 3. DATA-TYPE VALIDATION
    # =========================================================================

    section("3. DATA-TYPE VALIDATION")

    for column_name, expected_dtype in EXPECTED_DTYPES.items():
        if column_name not in fact.columns:
            failed(
                "fact_credit_risk",
                f"dtype_{column_name}",
                f"Column '{column_name}' is missing",
            )
            continue

        actual_dtype = fact.schema[column_name]

        require(
            actual_dtype == expected_dtype,
            "fact_credit_risk",
            f"dtype_{column_name}",
            f"{column_name} is {actual_dtype}",
            (
                f"{column_name}: expected {expected_dtype}, "
                f"found {actual_dtype}"
            ),
        )

    # =========================================================================
    # 4. KEY COMPLETENESS
    # =========================================================================

    section("4. BUSINESS KEY COMPLETENESS")

    for column_name in BUSINESS_GRAIN:
        if column_name not in fact.columns:
            failed(
                "fact_credit_risk",
                f"null_{column_name}",
                f"Required grain column '{column_name}' missing",
            )
            continue

        null_count = fact[column_name].null_count()

        require(
            null_count == 0,
            "fact_credit_risk",
            f"null_{column_name}",
            f"No NULL values in '{column_name}'",
            f"{null_count:,} NULL value(s) in '{column_name}'",
        )

    # String identifier blank check
    if "lei_code" in fact.columns:
        blank_lei_count = fact.filter(
            pl.col("lei_code").str.strip_chars() == ""
        ).height

        require(
            blank_lei_count == 0,
            "fact_credit_risk",
            "blank_lei_code",
            "No blank LEI codes",
            f"{blank_lei_count:,} blank LEI code(s) detected",
        )

    # =========================================================================
    # 5. BUSINESS GRAIN
    # =========================================================================

    section("5. BUSINESS-GRAIN VALIDATION")

    if all(column in fact.columns for column in BUSINESS_GRAIN):
        unique_grain_count = (
            fact
            .select(BUSINESS_GRAIN)
            .unique()
            .height
        )

        duplicate_grain_rows = (
            fact.height - unique_grain_count
        )

        require(
            unique_grain_count == fact.height,
            "fact_credit_risk",
            "business_grain_unique",
            (
                "Declared nine-column business grain is unique "
                f"across {fact.height:,} rows"
            ),
            (
                f"Business grain is not unique: "
                f"{duplicate_grain_rows:,} excess row(s)"
            ),
        )

    exact_unique_count = fact.unique().height

    require(
        exact_unique_count == fact.height,
        "fact_credit_risk",
        "full_row_uniqueness",
        "No exact full-row duplicates found",
        (
            f"{fact.height - exact_unique_count:,} exact duplicate "
            "row(s) detected"
        ),
    )

    # =========================================================================
    # 6. TRACEABILITY
    # =========================================================================

    section("6. SOURCE TRACEABILITY")

    for column_name in TRACEABILITY_COLUMNS:
        require(
            column_name in fact.columns,
            "fact_credit_risk",
            f"traceability_{column_name}",
            f"Traceability field '{column_name}' exists",
            f"Traceability field '{column_name}' is missing",
        )

    # =========================================================================
    # 7. POPULATION PROFILE
    # =========================================================================

    section("7. FACT POPULATION PROFILE")

    distinct_lei = fact["lei_code"].n_unique()
    distinct_item = fact["item"].n_unique()
    distinct_period = fact["period"].n_unique()

    require(
        distinct_lei == EXPECTED_DISTINCT_LEI_COUNT,
        "fact_credit_risk",
        "distinct_lei",
        f"{distinct_lei} distinct reporting entity code(s)",
        (
            f"Expected {EXPECTED_DISTINCT_LEI_COUNT}, "
            f"found {distinct_lei}"
        ),
    )

    require(
        distinct_item == EXPECTED_DISTINCT_ITEM_COUNT,
        "fact_credit_risk",
        "distinct_item",
        f"{distinct_item} distinct Item code(s)",
        (
            f"Expected {EXPECTED_DISTINCT_ITEM_COUNT}, "
            f"found {distinct_item}"
        ),
    )

    require(
        distinct_period == EXPECTED_DISTINCT_PERIOD_COUNT,
        "fact_credit_risk",
        "distinct_period",
        f"{distinct_period} distinct reporting period(s)",
        (
            f"Expected {EXPECTED_DISTINCT_PERIOD_COUNT}, "
            f"found {distinct_period}"
        ),
    )

    actual_periods = sorted_unique_values(
        fact,
        "period",
    )

    require(
        actual_periods == EXPECTED_PERIODS,
        "fact_credit_risk",
        "period_values",
        f"Reporting periods reconcile: {actual_periods}",
        (
            f"Unexpected reporting periods. "
            f"Expected {EXPECTED_PERIODS}, "
            f"found {actual_periods}"
        ),
    )

    aggregate_rows = fact.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    ).height

    require(
        aggregate_rows == EXPECTED_ALL_OTHER_BANKS_ROWS,
        "fact_credit_risk",
        "all_other_banks",
        (
            "Recognised EBA 'All other banks' aggregate preserved: "
            f"{aggregate_rows:,} row(s)"
        ),
        (
            "Unexpected 'All other banks' population. "
            f"Expected {EXPECTED_ALL_OTHER_BANKS_ROWS:,}, "
            f"found {aggregate_rows:,}"
        ),
    )

    # =========================================================================
    # 8. AMOUNT VALIDATION
    # =========================================================================

    section("8. AMOUNT VALIDATION")

    amount_profile = get_amount_profile(fact)

    require(
        amount_profile["null_count"]
        == EXPECTED_NULL_AMOUNT_COUNT,
        "fact_credit_risk",
        "amount_nulls",
        (
            f"Known source missingness preserved: "
            f"{amount_profile['null_count']:,} NULL Amount value(s)"
        ),
        (
            f"Expected {EXPECTED_NULL_AMOUNT_COUNT:,} NULL Amount "
            f"value(s), found {amount_profile['null_count']:,}"
        ),
    )

    print()
    print("Amount profile:")
    print(SUBLINE)
    print(
        f"NULL values     : "
        f"{amount_profile['null_count']:,}"
    )
    print(
        f"Negative values : "
        f"{amount_profile['negative_count']:,}"
    )
    print(
        f"Zero values     : "
        f"{amount_profile['zero_count']:,}"
    )
    print(
        f"Positive values : "
        f"{amount_profile['positive_count']:,}"
    )
    print(
        f"Minimum         : "
        f"{amount_profile['minimum']}"
    )
    print(
        f"Maximum         : "
        f"{amount_profile['maximum']}"
    )
    print()
    print(
        "Interpretation: NULL Amount values remain distinct from "
        "reported numerical zero."
    )
    print(
        "Negative values are retained for investigation/interpretation "
        "and are not automatically treated as data-quality failures."
    )

    # =========================================================================
    # 9. STAGING RECONCILIATION
    # =========================================================================

    section("9. INDEPENDENT STAGING RECONCILIATION")

    staging = load_parquet(
        STAGING_FILE,
        "stg_credit_risk",
    )

    if staging is not None:
        require(
            staging.height == fact.height,
            "RECONCILIATION",
            "row_count",
            (
                f"Fact and staging both contain "
                f"{fact.height:,} rows"
            ),
            (
                f"Staging rows={staging.height:,}; "
                f"fact rows={fact.height:,}"
            ),
        )

        staging_required_columns = set(
            EXPECTED_FACT_COLUMNS
        )

        missing_staging_columns = (
            staging_required_columns
            - set(staging.columns)
        )

        require(
            not missing_staging_columns,
            "RECONCILIATION",
            "source_columns",
            "All fact columns are available in staging",
            (
                "Staging is missing required comparison columns: "
                f"{sorted(missing_staging_columns)}"
            ),
        )

        if not missing_staging_columns:
            staging_projection = staging.select(
                EXPECTED_FACT_COLUMNS
            )

            require(
                staging_projection.equals(fact),
                "RECONCILIATION",
                "full_projection",
                (
                    "Analytical fact exactly matches the approved "
                    "staging column projection"
                ),
                (
                    "Analytical fact differs from the equivalent "
                    "staging projection"
                ),
            )

            staging_amount_profile = get_amount_profile(
                staging_projection
            )

            for metric in [
                "null_count",
                "negative_count",
                "zero_count",
                "positive_count",
                "minimum",
                "maximum",
            ]:
                require(
                    staging_amount_profile[metric]
                    == amount_profile[metric],
                    "RECONCILIATION",
                    f"amount_{metric}",
                    (
                        f"Amount {metric} reconciles exactly "
                        "to staging"
                    ),
                    (
                        f"Amount {metric} mismatch: "
                        f"staging="
                        f"{staging_amount_profile[metric]}, "
                        f"fact={amount_profile[metric]}"
                    ),
                )

    # =========================================================================
    # 10. ANALYTICAL DIMENSION COVERAGE
    # =========================================================================

    section("10. ANALYTICAL DIMENSION COVERAGE")

    for fact_column, config in DIMENSION_CHECKS.items():
        dimension_file = (
            DIMENSION_DIR / config["file"]
        )

        dimension_column = config[
            "dimension_column"
        ]

        dimension_area = config["file"].replace(
            ".parquet",
            "",
        )

        if not dimension_file.exists():
            failed(
                dimension_area,
                "file_exists",
                f"Dimension file not found: {dimension_file}",
            )
            continue

        passed(
            dimension_area,
            "file_exists",
            f"Found {config['file']}",
        )

        try:
            dimension = pl.read_parquet(
                dimension_file
            )
        except Exception as exc:
            failed(
                dimension_area,
                "parquet_read",
                f"Could not read dimension: {exc}",
            )
            continue

        passed(
            dimension_area,
            "parquet_read",
            "Dimension opened successfully",
        )

        if dimension_column not in dimension.columns:
            failed(
                dimension_area,
                "key_column",
                (
                    f"Expected key '{dimension_column}' "
                    "not found"
                ),
            )
            continue

        fact_values = set(
            fact
            .select(fact_column)
            .drop_nulls()
            .unique()
            .get_column(fact_column)
            .to_list()
        )

        dimension_values = set(
            dimension
            .select(dimension_column)
            .drop_nulls()
            .unique()
            .get_column(dimension_column)
            .to_list()
        )

        missing_values = (
            fact_values - dimension_values
        )

        require(
            len(missing_values) == 0,
            "DIMENSION COVERAGE",
            fact_column,
            (
                f"All {len(fact_values):,} distinct fact "
                f"'{fact_column}' value(s) match "
                f"{config['file']}"
            ),
            (
                f"{len(missing_values):,} fact value(s) "
                f"missing from {config['file']}. "
                f"Examples: "
                f"{sorted(missing_values)[:10]}"
            ),
        )

    # =========================================================================
    # 11. SPECIAL INSTITUTION CONTROL
    # =========================================================================

    section("11. EBA AGGREGATE ENTITY CONTROL")

    institution_file = (
        DIMENSION_DIR / "dim_institution.parquet"
    )

    if institution_file.exists():
        institution = pl.read_parquet(
            institution_file
        )

        aggregate_dimension_rows = (
            institution
            .filter(
                pl.col("lei_code")
                == ALL_OTHER_BANKS_LEI
            )
        )

        require(
            aggregate_dimension_rows.height == 1,
            "dim_institution",
            "aggregate_member_count",
            (
                "Exactly one analytical institution member "
                "represents EBA 'All other banks'"
            ),
            (
                f"Expected one aggregate dimension member, "
                f"found {aggregate_dimension_rows.height}"
            ),
        )

        if (
            aggregate_dimension_rows.height == 1
            and "is_aggregate"
            in institution.columns
        ):
            is_aggregate = (
                aggregate_dimension_rows
                .get_column("is_aggregate")
                .item()
            )

            require(
                is_aggregate is True,
                "dim_institution",
                "aggregate_flag",
                (
                    "'All other banks' is correctly classified "
                    "as an aggregate entity"
                ),
                (
                    "'All other banks' exists but "
                    "is_aggregate is not True"
                ),
            )

        if (
            aggregate_dimension_rows.height == 1
            and "entity_type"
            in institution.columns
        ):
            entity_type = (
                aggregate_dimension_rows
                .get_column("entity_type")
                .item()
            )

            require(
                entity_type == "aggregate",
                "dim_institution",
                "entity_type",
                (
                    "'All other banks' entity_type "
                    "is 'aggregate'"
                ),
                (
                    "'All other banks' has unexpected "
                    f"entity_type: {entity_type}"
                ),
            )

    # =========================================================================
    # 12. SUMMARY
    # =========================================================================

    print_summary()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_summary() -> None:
    section("CREDIT-RISK FACT VALIDATION SUMMARY")

    total_checks = len(results)

    pass_count = sum(
        result["status"] == "PASS"
        for result in results
    )

    warning_count = sum(
        result["status"] == "WARNING"
        for result in results
    )

    fail_count = sum(
        result["status"] == "FAIL"
        for result in results
    )

    print(f"Total checks : {total_checks}")
    print(f"PASS         : {pass_count}")
    print(f"WARNING      : {warning_count}")
    print(f"FAIL         : {fail_count}")
    print(SUBLINE)

    if fail_count > 0:
        print("OVERALL STATUS : FAIL")
        print(
            "INTERPRETATION  : One or more blocking analytical-fact "
            "controls failed."
        )
        print(
            "ACTION          : Investigate before using "
            "fact_credit_risk for KPIs, marts, dashboards, "
            "or management reporting."
        )

    elif warning_count > 0:
        print("OVERALL STATUS : WARNING")
        print(
            "INTERPRETATION  : No blocking failures were found, "
            "but documented observations require awareness."
        )
        print(
            "ACTION          : Review warnings before formally "
            "approving the analytical fact."
        )

    else:
        print("OVERALL STATUS : PASS")
        print(
            "INTERPRETATION  : The analytical credit-risk fact passed "
            "the configured structural, reconciliation, grain, "
            "amount, traceability, and dimension-coverage controls."
        )
        print(
            "ACTION          : fact_credit_risk can be considered "
            "for approval for downstream analytical development."
        )

    print()
    print("VALIDATION MODE : READ ONLY")
    print(
        "No analytical, staging, reference, or raw data was modified."
    )


if __name__ == "__main__":
    main()
