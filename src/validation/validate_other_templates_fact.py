"""
Independent validation of the analytical Other-Templates fact table.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Analytical fact:
    data/analytical/facts/fact_other_templates.parquet

Independent comparator:
    data/staging/facts/stg_other_templates.parquet

Purpose:
    Provide an independent approval gate after fact construction.

Validation areas:
    - file/readability
    - structure
    - data types
    - business-key completeness
    - business-grain uniqueness
    - source traceability
    - population reconciliation
    - Amount reconciliation
    - independent staging reconciliation
    - analytical dimension coverage
    - EBA aggregate entity control

READ ONLY:
    This script does not modify analytical, staging, reference, or raw data.
"""

from pathlib import Path
import sys

import polars as pl


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "facts"
    / "fact_other_templates.parquet"
)

STAGING_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
    / "stg_other_templates.parquet"
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

FACT_COLUMNS = [
    "lei_code",
    "period",
    "item",
    "assets_fv",
    "assets_stages",
    "exposure",
    "financial_instruments",
    "amount",
    "fin_end_year",
    "n_quarters",
    "row",
    "column",
    "sheet",
]


EXPECTED_DTYPES = {
    "lei_code": pl.String,
    "period": pl.Int64,
    "item": pl.Int64,
    "assets_fv": pl.Int64,
    "assets_stages": pl.Int64,
    "exposure": pl.Int64,
    "financial_instruments": pl.Int64,
    "amount": pl.Float64,
    "fin_end_year": pl.Int64,
    "n_quarters": pl.Int64,
    "row": pl.Int64,
    "column": pl.Int64,
    "sheet": pl.String,
}


BUSINESS_GRAIN = [
    "lei_code",
    "period",
    "item",
    "assets_fv",
    "assets_stages",
    "exposure",
    "financial_instruments",
]


TRACEABILITY_COLUMNS = [
    "row",
    "column",
    "sheet",
]


# =============================================================================
# APPROVED BASELINES
# =============================================================================

EXPECTED_ROW_COUNT = 91_277
EXPECTED_COLUMN_COUNT = 13

EXPECTED_DISTINCT_LEI = 120
EXPECTED_DISTINCT_ITEM = 158
EXPECTED_DISTINCT_PERIOD = 4
EXPECTED_DISTINCT_SHEET = 6

EXPECTED_PERIODS = [
    202409,
    202412,
    202503,
    202506,
]

EXPECTED_NULL_AMOUNT = 3_964
EXPECTED_NEGATIVE_AMOUNT = 8_233
EXPECTED_ZERO_AMOUNT = 31_525
EXPECTED_POSITIVE_AMOUNT = 47_555

EXPECTED_AMOUNT_MIN = -39593.3608
EXPECTED_AMOUNT_MAX = 2604849.7692097

ALL_OTHER_BANKS_LEI = "XXXXXXXXXXXXXXXXXXXX"
EXPECTED_ALL_OTHER_BANKS_ROWS = 824


# =============================================================================
# ANALYTICAL DIMENSION COVERAGE
# =============================================================================

DIMENSION_TESTS = {
    "lei_code": (
        "dim_institution.parquet",
        "lei_code",
    ),
    "period": (
        "dim_period.parquet",
        "period_code",
    ),
    "item": (
        "dim_item_analytic.parquet",
        "item",
    ),
    "assets_fv": (
        "dim_asset_fv.parquet",
        "assets_fv",
    ),
    "assets_stages": (
        "dim_asset_stage.parquet",
        "assets_stages",
    ),
    "exposure": (
        "dim_exposure.parquet",
        "exposure",
    ),
    "financial_instruments": (
        "dim_financial_instrument.parquet",
        "financial_instruments",
    ),
    "fin_end_year": (
        "dim_fin_year_end.parquet",
        "fin_end_year",
    ),
}


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

LINE = "=" * 100
SUBLINE = "-" * 100

results = []


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def record(
    status: str,
    area: str,
    check_name: str,
    message: str,
) -> None:
    results.append(
        {
            "status": status,
            "area": area,
            "check": check_name,
            "message": message,
        }
    )

    print(
        f"[{status:<7}] "
        f"{area:<24} | "
        f"{check_name:<28} | "
        f"{message}"
    )


def check(
    condition: bool,
    area: str,
    check_name: str,
    success: str,
    failure: str,
) -> None:
    if condition:
        record("PASS", area, check_name, success)
    else:
        record("FAIL", area, check_name, failure)


# =============================================================================
# AMOUNT PROFILE
# =============================================================================

def amount_profile(df: pl.DataFrame) -> dict:
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


# =============================================================================
# SAFE PARQUET READ
# =============================================================================

def read_parquet_checked(
    path: Path,
    area: str,
) -> pl.DataFrame | None:

    check(
        path.exists(),
        area,
        "file_exists",
        f"Found {path.name}",
        f"Missing file: {path}",
    )

    if not path.exists():
        return None

    try:
        df = pl.read_parquet(path)

        record(
            "PASS",
            area,
            "parquet_read",
            "Parquet file opened successfully",
        )

        return df

    except Exception as exc:
        record(
            "FAIL",
            area,
            "parquet_read",
            f"Unable to open Parquet: {exc}",
        )

        return None


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(LINE)
    print("EBA ANALYTICAL OTHER-TEMPLATES FACT VALIDATION")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    print(LINE)

    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Analytical fact     : {FACT_FILE}")
    print(f"Staging comparator  : {STAGING_FILE}")
    print(f"Dimension directory : {DIMENSION_DIR}")

    print()
    print("VALIDATION MODE: READ ONLY")
    print(
        "No analytical, staging, reference, or raw data "
        "will be modified."
    )

    # =========================================================================
    # 1. FACT FILE
    # =========================================================================

    section("1. ANALYTICAL FACT FILE")

    fact = read_parquet_checked(
        FACT_FILE,
        "fact_other_templates",
    )

    if fact is None:
        print()
        print("Validation cannot continue because the analytical fact is unreadable.")
        sys.exit(1)

    # =========================================================================
    # 2. STRUCTURAL VALIDATION
    # =========================================================================

    section("2. STRUCTURAL VALIDATION")

    check(
        fact.height == EXPECTED_ROW_COUNT,
        "fact_other_templates",
        "row_count",
        f"{fact.height:,} rows match approved baseline",
        (
            f"Expected {EXPECTED_ROW_COUNT:,} rows, "
            f"found {fact.height:,}"
        ),
    )

    check(
        fact.width == EXPECTED_COLUMN_COUNT,
        "fact_other_templates",
        "column_count",
        f"{fact.width} columns match approved design",
        (
            f"Expected {EXPECTED_COLUMN_COUNT} columns, "
            f"found {fact.width}"
        ),
    )

    check(
        fact.columns == FACT_COLUMNS,
        "fact_other_templates",
        "column_order",
        "All approved fact columns are present in the expected order",
        (
            f"Column mismatch. Expected {FACT_COLUMNS}; "
            f"found {fact.columns}"
        ),
    )

    check(
        len(fact.columns) == len(set(fact.columns)),
        "fact_other_templates",
        "duplicate_columns",
        "Column names are unique",
        "Duplicate column names detected",
    )

    # =========================================================================
    # 3. DATA TYPES
    # =========================================================================

    section("3. DATA-TYPE VALIDATION")

    for column_name, expected_dtype in EXPECTED_DTYPES.items():

        actual_dtype = fact.schema.get(column_name)

        check(
            actual_dtype == expected_dtype,
            "fact_other_templates",
            f"dtype_{column_name}",
            f"{column_name} is {actual_dtype}",
            (
                f"{column_name}: expected {expected_dtype}, "
                f"found {actual_dtype}"
            ),
        )

    # =========================================================================
    # 4. BUSINESS KEY COMPLETENESS
    # =========================================================================

    section("4. BUSINESS KEY COMPLETENESS")

    for column_name in BUSINESS_GRAIN:

        null_count = fact[column_name].null_count()

        check(
            null_count == 0,
            "fact_other_templates",
            f"null_{column_name}",
            f"No NULL values in '{column_name}'",
            f"{null_count:,} NULL value(s) in '{column_name}'",
        )

    blank_lei_count = fact.filter(
        pl.col("lei_code").str.strip_chars() == ""
    ).height

    check(
        blank_lei_count == 0,
        "fact_other_templates",
        "blank_lei_code",
        "No blank LEI codes",
        f"{blank_lei_count:,} blank LEI code(s)",
    )

    # =========================================================================
    # 5. BUSINESS GRAIN
    # =========================================================================

    section("5. BUSINESS-GRAIN VALIDATION")

    grain_count = (
        fact
        .select(BUSINESS_GRAIN)
        .unique()
        .height
    )

    check(
        grain_count == fact.height,
        "fact_other_templates",
        "business_grain_unique",
        (
            "Declared seven-column business grain is unique across "
            f"{fact.height:,} rows"
        ),
        (
            f"Business grain produced only {grain_count:,} "
            f"unique combinations across {fact.height:,} rows"
        ),
    )

    exact_unique = fact.unique().height

    check(
        exact_unique == fact.height,
        "fact_other_templates",
        "full_row_uniqueness",
        "No exact full-row duplicates found",
        (
            f"{fact.height - exact_unique:,} "
            "exact duplicate row(s) found"
        ),
    )

    # =========================================================================
    # 6. SOURCE TRACEABILITY
    # =========================================================================

    section("6. SOURCE TRACEABILITY")

    for column_name in TRACEABILITY_COLUMNS:

        check(
            column_name in fact.columns,
            "fact_other_templates",
            f"traceability_{column_name}",
            f"Traceability field '{column_name}' exists",
            f"Traceability field '{column_name}' missing",
        )

    # =========================================================================
    # 7. POPULATION PROFILE
    # =========================================================================

    section("7. FACT POPULATION PROFILE")

    check(
        fact["lei_code"].n_unique() == EXPECTED_DISTINCT_LEI,
        "fact_other_templates",
        "distinct_lei",
        (
            f"{EXPECTED_DISTINCT_LEI} distinct "
            "reporting entity code(s)"
        ),
        (
            "Unexpected LEI population: "
            f"{fact['lei_code'].n_unique()}"
        ),
    )

    check(
        fact["item"].n_unique() == EXPECTED_DISTINCT_ITEM,
        "fact_other_templates",
        "distinct_item",
        f"{EXPECTED_DISTINCT_ITEM} distinct Item code(s)",
        (
            "Unexpected Item population: "
            f"{fact['item'].n_unique()}"
        ),
    )

    check(
        fact["period"].n_unique() == EXPECTED_DISTINCT_PERIOD,
        "fact_other_templates",
        "distinct_period",
        f"{EXPECTED_DISTINCT_PERIOD} distinct reporting period(s)",
        (
            "Unexpected reporting-period population: "
            f"{fact['period'].n_unique()}"
        ),
    )

    actual_periods = (
        fact
        .select("period")
        .unique()
        .sort("period")
        .get_column("period")
        .to_list()
    )

    check(
        actual_periods == EXPECTED_PERIODS,
        "fact_other_templates",
        "period_values",
        f"Reporting periods reconcile: {actual_periods}",
        (
            f"Expected periods {EXPECTED_PERIODS}; "
            f"found {actual_periods}"
        ),
    )

    check(
        fact["sheet"].n_unique() == EXPECTED_DISTINCT_SHEET,
        "fact_other_templates",
        "distinct_sheet",
        f"{EXPECTED_DISTINCT_SHEET} distinct source template(s)",
        (
            "Unexpected source-template population: "
            f"{fact['sheet'].n_unique()}"
        ),
    )

    aggregate_rows = fact.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    ).height

    check(
        aggregate_rows == EXPECTED_ALL_OTHER_BANKS_ROWS,
        "fact_other_templates",
        "all_other_banks",
        (
            "Recognised EBA 'All other banks' aggregate preserved: "
            f"{aggregate_rows:,} row(s)"
        ),
        (
            f"Expected {EXPECTED_ALL_OTHER_BANKS_ROWS:,} "
            f"aggregate rows, found {aggregate_rows:,}"
        ),
    )

    # =========================================================================
    # 8. AMOUNT VALIDATION
    # =========================================================================

    section("8. AMOUNT VALIDATION")

    amount = amount_profile(fact)

    amount_tests = {
        "null_count": EXPECTED_NULL_AMOUNT,
        "negative_count": EXPECTED_NEGATIVE_AMOUNT,
        "zero_count": EXPECTED_ZERO_AMOUNT,
        "positive_count": EXPECTED_POSITIVE_AMOUNT,
        "minimum": EXPECTED_AMOUNT_MIN,
        "maximum": EXPECTED_AMOUNT_MAX,
    }

    for metric, expected_value in amount_tests.items():

        actual_value = amount[metric]

        check(
            actual_value == expected_value,
            "fact_other_templates",
            f"amount_{metric}",
            f"{metric} reconciles to approved baseline ({actual_value})",
            (
                f"{metric}: expected {expected_value}, "
                f"found {actual_value}"
            ),
        )

    print()
    print("Amount profile:")
    print(SUBLINE)
    print(f"NULL values     : {amount['null_count']:,}")
    print(f"Negative values : {amount['negative_count']:,}")
    print(f"Zero values     : {amount['zero_count']:,}")
    print(f"Positive values : {amount['positive_count']:,}")
    print(f"Minimum         : {amount['minimum']}")
    print(f"Maximum         : {amount['maximum']}")

    print()
    print(
        "Interpretation: NULL Amount values remain distinct "
        "from reported numerical zero."
    )
    print(
        "Negative values are retained for investigation/interpretation "
        "and are not automatically treated as data-quality failures."
    )

    # =========================================================================
    # 9. INDEPENDENT STAGING RECONCILIATION
    # =========================================================================

    section("9. INDEPENDENT STAGING RECONCILIATION")

    staging = read_parquet_checked(
        STAGING_FILE,
        "stg_other_templates",
    )

    if staging is not None:

        check(
            fact.height == staging.height,
            "RECONCILIATION",
            "row_count",
            (
                "Fact and staging both contain "
                f"{fact.height:,} rows"
            ),
            (
                f"Fact rows={fact.height:,}; "
                f"staging rows={staging.height:,}"
            ),
        )

        source_columns_available = all(
            column_name in staging.columns
            for column_name in FACT_COLUMNS
        )

        check(
            source_columns_available,
            "RECONCILIATION",
            "source_columns",
            "All fact columns are available in staging",
            "One or more approved fact columns are missing from staging",
        )

        if source_columns_available:

            staging_projection = staging.select(FACT_COLUMNS)

            check(
                fact.equals(staging_projection),
                "RECONCILIATION",
                "full_projection",
                (
                    "Analytical fact exactly matches the approved "
                    "staging column projection"
                ),
                (
                    "Analytical fact does not exactly match "
                    "the staging projection"
                ),
            )

            staging_amount = amount_profile(staging_projection)

            for metric in [
                "null_count",
                "negative_count",
                "zero_count",
                "positive_count",
                "minimum",
                "maximum",
            ]:

                check(
                    amount[metric] == staging_amount[metric],
                    "RECONCILIATION",
                    f"amount_{metric}",
                    (
                        f"Amount {metric} reconciles "
                        "exactly to staging"
                    ),
                    (
                        f"Amount {metric} differs: "
                        f"fact={amount[metric]}, "
                        f"staging={staging_amount[metric]}"
                    ),
                )

    # =========================================================================
    # 10. ANALYTICAL DIMENSION COVERAGE
    # =========================================================================

    section("10. ANALYTICAL DIMENSION COVERAGE")

    for fact_column, (
        dimension_filename,
        dimension_key,
    ) in DIMENSION_TESTS.items():

        dimension_path = DIMENSION_DIR / dimension_filename

        dimension = read_parquet_checked(
            dimension_path,
            dimension_filename.replace(".parquet", ""),
        )

        if dimension is None:
            continue

        check(
            dimension_key in dimension.columns,
            "DIMENSION COVERAGE",
            fact_column,
            (
                f"Dimension key '{dimension_key}' "
                "exists"
            ),
            (
                f"Dimension key '{dimension_key}' "
                f"missing from {dimension_filename}"
            ),
        )

        if dimension_key not in dimension.columns:
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
            .select(dimension_key)
            .drop_nulls()
            .unique()
            .get_column(dimension_key)
            .to_list()
        )

        missing_values = fact_values - dimension_values

        check(
            len(missing_values) == 0,
            "DIMENSION COVERAGE",
            fact_column,
            (
                f"All {len(fact_values)} distinct fact "
                f"'{fact_column}' value(s) match "
                f"{dimension_filename}"
            ),
            (
                f"{len(missing_values)} fact value(s) missing "
                f"from {dimension_filename}: "
                f"{sorted(missing_values)[:10]}"
            ),
        )

    # =========================================================================
    # 11. EBA AGGREGATE ENTITY CONTROL
    # =========================================================================

    section("11. EBA AGGREGATE ENTITY CONTROL")

    institution_path = (
        DIMENSION_DIR
        / "dim_institution.parquet"
    )

    if institution_path.exists():

        institution = pl.read_parquet(institution_path)

        aggregate = institution.filter(
            pl.col("lei_code") == ALL_OTHER_BANKS_LEI
        )

        check(
            aggregate.height == 1,
            "dim_institution",
            "aggregate_member_count",
            (
                "Exactly one analytical institution member represents "
                "EBA 'All other banks'"
            ),
            (
                "Expected exactly one 'All other banks' "
                f"dimension member; found {aggregate.height}"
            ),
        )

        if aggregate.height == 1:

            if "is_aggregate" in aggregate.columns:

                check(
                    bool(aggregate["is_aggregate"][0]) is True,
                    "dim_institution",
                    "aggregate_flag",
                    (
                        "'All other banks' is correctly classified "
                        "as an aggregate entity"
                    ),
                    (
                        "'All other banks' does not have "
                        "is_aggregate=True"
                    ),
                )

            if "entity_type" in aggregate.columns:

                check(
                    aggregate["entity_type"][0] == "aggregate",
                    "dim_institution",
                    "entity_type",
                    "'All other banks' entity_type is 'aggregate'",
                    (
                        "'All other banks' entity_type is "
                        f"{aggregate['entity_type'][0]!r}"
                    ),
                )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    section("OTHER-TEMPLATES FACT VALIDATION SUMMARY")

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

    if fail_count == 0 and warning_count == 0:

        print("OVERALL STATUS : PASS")
        print(
            "INTERPRETATION  : The analytical other-templates fact "
            "passed the configured structural, reconciliation, grain, "
            "amount, traceability, and dimension-coverage controls."
        )
        print(
            "ACTION          : fact_other_templates can be considered "
            "for approval for downstream analytical development."
        )

    elif fail_count == 0:

        print("OVERALL STATUS : WARNING")
        print(
            "INTERPRETATION  : No blocking validation failure was found, "
            "but one or more documented observations require review."
        )
        print(
            "ACTION          : Review warnings before approving the fact."
        )

    else:

        print("OVERALL STATUS : FAIL")
        print(
            "INTERPRETATION  : One or more blocking controls failed."
        )
        print(
            "ACTION          : Do not approve fact_other_templates "
            "until the failures are investigated."
        )

    print()
    print("VALIDATION MODE : READ ONLY")
    print(
        "No analytical, staging, reference, or raw data was modified."
    )

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
