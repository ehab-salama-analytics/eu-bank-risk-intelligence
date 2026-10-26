"""
Build the analytical credit-risk fact table.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Input:
    data/staging/facts/stg_credit_risk.parquet

Output:
    data/analytical/facts/fact_credit_risk.parquet

Important:
    - Raw and staging data are READ ONLY.
    - No aggregation is performed.
    - No deduplication is performed.
    - No missing Amount values are filled.
    - No banking KPIs are calculated.
    - No descriptive dimensions are joined.
"""

from pathlib import Path
import sys

import polars as pl


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
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
    / "facts"
)

OUTPUT_FILE = OUTPUT_DIR / "fact_credit_risk.parquet"


# =============================================================================
# EXPECTED SOURCE / TARGET STRUCTURE
# =============================================================================

EXPECTED_SOURCE_COLUMNS = [
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

OUTPUT_COLUMNS = [
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

EXPECTED_OUTPUT_DTYPES = {
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
# VALIDATED EXPECTATIONS
# =============================================================================

EXPECTED_ROW_COUNT = 648_951
EXPECTED_COLUMN_COUNT = 14
EXPECTED_NULL_AMOUNT_COUNT = 1_637

EXPECTED_DISTINCT_LEI_COUNT = 120
EXPECTED_DISTINCT_ITEM_COUNT = 89
EXPECTED_DISTINCT_PERIOD_COUNT = 4

ALL_OTHER_BANKS_LEI = "XXXXXXXXXXXXXXXXXXXX"


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

LINE = "=" * 100
SUBLINE = "-" * 100


def print_header() -> None:
    """Print build header."""
    print(LINE)
    print("BUILDING ANALYTICAL CREDIT-RISK FACT")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    print(LINE)
    print(f"Source : {SOURCE_FILE}")
    print(f"Output : {OUTPUT_FILE}")
    print()
    print("BUILD MODE:")
    print("- preserve every validated staging observation")
    print("- no aggregation")
    print("- no deduplication")
    print("- preserve NULL Amount values")
    print("- preserve zero and negative Amount values")
    print("- retain source traceability")
    print("- no analytical dimension joins")
    print("- no KPI calculations")
    print()


def section(title: str) -> None:
    """Print a formatted section heading."""
    print()
    print(LINE)
    print(title)
    print(LINE)


def pass_message(message: str) -> None:
    """Print successful validation message."""
    print(f"[PASS] {message}")


def info_message(message: str) -> None:
    """Print informational message."""
    print(f"[INFO] {message}")


def fail(message: str) -> None:
    """
    Print a blocking failure and terminate.

    The build must not continue after a blocking validation failure.
    """
    print(f"[FAIL] {message}")
    print()
    print("BUILD STOPPED.")
    print("No raw or staging data was modified.")
    sys.exit(1)


def require(condition: bool, success: str, failure: str) -> None:
    """Apply a blocking validation condition."""
    if condition:
        pass_message(success)
    else:
        fail(failure)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def get_amount_profile(df: pl.DataFrame) -> dict:
    """
    Return the principal Amount reconciliation statistics.

    NULL and zero are deliberately measured separately.
    """
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


def print_amount_profile(profile: dict) -> None:
    """Print Amount population statistics."""
    print(f"NULL values     : {profile['null_count']:,}")
    print(f"Negative values : {profile['negative_count']:,}")
    print(f"Zero values     : {profile['zero_count']:,}")
    print(f"Positive values : {profile['positive_count']:,}")
    print(f"Minimum         : {profile['minimum']}")
    print(f"Maximum         : {profile['maximum']}")


def validate_business_grain(
    df: pl.DataFrame,
    dataset_name: str,
) -> None:
    """
    Confirm that the declared nine-column business grain is unique.

    No automatic deduplication is allowed.
    """
    unique_grain_count = df.select(BUSINESS_GRAIN).unique().height
    duplicate_rows = df.height - unique_grain_count

    require(
        unique_grain_count == df.height,
        (
            f"{dataset_name}: declared business grain is unique "
            f"across {df.height:,} rows."
        ),
        (
            f"{dataset_name}: declared business grain is NOT unique. "
            f"Rows={df.height:,}, unique grain combinations="
            f"{unique_grain_count:,}, difference={duplicate_rows:,}. "
            "Investigate before proceeding; do not deduplicate automatically."
        ),
    )


def validate_full_row_uniqueness(
    df: pl.DataFrame,
    dataset_name: str,
) -> None:
    """Confirm that no exact full-row duplicates exist."""
    unique_row_count = df.unique().height

    require(
        unique_row_count == df.height,
        f"{dataset_name}: no exact full-row duplicates found.",
        (
            f"{dataset_name}: exact full-row duplicates detected. "
            f"Rows={df.height:,}, unique rows={unique_row_count:,}."
        ),
    )


def validate_output_dtypes(
    df: pl.DataFrame,
    dataset_name: str,
) -> None:
    """Validate the approved analytical data types."""
    for column_name, expected_dtype in EXPECTED_OUTPUT_DTYPES.items():
        if column_name not in df.columns:
            fail(
                f"{dataset_name}: cannot validate dtype because "
                f"'{column_name}' is missing."
            )

        actual_dtype = df.schema[column_name]

        require(
            actual_dtype == expected_dtype,
            (
                f"{dataset_name}: {column_name} dtype is "
                f"{actual_dtype}."
            ),
            (
                f"{dataset_name}: unexpected dtype for '{column_name}'. "
                f"Expected {expected_dtype}, found {actual_dtype}."
            ),
        )


def compare_amount_profiles(
    source_profile: dict,
    target_profile: dict,
    target_name: str,
) -> None:
    """Ensure Amount characteristics reconcile exactly to staging."""
    fields = [
        "null_count",
        "negative_count",
        "zero_count",
        "positive_count",
        "minimum",
        "maximum",
    ]

    for field in fields:
        require(
            source_profile[field] == target_profile[field],
            (
                f"{target_name}: Amount {field} reconciles to staging "
                f"({target_profile[field]})."
            ),
            (
                f"{target_name}: Amount {field} does not reconcile. "
                f"Staging={source_profile[field]}, "
                f"target={target_profile[field]}."
            ),
        )


# =============================================================================
# MAIN BUILD
# =============================================================================

def main() -> None:
    """Build and validate fact_credit_risk."""

    print_header()

    # -------------------------------------------------------------------------
    # 1. SOURCE FILE CONTROL
    # -------------------------------------------------------------------------

    section("SOURCE FILE VALIDATION")

    require(
        SOURCE_FILE.exists(),
        f"Staging source exists: {SOURCE_FILE.name}",
        f"Staging source not found: {SOURCE_FILE}",
    )

    try:
        source = pl.read_parquet(SOURCE_FILE)
    except Exception as exc:
        fail(f"Could not read staging Parquet: {exc}")

    pass_message("Staging Parquet opened successfully.")

    # -------------------------------------------------------------------------
    # 2. SOURCE STRUCTURE
    # -------------------------------------------------------------------------

    section("SOURCE STRUCTURE VALIDATION")

    info_message(f"Source rows    : {source.height:,}")
    info_message(f"Source columns : {source.width}")

    require(
        source.columns == EXPECTED_SOURCE_COLUMNS,
        "All 16 expected staging columns are present in the expected order.",
        (
            "Unexpected staging schema.\n"
            f"Expected: {EXPECTED_SOURCE_COLUMNS}\n"
            f"Actual  : {source.columns}"
        ),
    )

    require(
        source.height == EXPECTED_ROW_COUNT,
        f"Source row count reconciles: {source.height:,} rows.",
        (
            f"Unexpected source row count. "
            f"Expected {EXPECTED_ROW_COUNT:,}, "
            f"found {source.height:,}."
        ),
    )

    require(
        source.schema["amount"] == pl.Float64,
        "Source amount dtype is Float64.",
        (
            "Unexpected source amount dtype. "
            f"Expected Float64, found {source.schema['amount']}."
        ),
    )

    # -------------------------------------------------------------------------
    # 3. SOURCE NULL / KEY CONTROLS
    # -------------------------------------------------------------------------

    section("SOURCE KEY AND MISSINGNESS VALIDATION")

    for key_column in BUSINESS_GRAIN:
        null_count = source[key_column].null_count()

        require(
            null_count == 0,
            f"{key_column}: 0 NULL value(s).",
            (
                f"{key_column}: {null_count:,} NULL value(s) detected "
                "in a declared business-grain field."
            ),
        )

    source_null_amounts = source["amount"].null_count()

    require(
        source_null_amounts == EXPECTED_NULL_AMOUNT_COUNT,
        (
            f"Source Amount missingness reconciles: "
            f"{source_null_amounts:,} NULL value(s)."
        ),
        (
            f"Unexpected source Amount NULL count. "
            f"Expected {EXPECTED_NULL_AMOUNT_COUNT:,}, "
            f"found {source_null_amounts:,}."
        ),
    )

    # -------------------------------------------------------------------------
    # 4. SOURCE GRAIN
    # -------------------------------------------------------------------------

    section("SOURCE BUSINESS-GRAIN VALIDATION")

    print("Declared grain:")
    for column_name in BUSINESS_GRAIN:
        print(f"  - {column_name}")

    validate_business_grain(
        source,
        "stg_credit_risk",
    )

    # -------------------------------------------------------------------------
    # 5. SOURCE POPULATION PROFILE
    # -------------------------------------------------------------------------

    section("SOURCE POPULATION PROFILE")

    source_distinct_lei = source["lei_code"].n_unique()
    source_distinct_item = source["item"].n_unique()
    source_distinct_period = source["period"].n_unique()

    require(
        source_distinct_lei == EXPECTED_DISTINCT_LEI_COUNT,
        (
            f"Distinct LEI population reconciles: "
            f"{source_distinct_lei:,} code(s)."
        ),
        (
            f"Unexpected distinct LEI count. "
            f"Expected {EXPECTED_DISTINCT_LEI_COUNT:,}, "
            f"found {source_distinct_lei:,}."
        ),
    )

    require(
        source_distinct_item == EXPECTED_DISTINCT_ITEM_COUNT,
        (
            f"Distinct Item population reconciles: "
            f"{source_distinct_item:,} code(s)."
        ),
        (
            f"Unexpected distinct Item count. "
            f"Expected {EXPECTED_DISTINCT_ITEM_COUNT:,}, "
            f"found {source_distinct_item:,}."
        ),
    )

    require(
        source_distinct_period == EXPECTED_DISTINCT_PERIOD_COUNT,
        (
            f"Distinct reporting-period population reconciles: "
            f"{source_distinct_period:,} period(s)."
        ),
        (
            f"Unexpected distinct reporting-period count. "
            f"Expected {EXPECTED_DISTINCT_PERIOD_COUNT:,}, "
            f"found {source_distinct_period:,}."
        ),
    )

    period_values = (
        source
        .select("period")
        .unique()
        .sort("period")
        .get_column("period")
        .to_list()
    )

    info_message(f"Reporting periods: {period_values}")

    source_aggregate_rows = source.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    ).height

    require(
        source_aggregate_rows > 0,
        (
            "Recognised EBA 'All other banks' aggregate is present "
            f"({source_aggregate_rows:,} row(s))."
        ),
        (
            "Recognised EBA 'All other banks' aggregate "
            f"'{ALL_OTHER_BANKS_LEI}' was not found."
        ),
    )

    # -------------------------------------------------------------------------
    # 6. SOURCE AMOUNT PROFILE
    # -------------------------------------------------------------------------

    section("SOURCE AMOUNT PROFILE")

    source_amount_profile = get_amount_profile(source)
    print_amount_profile(source_amount_profile)

    print()
    print(
        "Note: NULL, zero, positive and negative values are preserved "
        "as separate source states."
    )

    # -------------------------------------------------------------------------
    # 7. BUILD ANALYTICAL FACT
    # -------------------------------------------------------------------------

    section("MATERIALISING ANALYTICAL FACT")

    print("Columns deliberately excluded from the analytical fact:")
    print("  - nsa")
    print("  - label")
    print()

    fact = source.select(OUTPUT_COLUMNS)

    pass_message(
        "Analytical fact created in memory using controlled column projection."
    )

    # -------------------------------------------------------------------------
    # 8. PRE-SAVE VALIDATION
    # -------------------------------------------------------------------------

    section("PRE-SAVE ANALYTICAL FACT VALIDATION")

    require(
        fact.height == EXPECTED_ROW_COUNT,
        f"Row reconciliation: {fact.height:,} rows preserved.",
        (
            f"Analytical fact row count mismatch. "
            f"Expected {EXPECTED_ROW_COUNT:,}, "
            f"found {fact.height:,}."
        ),
    )

    require(
        fact.width == EXPECTED_COLUMN_COUNT,
        f"Column count: {fact.width} approved analytical columns.",
        (
            f"Unexpected analytical column count. "
            f"Expected {EXPECTED_COLUMN_COUNT}, "
            f"found {fact.width}."
        ),
    )

    require(
        fact.columns == OUTPUT_COLUMNS,
        "Analytical fact columns are present in the approved order.",
        (
            "Analytical output column order differs from specification.\n"
            f"Expected: {OUTPUT_COLUMNS}\n"
            f"Actual  : {fact.columns}"
        ),
    )

    validate_output_dtypes(
        fact,
        "fact_credit_risk",
    )

    validate_business_grain(
        fact,
        "fact_credit_risk",
    )

    validate_full_row_uniqueness(
        fact,
        "fact_credit_risk",
    )

    for trace_column in TRACEABILITY_COLUMNS:
        require(
            trace_column in fact.columns,
            f"Traceability field '{trace_column}' preserved.",
            f"Traceability field '{trace_column}' is missing.",
        )

    fact_amount_profile = get_amount_profile(fact)

    compare_amount_profiles(
        source_amount_profile,
        fact_amount_profile,
        "fact_credit_risk",
    )

    fact_aggregate_rows = fact.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    ).height

    require(
        fact_aggregate_rows == source_aggregate_rows,
        (
            "EBA 'All other banks' aggregate population reconciles: "
            f"{fact_aggregate_rows:,} row(s)."
        ),
        (
            "EBA aggregate population changed during transformation. "
            f"Staging={source_aggregate_rows:,}, "
            f"fact={fact_aggregate_rows:,}."
        ),
    )

    # Because this build is only a controlled projection, the selected
    # source columns should be identical to the in-memory analytical fact.
    source_projection = source.select(OUTPUT_COLUMNS)

    require(
        source_projection.equals(fact),
        (
            "Full selected-column reconciliation passed: analytical fact "
            "matches the staging projection exactly."
        ),
        (
            "Selected staging values differ from the in-memory analytical "
            "fact. Build must stop."
        ),
    )

    # -------------------------------------------------------------------------
    # 9. SAVE
    # -------------------------------------------------------------------------

    section("SAVING ANALYTICAL FACT")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        info_message(
            "Existing analytical fact file detected. "
            "It will be replaced by this reproducible build."
        )

    try:
        fact.write_parquet(
            OUTPUT_FILE,
            compression="zstd",
        )
    except Exception as exc:
        fail(f"Could not write analytical fact Parquet: {exc}")

    pass_message("Analytical fact written successfully.")
    print()
    print(f"Saved analytical fact:")
    print(f"  {OUTPUT_FILE}")

    # -------------------------------------------------------------------------
    # 10. POST-SAVE READ
    # -------------------------------------------------------------------------

    section("POST-SAVE VALIDATION")

    require(
        OUTPUT_FILE.exists(),
        f"Saved file exists: {OUTPUT_FILE.name}",
        f"Expected saved file was not found: {OUTPUT_FILE}",
    )

    try:
        saved = pl.read_parquet(OUTPUT_FILE)
    except Exception as exc:
        fail(f"Could not reopen saved analytical fact: {exc}")

    pass_message("Saved analytical fact reopened successfully.")

    # -------------------------------------------------------------------------
    # 11. POST-SAVE STRUCTURE
    # -------------------------------------------------------------------------

    require(
        saved.height == EXPECTED_ROW_COUNT,
        f"Saved row count: {saved.height:,}",
        (
            f"Saved row count mismatch. "
            f"Expected {EXPECTED_ROW_COUNT:,}, "
            f"found {saved.height:,}."
        ),
    )

    require(
        saved.width == EXPECTED_COLUMN_COUNT,
        f"Saved column count: {saved.width}",
        (
            f"Saved column count mismatch. "
            f"Expected {EXPECTED_COLUMN_COUNT}, "
            f"found {saved.width}."
        ),
    )

    require(
        saved.columns == OUTPUT_COLUMNS,
        "Saved analytical columns remain in the approved order.",
        (
            "Saved analytical schema order differs from specification.\n"
            f"Expected: {OUTPUT_COLUMNS}\n"
            f"Actual  : {saved.columns}"
        ),
    )

    validate_output_dtypes(
        saved,
        "saved fact_credit_risk",
    )

    # -------------------------------------------------------------------------
    # 12. POST-SAVE KEY / GRAIN / TRACEABILITY CONTROLS
    # -------------------------------------------------------------------------

    for key_column in BUSINESS_GRAIN:
        null_count = saved[key_column].null_count()

        require(
            null_count == 0,
            f"Saved {key_column}: 0 NULL value(s).",
            (
                f"Saved {key_column}: {null_count:,} NULL value(s) detected."
            ),
        )

    validate_business_grain(
        saved,
        "saved fact_credit_risk",
    )

    validate_full_row_uniqueness(
        saved,
        "saved fact_credit_risk",
    )

    for trace_column in TRACEABILITY_COLUMNS:
        require(
            trace_column in saved.columns,
            f"Saved traceability field '{trace_column}' exists.",
            f"Saved traceability field '{trace_column}' is missing.",
        )

    # -------------------------------------------------------------------------
    # 13. POST-SAVE AMOUNT RECONCILIATION
    # -------------------------------------------------------------------------

    saved_amount_profile = get_amount_profile(saved)

    require(
        saved_amount_profile["null_count"] == EXPECTED_NULL_AMOUNT_COUNT,
        (
            f"Saved Amount missingness preserved: "
            f"{saved_amount_profile['null_count']:,} NULL value(s)."
        ),
        (
            f"Saved Amount NULL count mismatch. "
            f"Expected {EXPECTED_NULL_AMOUNT_COUNT:,}, "
            f"found {saved_amount_profile['null_count']:,}."
        ),
    )

    compare_amount_profiles(
        source_amount_profile,
        saved_amount_profile,
        "saved fact_credit_risk",
    )

    # -------------------------------------------------------------------------
    # 14. COMPLETE VALUE RECONCILIATION
    # -------------------------------------------------------------------------

    require(
        saved.equals(fact),
        (
            "Saved Parquet reconciles exactly to the validated "
            "in-memory analytical fact."
        ),
        (
            "Saved Parquet differs from the validated in-memory "
            "analytical fact."
        ),
    )

    # -------------------------------------------------------------------------
    # 15. POPULATION RECONCILIATION
    # -------------------------------------------------------------------------

    saved_distinct_lei = saved["lei_code"].n_unique()
    saved_distinct_item = saved["item"].n_unique()
    saved_distinct_period = saved["period"].n_unique()

    require(
        saved_distinct_lei == source_distinct_lei,
        f"Distinct LEI codes: {saved_distinct_lei:,}",
        (
            "Distinct LEI population changed after save. "
            f"Source={source_distinct_lei:,}, "
            f"saved={saved_distinct_lei:,}."
        ),
    )

    require(
        saved_distinct_item == source_distinct_item,
        f"Distinct Item codes: {saved_distinct_item:,}",
        (
            "Distinct Item population changed after save. "
            f"Source={source_distinct_item:,}, "
            f"saved={saved_distinct_item:,}."
        ),
    )

    require(
        saved_distinct_period == source_distinct_period,
        f"Distinct periods: {saved_distinct_period:,}",
        (
            "Distinct period population changed after save. "
            f"Source={source_distinct_period:,}, "
            f"saved={saved_distinct_period:,}."
        ),
    )

    saved_aggregate_rows = saved.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    ).height

    require(
        saved_aggregate_rows == source_aggregate_rows,
        (
            "Saved EBA 'All other banks' population preserved: "
            f"{saved_aggregate_rows:,} row(s)."
        ),
        (
            "Saved EBA aggregate population does not reconcile. "
            f"Source={source_aggregate_rows:,}, "
            f"saved={saved_aggregate_rows:,}."
        ),
    )

    # -------------------------------------------------------------------------
    # 16. FINAL PROFILE
    # -------------------------------------------------------------------------

    section("FINAL ANALYTICAL FACT PROFILE")

    print(f"Rows                : {saved.height:,}")
    print(f"Columns             : {saved.width}")
    print(f"Distinct LEI codes  : {saved_distinct_lei:,}")
    print(f"Distinct Item codes : {saved_distinct_item:,}")
    print(f"Distinct periods    : {saved_distinct_period:,}")
    print(f"NULL Amount values  : {saved_amount_profile['null_count']:,}")
    print(
        f"All other banks rows: {saved_aggregate_rows:,}"
    )

    print()
    print("Amount profile:")
    print(SUBLINE)
    print_amount_profile(saved_amount_profile)

    # -------------------------------------------------------------------------
    # 17. COMPLETION
    # -------------------------------------------------------------------------

    section("CREDIT-RISK ANALYTICAL FACT BUILD COMPLETED SUCCESSFULLY")

    print(f"Output: {OUTPUT_FILE}")
    print()
    print("No raw or staging data was modified.")
    print(
        "The analytical fact has not yet received independent validation."
    )


if __name__ == "__main__":
    main()
