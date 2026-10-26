"""
Build the analytical Other-Templates fact table.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Source:
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/facts/fact_other_templates.parquet

Design principles:
    - preserve every validated staging observation
    - no aggregation
    - no deduplication
    - preserve NULL Amount values
    - preserve zero and negative Amount values
    - preserve source traceability
    - do not join analytical dimensions
    - do not calculate KPIs
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
    / "stg_other_templates.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "facts"
)

OUTPUT_FILE = OUTPUT_DIR / "fact_other_templates.parquet"


# =============================================================================
# APPROVED SOURCE STRUCTURE
# =============================================================================

EXPECTED_SOURCE_COLUMNS = [
    "lei_code",
    "nsa",
    "period",
    "item",
    "label",
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


# nsa and label are descriptive/source-support fields already represented
# through analytical dimensions or metadata. They are therefore excluded
# from the analytical fact itself.

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


# =============================================================================
# VALIDATED SOURCE BASELINES
# =============================================================================

EXPECTED_ROW_COUNT = 91_277
EXPECTED_NULL_AMOUNT_COUNT = 3_964
EXPECTED_DISTINCT_LEI_COUNT = 120
EXPECTED_DISTINCT_ITEM_COUNT = 158
EXPECTED_DISTINCT_PERIOD_COUNT = 4
EXPECTED_DISTINCT_SHEET_COUNT = 6

EXPECTED_PERIODS = [
    202409,
    202412,
    202503,
    202506,
]

ALL_OTHER_BANKS_LEI = "XXXXXXXXXXXXXXXXXXXX"
EXPECTED_ALL_OTHER_BANKS_ROWS = 824


# =============================================================================
# BUSINESS GRAIN
# =============================================================================

# Previous grain-minimality investigation established that the following
# dimensions are sufficient to uniquely identify the reported analytical
# observation.
#
# fin_end_year and n_quarters are retained in the fact because they carry
# analytical/source context, but they are not required for uniqueness.

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
# OUTPUT HELPERS
# =============================================================================

LINE = "=" * 100
SUBLINE = "-" * 100


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def check(condition: bool, success: str, failure: str) -> None:
    if condition:
        print(f"[PASS] {success}")
    else:
        fail(failure)


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
# MAIN
# =============================================================================

def main() -> None:
    print(LINE)
    print("BUILDING ANALYTICAL OTHER-TEMPLATES FACT")
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
    print("- retain analytical context fields")
    print("- retain source traceability")
    print("- no analytical dimension joins")
    print("- no KPI calculations")

    # =========================================================================
    # SOURCE FILE
    # =========================================================================

    section("SOURCE FILE VALIDATION")

    check(
        SOURCE_FILE.exists(),
        "Staging source exists: stg_other_templates.parquet",
        f"Staging source not found: {SOURCE_FILE}",
    )

    try:
        staging = pl.read_parquet(SOURCE_FILE)
    except Exception as exc:
        fail(f"Unable to read staging source: {exc}")

    print("[PASS] Staging Parquet opened successfully.")

    # =========================================================================
    # SOURCE STRUCTURE
    # =========================================================================

    section("SOURCE STRUCTURE VALIDATION")

    print(f"[INFO] Source rows    : {staging.height:,}")
    print(f"[INFO] Source columns : {staging.width}")

    check(
        staging.columns == EXPECTED_SOURCE_COLUMNS,
        "All 15 expected staging columns are present in the expected order.",
        (
            "Staging schema differs from validated structure.\n"
            f"Expected: {EXPECTED_SOURCE_COLUMNS}\n"
            f"Actual  : {staging.columns}"
        ),
    )

    check(
        staging.height == EXPECTED_ROW_COUNT,
        f"Source row count reconciles: {staging.height:,} rows.",
        (
            f"Expected {EXPECTED_ROW_COUNT:,} rows, "
            f"found {staging.height:,}."
        ),
    )

    check(
        staging.schema["amount"] == pl.Float64,
        "Source amount dtype is Float64.",
        f"Unexpected amount dtype: {staging.schema['amount']}",
    )

    # =========================================================================
    # SOURCE KEY / MISSINGNESS
    # =========================================================================

    section("SOURCE KEY AND MISSINGNESS VALIDATION")

    required_non_null = [
        "lei_code",
        "period",
        "item",
        "assets_fv",
        "assets_stages",
        "exposure",
        "financial_instruments",
        "fin_end_year",
        "n_quarters",
    ]

    for column_name in required_non_null:
        null_count = staging[column_name].null_count()

        check(
            null_count == 0,
            f"{column_name}: 0 NULL value(s).",
            f"{column_name}: {null_count:,} NULL value(s).",
        )

    source_null_amounts = staging["amount"].null_count()

    check(
        source_null_amounts == EXPECTED_NULL_AMOUNT_COUNT,
        (
            "Source Amount missingness reconciles: "
            f"{source_null_amounts:,} NULL value(s)."
        ),
        (
            f"Expected {EXPECTED_NULL_AMOUNT_COUNT:,} NULL Amounts, "
            f"found {source_null_amounts:,}."
        ),
    )

    # =========================================================================
    # BUSINESS GRAIN
    # =========================================================================

    section("SOURCE BUSINESS-GRAIN VALIDATION")

    print("Declared minimal analytical grain:")

    for column_name in BUSINESS_GRAIN:
        print(f"  - {column_name}")

    unique_grain_rows = (
        staging
        .select(BUSINESS_GRAIN)
        .unique()
        .height
    )

    check(
        unique_grain_rows == staging.height,
        (
            "stg_other_templates: declared business grain is unique "
            f"across {staging.height:,} rows."
        ),
        (
            "Declared business grain is not unique. "
            f"Unique grain rows={unique_grain_rows:,}; "
            f"source rows={staging.height:,}."
        ),
    )

    # =========================================================================
    # SOURCE POPULATION PROFILE
    # =========================================================================

    section("SOURCE POPULATION PROFILE")

    distinct_lei = staging["lei_code"].n_unique()
    distinct_item = staging["item"].n_unique()
    distinct_period = staging["period"].n_unique()
    distinct_sheet = staging["sheet"].n_unique()

    check(
        distinct_lei == EXPECTED_DISTINCT_LEI_COUNT,
        f"Distinct LEI population reconciles: {distinct_lei} code(s).",
        (
            f"Expected {EXPECTED_DISTINCT_LEI_COUNT} LEI codes; "
            f"found {distinct_lei}."
        ),
    )

    check(
        distinct_item == EXPECTED_DISTINCT_ITEM_COUNT,
        f"Distinct Item population reconciles: {distinct_item} code(s).",
        (
            f"Expected {EXPECTED_DISTINCT_ITEM_COUNT} Item codes; "
            f"found {distinct_item}."
        ),
    )

    check(
        distinct_period == EXPECTED_DISTINCT_PERIOD_COUNT,
        (
            "Distinct reporting-period population reconciles: "
            f"{distinct_period} period(s)."
        ),
        (
            f"Expected {EXPECTED_DISTINCT_PERIOD_COUNT} periods; "
            f"found {distinct_period}."
        ),
    )

    actual_periods = (
        staging
        .select("period")
        .unique()
        .sort("period")
        .get_column("period")
        .to_list()
    )

    check(
        actual_periods == EXPECTED_PERIODS,
        f"Reporting periods reconcile: {actual_periods}",
        (
            f"Expected periods {EXPECTED_PERIODS}; "
            f"found {actual_periods}."
        ),
    )

    check(
        distinct_sheet == EXPECTED_DISTINCT_SHEET_COUNT,
        f"Distinct source templates reconcile: {distinct_sheet}.",
        (
            f"Expected {EXPECTED_DISTINCT_SHEET_COUNT} templates; "
            f"found {distinct_sheet}."
        ),
    )

    aggregate_rows = staging.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    ).height

    check(
        aggregate_rows == EXPECTED_ALL_OTHER_BANKS_ROWS,
        (
            "Recognised EBA 'All other banks' aggregate is present "
            f"({aggregate_rows:,} row(s))."
        ),
        (
            f"Expected {EXPECTED_ALL_OTHER_BANKS_ROWS:,} aggregate rows; "
            f"found {aggregate_rows:,}."
        ),
    )

    # =========================================================================
    # AMOUNT PROFILE
    # =========================================================================

    section("SOURCE AMOUNT PROFILE")

    source_amount = amount_profile(staging)

    print(f"NULL values     : {source_amount['null_count']:,}")
    print(f"Negative values : {source_amount['negative_count']:,}")
    print(f"Zero values     : {source_amount['zero_count']:,}")
    print(f"Positive values : {source_amount['positive_count']:,}")
    print(f"Minimum         : {source_amount['minimum']}")
    print(f"Maximum         : {source_amount['maximum']}")

    print()
    print(
        "Note: NULL, zero, positive and negative values are preserved "
        "as separate source states."
    )

    # =========================================================================
    # MATERIALISE FACT
    # =========================================================================

    section("MATERIALISING ANALYTICAL FACT")

    print("Columns deliberately excluded from the analytical fact:")
    print("  - nsa")
    print("  - label")

    fact = staging.select(FACT_COLUMNS)

    check(
        fact.height == staging.height,
        (
            "Analytical fact created in memory with all "
            f"{fact.height:,} staging rows preserved."
        ),
        "Row loss occurred during analytical projection.",
    )

    # =========================================================================
    # PRE-SAVE VALIDATION
    # =========================================================================

    section("PRE-SAVE ANALYTICAL FACT VALIDATION")

    check(
        fact.height == EXPECTED_ROW_COUNT,
        f"Row reconciliation: {fact.height:,} rows preserved.",
        "Unexpected analytical fact row count.",
    )

    check(
        fact.width == len(FACT_COLUMNS),
        f"Column count: {fact.width} approved analytical columns.",
        "Unexpected analytical fact column count.",
    )

    check(
        fact.columns == FACT_COLUMNS,
        "Analytical fact columns are present in the approved order.",
        "Analytical fact column order differs from approved design.",
    )

    for column_name, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = fact.schema[column_name]

        check(
            actual_dtype == expected_dtype,
            f"{column_name} dtype is {actual_dtype}.",
            (
                f"{column_name}: expected {expected_dtype}, "
                f"found {actual_dtype}."
            ),
        )

    fact_unique_grain = (
        fact
        .select(BUSINESS_GRAIN)
        .unique()
        .height
    )

    check(
        fact_unique_grain == fact.height,
        (
            "fact_other_templates: declared business grain is unique "
            f"across {fact.height:,} rows."
        ),
        "Business-grain uniqueness was lost during projection.",
    )

    exact_unique_rows = fact.unique().height

    check(
        exact_unique_rows == fact.height,
        "fact_other_templates: no exact full-row duplicates found.",
        (
            f"{fact.height - exact_unique_rows:,} duplicate "
            "full row(s) detected."
        ),
    )

    for column_name in TRACEABILITY_COLUMNS:
        check(
            column_name in fact.columns,
            f"Traceability field '{column_name}' preserved.",
            f"Traceability field '{column_name}' missing.",
        )

    fact_amount = amount_profile(fact)

    for metric in [
        "null_count",
        "negative_count",
        "zero_count",
        "positive_count",
        "minimum",
        "maximum",
    ]:
        check(
            fact_amount[metric] == source_amount[metric],
            f"Amount {metric} reconciles to staging ({fact_amount[metric]}).",
            (
                f"Amount {metric} mismatch: "
                f"source={source_amount[metric]}, "
                f"fact={fact_amount[metric]}"
            ),
        )

    check(
        fact.equals(staging.select(FACT_COLUMNS)),
        "Full selected-column reconciliation passed.",
        "Analytical fact differs from the approved staging projection.",
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    section("SAVING ANALYTICAL FACT")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fact.write_parquet(OUTPUT_FILE)

    check(
        OUTPUT_FILE.exists(),
        "Analytical fact written successfully.",
        "Analytical fact was not written.",
    )

    print()
    print("Saved analytical fact:")
    print(f"  {OUTPUT_FILE}")

    # =========================================================================
    # POST-SAVE VALIDATION
    # =========================================================================

    section("POST-SAVE VALIDATION")

    saved = pl.read_parquet(OUTPUT_FILE)

    check(
        saved.height == fact.height,
        f"Saved row count: {saved.height:,}",
        "Saved row count differs from validated in-memory fact.",
    )

    check(
        saved.columns == FACT_COLUMNS,
        "Saved analytical columns remain in the approved order.",
        "Saved analytical fact column order changed.",
    )

    check(
        saved.equals(fact),
        "Saved Parquet reconciles exactly to validated in-memory fact.",
        "Saved Parquet differs from validated in-memory fact.",
    )

    saved_amount = amount_profile(saved)

    check(
        saved_amount["null_count"] == EXPECTED_NULL_AMOUNT_COUNT,
        (
            "Saved Amount missingness preserved: "
            f"{saved_amount['null_count']:,} NULL value(s)."
        ),
        "Saved Amount missingness differs from approved baseline.",
    )

    saved_aggregate_rows = saved.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    ).height

    check(
        saved_aggregate_rows == EXPECTED_ALL_OTHER_BANKS_ROWS,
        (
            "Saved EBA 'All other banks' population preserved: "
            f"{saved_aggregate_rows:,} row(s)."
        ),
        "Saved aggregate population differs from source.",
    )

    # =========================================================================
    # FINAL PROFILE
    # =========================================================================

    section("FINAL ANALYTICAL FACT PROFILE")

    print(f"Rows               : {saved.height:,}")
    print(f"Columns            : {saved.width}")
    print(f"Distinct LEI codes : {saved['lei_code'].n_unique()}")
    print(f"Distinct Item codes: {saved['item'].n_unique()}")
    print(f"Distinct periods   : {saved['period'].n_unique()}")
    print(f"Distinct sheets    : {saved['sheet'].n_unique()}")
    print(f"NULL Amount values : {saved['amount'].null_count():,}")
    print(f"All other banks rows: {saved_aggregate_rows:,}")

    print()
    print("Amount profile:")
    print(SUBLINE)
    print(f"NULL values     : {saved_amount['null_count']:,}")
    print(f"Negative values : {saved_amount['negative_count']:,}")
    print(f"Zero values     : {saved_amount['zero_count']:,}")
    print(f"Positive values : {saved_amount['positive_count']:,}")
    print(f"Minimum         : {saved_amount['minimum']}")
    print(f"Maximum         : {saved_amount['maximum']}")

    section("OTHER-TEMPLATES ANALYTICAL FACT BUILD COMPLETED SUCCESSFULLY")

    print(f"Output: {OUTPUT_FILE}")
    print()
    print("No raw or staging data was modified.")
    print(
        "The analytical fact has not yet received "
        "independent validation."
    )


if __name__ == "__main__":
    main()
