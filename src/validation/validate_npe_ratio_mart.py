from pathlib import Path
import sys

import polars as pl


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MART_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_ratio.parquet"
)

FACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "facts"
    / "fact_credit_risk.parquet"
)

INSTITUTION_DIM_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
    / "dim_institution.parquet"
)

PERIOD_DIM_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
    / "dim_period.parquet"
)


# Approved KPI definition
TARGET_ITEM = 2520603

DENOMINATOR_PERF_STATUS = 0
NUMERATOR_PERF_STATUS = 2

BUSINESS_TOLERANCE = 0.001
FLOAT_EPSILON = 1e-9
COMPARISON_LIMIT = BUSINESS_TOLERANCE + FLOAT_EPSILON


EXPECTED_MART_COLUMNS = [
    "lei_code",
    "institution_name",
    "institution_country_code",
    "institution_country",
    "entity_type",
    "is_aggregate",
    "period",
    "period_end_date",
    "year",
    "quarter",
    "month",
    "year_quarter",
    "is_year_end",
    "total_gross_carrying_amount",
    "non_performing_amount",
    "npe_ratio",
    "npe_ratio_pct",
    "zero_denominator_flag",
]


# ============================================================
# OUTPUT HELPERS
# ============================================================

def separator() -> None:
    print("=" * 92)


def section(title: str) -> None:
    print()
    separator()
    print(title)
    separator()


def pass_message(message: str) -> None:
    print(f"[PASS] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def info(message: str) -> None:
    print(f"[INFO] {message}")


# ============================================================
# FILE CHECKS
# ============================================================

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} not found: {path}")

    pass_message(f"Found: {path.name}")


# ============================================================
# MAIN VALIDATION
# ============================================================

def main() -> None:

    separator()
    print("NPE RATIO MART INDEPENDENT VALIDATION")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    separator()

    print()
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"NPE Ratio mart    : {MART_PATH}")
    print(f"Credit-risk fact  : {FACT_PATH}")
    print(f"Institution dim   : {INSTITUTION_DIM_PATH}")
    print(f"Period dim        : {PERIOD_DIM_PATH}")

    print()
    print("VALIDATION MODE: READ ONLY")
    print("No analytical, staging, reference, or raw data will be modified.")

    # ========================================================
    # 1. FILE AVAILABILITY
    # ========================================================

    section("1. FILE AVAILABILITY")

    require_file(MART_PATH, "NPE Ratio mart")
    require_file(FACT_PATH, "Credit-risk analytical fact")
    require_file(INSTITUTION_DIM_PATH, "Institution dimension")
    require_file(PERIOD_DIM_PATH, "Period dimension")

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    section("2. LOAD APPROVED ANALYTICAL DATA")

    try:
        mart = pl.read_parquet(MART_PATH)
        fact = pl.read_parquet(FACT_PATH)
        institution_dim = pl.read_parquet(INSTITUTION_DIM_PATH)
        period_dim = pl.read_parquet(PERIOD_DIM_PATH)
    except Exception as exc:
        fail(f"Unable to open required Parquet file: {exc}")

    pass_message(f"mart_npe_ratio opened: {mart.height:,} rows")
    pass_message(f"fact_credit_risk opened: {fact.height:,} rows")
    pass_message(f"dim_institution opened: {institution_dim.height:,} rows")
    pass_message(f"dim_period opened: {period_dim.height:,} rows")

    # ========================================================
    # 3. STRUCTURAL VALIDATION
    # ========================================================

    section("3. MART STRUCTURAL VALIDATION")

    if mart.height == 0:
        fail("NPE Ratio mart is empty.")

    pass_message(f"Mart contains {mart.height:,} row(s).")

    if mart.columns != EXPECTED_MART_COLUMNS:
        print("Expected columns:")
        print(EXPECTED_MART_COLUMNS)

        print("Actual columns:")
        print(mart.columns)

        fail("Mart columns do not match the approved design.")

    pass_message("Mart columns match the approved design and order.")

    expected_dtypes = {
        "lei_code": pl.String,
        "institution_name": pl.String,
        "period": pl.Int64,
        "total_gross_carrying_amount": pl.Float64,
        "non_performing_amount": pl.Float64,
        "npe_ratio": pl.Float64,
        "npe_ratio_pct": pl.Float64,
        "is_aggregate": pl.Boolean,
    }

    schema = mart.schema

    for column_name, expected_dtype in expected_dtypes.items():
        actual_dtype = schema[column_name]

        if actual_dtype != expected_dtype:
            fail(
                f"Column '{column_name}' has dtype {actual_dtype}; "
                f"expected {expected_dtype}."
            )

        pass_message(
            f"{column_name} dtype is {expected_dtype}."
        )

    # ========================================================
    # 4. BUSINESS-GRAIN VALIDATION
    # ========================================================

    section("4. BUSINESS-GRAIN VALIDATION")

    grain_columns = ["lei_code", "period"]

    duplicate_grain = (
        mart
        .group_by(grain_columns)
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicate_grain.height > 0:
        print(duplicate_grain.head(20))
        fail(
            "NPE Ratio mart contains duplicate institution-period "
            "observations."
        )

    pass_message(
        f"Institution-period grain is unique across {mart.height:,} rows."
    )

    null_grain = mart.select(
        [
            pl.col("lei_code").is_null().sum().alias("null_lei_code"),
            pl.col("period").is_null().sum().alias("null_period"),
        ]
    )

    if (
        null_grain["null_lei_code"][0] > 0
        or null_grain["null_period"][0] > 0
    ):
        print(null_grain)
        fail("Business grain contains NULL values.")

    pass_message("Business grain contains no NULL values.")

    # ========================================================
    # 5. SOURCE DEFINITION RECONSTRUCTION
    # ========================================================

    section("5. INDEPENDENT SOURCE RECONSTRUCTION")

    required_fact_columns = [
        "lei_code",
        "period",
        "item",
        "portfolio",
        "country",
        "exposure",
        "status",
        "perf_status",
        "amount",
    ]

    missing_fact_columns = [
        column_name
        for column_name in required_fact_columns
        if column_name not in fact.columns
    ]

    if missing_fact_columns:
        fail(
            "Required fact columns missing: "
            + ", ".join(missing_fact_columns)
        )

    denominator = fact.filter(
        (pl.col("item") == TARGET_ITEM)
        & (pl.col("portfolio") == 0)
        & (pl.col("country") == 0)
        & (pl.col("exposure") == 0)
        & (pl.col("status") == 0)
        & (pl.col("perf_status") == DENOMINATOR_PERF_STATUS)
    ).select(
        [
            "lei_code",
            "period",
            pl.col("amount").alias(
                "expected_total_gross_carrying_amount"
            ),
        ]
    )

    numerator = fact.filter(
        (pl.col("item") == TARGET_ITEM)
        & (pl.col("portfolio") == 0)
        & (pl.col("country") == 0)
        & (pl.col("exposure") == 0)
        & (pl.col("status") == 0)
        & (pl.col("perf_status") == NUMERATOR_PERF_STATUS)
    ).select(
        [
            "lei_code",
            "period",
            pl.col("amount").alias(
                "expected_non_performing_amount"
            ),
        ]
    )

    info(f"Reconstructed denominator rows: {denominator.height:,}")
    info(f"Reconstructed numerator rows  : {numerator.height:,}")

    if denominator.height == 0:
        fail("Independent denominator reconstruction returned zero rows.")

    if numerator.height == 0:
        fail("Independent numerator reconstruction returned zero rows.")

    denominator_duplicates = (
        denominator
        .group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if denominator_duplicates.height > 0:
        fail(
            "Reconstructed denominator is not unique at "
            "institution-period grain."
        )

    pass_message(
        "Reconstructed denominator contains one observation per "
        "institution-period."
    )

    numerator_duplicates = (
        numerator
        .group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if numerator_duplicates.height > 0:
        fail(
            "Reconstructed numerator is not unique at "
            "institution-period grain."
        )

    pass_message(
        "Reconstructed numerator contains one observation per "
        "institution-period."
    )

    # ========================================================
    # 6. SOURCE COVERAGE RECONCILIATION
    # ========================================================

    section("6. SOURCE COVERAGE RECONCILIATION")

    reconstructed = denominator.join(
        numerator,
        on=["lei_code", "period"],
        how="full",
        coalesce=True,
    )

    denominator_keys = denominator.select(
        ["lei_code", "period"]
    ).unique()

    numerator_keys = numerator.select(
        ["lei_code", "period"]
    ).unique()

    missing_numerator = denominator_keys.join(
        numerator_keys,
        on=["lei_code", "period"],
        how="anti",
    )

    missing_denominator = numerator_keys.join(
        denominator_keys,
        on=["lei_code", "period"],
        how="anti",
    )

    info(
        f"Denominator institution-period rows: "
        f"{denominator_keys.height:,}"
    )
    info(
        f"Numerator institution-period rows  : "
        f"{numerator_keys.height:,}"
    )
    info(
        f"Missing numerator observations     : "
        f"{missing_numerator.height:,}"
    )
    info(
        f"Missing denominator observations   : "
        f"{missing_denominator.height:,}"
    )

    if (
        missing_numerator.height > 0
        or missing_denominator.height > 0
    ):
        fail(
            "Numerator and denominator institution-period coverage "
            "does not reconcile."
        )

    pass_message(
        "Numerator and denominator institution-period coverage is identical."
    )

    # ========================================================
    # 7. MART-TO-SOURCE RECONCILIATION
    # ========================================================

    section("7. MART-TO-SOURCE RECONCILIATION")

    source_comparison = mart.join(
        reconstructed,
        on=["lei_code", "period"],
        how="left",
    ).with_columns(
        [
            (
                pl.col("total_gross_carrying_amount")
                - pl.col("expected_total_gross_carrying_amount")
            ).abs().alias("denominator_difference"),

            (
                pl.col("non_performing_amount")
                - pl.col("expected_non_performing_amount")
            ).abs().alias("numerator_difference"),
        ]
    )

    missing_reconstructed_values = source_comparison.filter(
        pl.col(
            "expected_total_gross_carrying_amount"
        ).is_null()
        | pl.col(
            "expected_non_performing_amount"
        ).is_null()
    )

    if missing_reconstructed_values.height > 0:
        fail(
            "Some saved mart observations cannot be matched to the "
            "independently reconstructed source definition."
        )

    pass_message(
        "Every saved mart observation matches an independently "
        "reconstructed source observation."
    )

    denominator_mismatches = source_comparison.filter(
        pl.col("denominator_difference") > COMPARISON_LIMIT
    )

    numerator_mismatches = source_comparison.filter(
        pl.col("numerator_difference") > COMPARISON_LIMIT
    )

    if denominator_mismatches.height > 0:
        print(
            denominator_mismatches.select(
                [
                    "lei_code",
                    "period",
                    "total_gross_carrying_amount",
                    "expected_total_gross_carrying_amount",
                    "denominator_difference",
                ]
            ).head(20)
        )

        fail(
            "Saved denominator values do not reconcile to the "
            "approved fact definition."
        )

    pass_message(
        "Saved denominator amounts reconcile independently to "
        "fact_credit_risk."
    )

    if numerator_mismatches.height > 0:
        print(
            numerator_mismatches.select(
                [
                    "lei_code",
                    "period",
                    "non_performing_amount",
                    "expected_non_performing_amount",
                    "numerator_difference",
                ]
            ).head(20)
        )

        fail(
            "Saved numerator values do not reconcile to the "
            "approved fact definition."
        )

    pass_message(
        "Saved NPE numerator amounts reconcile independently to "
        "fact_credit_risk."
    )

    # ========================================================
    # 8. KPI FORMULA VALIDATION
    # ========================================================

    section("8. KPI FORMULA VALIDATION")

    recalculated = mart.with_columns(
        [
            pl.when(
                pl.col("total_gross_carrying_amount") == 0
            )
            .then(None)
            .otherwise(
                pl.col("non_performing_amount")
                / pl.col("total_gross_carrying_amount")
            )
            .alias("expected_npe_ratio")
        ]
    ).with_columns(
        [
            (
                pl.col("expected_npe_ratio") * 100
            ).alias("expected_npe_ratio_pct")
        ]
    ).with_columns(
        [
            (
                pl.col("npe_ratio")
                - pl.col("expected_npe_ratio")
            ).abs().alias("npe_ratio_difference"),

            (
                pl.col("npe_ratio_pct")
                - pl.col("expected_npe_ratio_pct")
            ).abs().alias("npe_ratio_pct_difference"),
        ]
    )

    ratio_mismatches = recalculated.filter(
        pl.col("npe_ratio_difference") > FLOAT_EPSILON
    )

    pct_mismatches = recalculated.filter(
        pl.col("npe_ratio_pct_difference") > FLOAT_EPSILON
    )

    if ratio_mismatches.height > 0:
        print(
            ratio_mismatches.select(
                [
                    "lei_code",
                    "period",
                    "total_gross_carrying_amount",
                    "non_performing_amount",
                    "npe_ratio",
                    "expected_npe_ratio",
                    "npe_ratio_difference",
                ]
            ).head(20)
        )

        fail("Saved NPE Ratio does not reconcile to the approved formula.")

    pass_message(
        "Saved NPE Ratio reconciles to numerator / denominator."
    )

    if pct_mismatches.height > 0:
        print(
            pct_mismatches.select(
                [
                    "lei_code",
                    "period",
                    "npe_ratio_pct",
                    "expected_npe_ratio_pct",
                    "npe_ratio_pct_difference",
                ]
            ).head(20)
        )

        fail(
            "Saved NPE Ratio percentage does not reconcile to "
            "NPE Ratio × 100."
        )

    pass_message(
        "Saved NPE Ratio percentage reconciles to NPE Ratio × 100."
    )

    # ========================================================
    # 9. KPI BUSINESS-RULE VALIDATION
    # ========================================================

    section("9. KPI BUSINESS-RULE VALIDATION")

    null_denominators = mart.filter(
        pl.col("total_gross_carrying_amount").is_null()
    ).height

    zero_denominators = mart.filter(
        pl.col("total_gross_carrying_amount") == 0
    ).height

    negative_denominators = mart.filter(
        pl.col("total_gross_carrying_amount") < 0
    ).height

    null_numerators = mart.filter(
        pl.col("non_performing_amount").is_null()
    ).height

    negative_numerators = mart.filter(
        pl.col("non_performing_amount") < 0
    ).height

    numerator_exceeds_denominator = mart.filter(
        pl.col("non_performing_amount")
        > pl.col("total_gross_carrying_amount")
        + COMPARISON_LIMIT
    ).height

    invalid_ratio_range = mart.filter(
        pl.col("npe_ratio").is_not_null()
        & (
            (pl.col("npe_ratio") < 0)
            | (pl.col("npe_ratio") > 1)
        )
    ).height

    info(f"NULL denominators                 : {null_denominators:,}")
    info(f"Zero denominators                 : {zero_denominators:,}")
    info(f"Negative denominators             : {negative_denominators:,}")
    info(f"NULL numerators                   : {null_numerators:,}")
    info(f"Negative numerators               : {negative_numerators:,}")
    info(
        f"Numerators exceeding denominator  : "
        f"{numerator_exceeds_denominator:,}"
    )
    info(f"Ratios outside [0,1]              : {invalid_ratio_range:,}")

    if null_denominators > 0:
        fail("NULL KPI denominators found.")

    pass_message("No NULL denominators found.")

    if zero_denominators > 0:
        fail(
            "Zero denominator observations exist and require "
            "explicit business treatment."
        )

    pass_message("No zero denominators found.")

    if negative_denominators > 0:
        fail("Negative denominator observations found.")

    pass_message("No negative denominators found.")

    if null_numerators > 0:
        fail("NULL NPE numerator observations found.")

    pass_message("No NULL NPE numerators found.")

    if negative_numerators > 0:
        fail("Negative NPE numerator observations found.")

    pass_message("No negative NPE numerators found.")

    if numerator_exceeds_denominator > 0:
        fail(
            "NPE numerator materially exceeds the corresponding denominator."
        )

    pass_message(
        "No NPE numerator materially exceeds its denominator."
    )

    if invalid_ratio_range > 0:
        fail("NPE Ratio observations outside the expected [0, 1] range.")

    pass_message("All NPE Ratio observations lie within [0, 1].")

    # ========================================================
    # 10. DIMENSION COVERAGE
    # ========================================================

    section("10. DIMENSION COVERAGE")

    institution_keys = institution_dim.select(
        "lei_code"
    ).unique()

    period_column = (
        "period_code"
        if "period_code" in period_dim.columns
        else "period"
    )

    if period_column not in period_dim.columns:
        fail(
            "Unable to identify reporting-period key in dim_period."
        )

    period_keys = (
        period_dim
        .select(
            pl.col(period_column)
            .cast(pl.Int64)
            .alias("period")
        )
        .unique()
    )

    missing_institutions = (
        mart
        .select("lei_code")
        .unique()
        .join(
            institution_keys,
            on="lei_code",
            how="anti",
        )
    )

    missing_periods = (
        mart
        .select("period")
        .unique()
        .join(
            period_keys,
            on="period",
            how="anti",
        )
    )

    if missing_institutions.height > 0:
        print(missing_institutions)
        fail(
            "Some NPE mart institutions do not exist in dim_institution."
        )

    pass_message(
        "All NPE mart institutions exist in dim_institution."
    )

    if missing_periods.height > 0:
        print(missing_periods)
        fail(
            "Some NPE mart reporting periods do not exist in dim_period."
        )

    pass_message(
        "All NPE mart reporting periods exist in dim_period."
    )

    # ========================================================
    # 11. AGGREGATE FLAG VALIDATION
    # ========================================================

    section("11. AGGREGATE ENTITY FLAG VALIDATION")

    required_institution_columns = [
        "lei_code",
        "is_aggregate",
    ]

    for column_name in required_institution_columns:
        if column_name not in institution_dim.columns:
            fail(
                f"dim_institution missing required column "
                f"'{column_name}'."
            )

    aggregate_comparison = (
        mart
        .select(
            [
                "lei_code",
                "is_aggregate",
            ]
        )
        .unique()
        .join(
            institution_dim.select(
                [
                    "lei_code",
                    pl.col("is_aggregate").alias(
                        "expected_is_aggregate"
                    ),
                ]
            ),
            on="lei_code",
            how="left",
        )
    )

    aggregate_mismatches = aggregate_comparison.filter(
        pl.col("expected_is_aggregate").is_null()
        | (
            pl.col("is_aggregate")
            != pl.col("expected_is_aggregate")
        )
    )

    if aggregate_mismatches.height > 0:
        print(aggregate_mismatches)
        fail(
            "Saved aggregate flags do not reconcile to dim_institution."
        )

    pass_message(
        "All saved aggregate flags reconcile to dim_institution."
    )

    # ========================================================
    # 12. FINAL PROFILE
    # ========================================================

    section("12. FINAL NPE RATIO MART PROFILE")

    profile = mart.select(
        [
            pl.len().alias("rows"),
            pl.col("lei_code").n_unique().alias("institutions"),
            pl.col("period").n_unique().alias("periods"),
            pl.col("npe_ratio").min().alias("minimum_npe_ratio"),
            pl.col("npe_ratio").mean().alias("mean_npe_ratio"),
            pl.col("npe_ratio").median().alias("median_npe_ratio"),
            pl.col("npe_ratio").max().alias("maximum_npe_ratio"),
        ]
    )

    print(profile)

    period_profile = (
        mart
        .group_by("period")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("lei_code")
                .n_unique()
                .alias("institutions"),
                pl.col("npe_ratio")
                .mean()
                .alias("mean_npe_ratio"),
                pl.col("npe_ratio")
                .median()
                .alias("median_npe_ratio"),
            ]
        )
        .sort("period")
    )

    print()
    print("Period profile:")
    print(period_profile)

    # ========================================================
    # FINAL RESULT
    # ========================================================

    separator()
    print("NPE RATIO MART INDEPENDENT VALIDATION COMPLETED SUCCESSFULLY")
    separator()

    print()
    print("OVERALL STATUS : PASS")
    print(
        "INTERPRETATION : The saved NPE Ratio mart independently "
        "reconciles to the approved analytical fact definition, "
        "business grain, KPI formula, dimensions, and business controls."
    )
    print(
        "ACTION         : mart_npe_ratio can be approved for downstream "
        "benchmarking, trend analysis, statistical analysis, and "
        "management reporting."
    )

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No data was modified.")


if __name__ == "__main__":
    main()
