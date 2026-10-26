from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "facts"
    / "fact_other_templates.parquet"
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

MART_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_ratio.parquet"
)


CET1_CAPITAL_ITEM = 2520102
TOTAL_RISK_EXPOSURE_ITEM = 2520138
CET1_RATIO_ITEM = 2520140

FLOAT_TOLERANCE = 1e-12


EXPECTED_COLUMNS = [
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
    "cet1_capital",
    "total_risk_exposure_amount",
    "cet1_ratio",
    "cet1_ratio_pct",
    "reconstructed_cet1_ratio",
    "reconciliation_difference",
    "reconciliation_difference_abs",
    "zero_trea_flag",
]


EXPECTED_DTYPES = {
    "lei_code": pl.String,
    "institution_name": pl.String,
    "institution_country_code": pl.String,
    "institution_country": pl.String,
    "entity_type": pl.String,
    "is_aggregate": pl.Boolean,
    "period": pl.Int64,
    "period_end_date": pl.Date,
    "year": pl.Int32,
    "quarter": pl.Int8,
    "month": pl.Int8,
    "year_quarter": pl.String,
    "is_year_end": pl.Boolean,
    "cet1_capital": pl.Float64,
    "total_risk_exposure_amount": pl.Float64,
    "cet1_ratio": pl.Float64,
    "cet1_ratio_pct": pl.Float64,
    "reconstructed_cet1_ratio": pl.Float64,
    "reconciliation_difference": pl.Float64,
    "reconciliation_difference_abs": pl.Float64,
    "zero_trea_flag": pl.Boolean,
}


checks_passed = 0
checks_failed = 0


def header(title: str) -> None:
    print()
    print("=" * 105)
    print(title)
    print("=" * 105)


def check(condition: bool, message: str) -> None:
    global checks_passed
    global checks_failed

    if condition:
        checks_passed += 1
        print(f"[PASS] {message}")
    else:
        checks_failed += 1
        print(f"[FAIL] {message}")


def assert_unique_grain(
    df: pl.DataFrame,
    name: str,
) -> bool:

    duplicate_groups = (
        df.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    result = duplicate_groups.height == 0

    check(
        result,
        f"{name} is unique at grain: lei_code x period.",
    )

    if not result:
        print()
        print("Duplicate groups:")
        print(duplicate_groups.head(20))

    return result


def compare_float_column(
    joined: pl.DataFrame,
    expected_column: str,
    saved_column: str,
    description: str,
) -> None:

    difference_column = f"__diff_{saved_column}"

    comparison = joined.with_columns(
        (
            pl.col(saved_column)
            - pl.col(expected_column)
        )
        .abs()
        .alias(difference_column)
    )

    null_mismatch = comparison.filter(
        (
            pl.col(saved_column).is_null()
            & pl.col(expected_column).is_not_null()
        )
        |
        (
            pl.col(saved_column).is_not_null()
            & pl.col(expected_column).is_null()
        )
    )

    numeric_mismatch = comparison.filter(
        pl.col(saved_column).is_not_null()
        & pl.col(expected_column).is_not_null()
        & (
            pl.col(difference_column)
            > FLOAT_TOLERANCE
        )
    )

    result = (
        null_mismatch.height == 0
        and numeric_mismatch.height == 0
    )

    check(
        result,
        f"{description} independently reconciles.",
    )

    if not result:
        print()
        print(
            f"NULL mismatches   : "
            f"{null_mismatch.height:,}"
        )
        print(
            f"Numeric mismatches: "
            f"{numeric_mismatch.height:,}"
        )

        if numeric_mismatch.height > 0:
            print()
            print(
                numeric_mismatch
                .select(
                    [
                        "lei_code",
                        "period",
                        expected_column,
                        saved_column,
                        difference_column,
                    ]
                )
                .sort(
                    difference_column,
                    descending=True,
                )
                .head(20)
            )


def main() -> None:

    global checks_passed
    global checks_failed

    header("CET1 RATIO MART INDEPENDENT VALIDATION")

    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print()
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Source fact      : {FACT_PATH}")
    print(f"Institution dim  : {INSTITUTION_DIM_PATH}")
    print(f"Period dim       : {PERIOD_DIM_PATH}")
    print(f"CET1 mart        : {MART_PATH}")
    print()
    print("VALIDATION MODE: READ ONLY")
    print(
        "No analytical, staging, reference, "
        "or raw data will be modified."
    )

    print()
    print("INDEPENDENT VALIDATION PRINCIPLE:")
    print(
        "Expected CET1 mart values are reconstructed directly "
        "from fact_other_templates.parquet and dimensions."
    )
    print(
        "Saved CET1 mart analytical values are not used "
        "to derive expected results."
    )

    # ------------------------------------------------------------------
    # 1. FILE AVAILABILITY
    # ------------------------------------------------------------------

    header("1. FILE AVAILABILITY")

    required_files = [
        FACT_PATH,
        INSTITUTION_DIM_PATH,
        PERIOD_DIM_PATH,
        MART_PATH,
    ]

    for path in required_files:
        exists = path.exists()

        check(
            exists,
            f"Found: {path.name}",
        )

        if not exists:
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # ------------------------------------------------------------------
    # 2. LOAD DATA
    # ------------------------------------------------------------------

    header("2. LOAD VALIDATION INPUTS")

    fact = pl.read_parquet(FACT_PATH)
    institutions = pl.read_parquet(
        INSTITUTION_DIM_PATH
    )
    periods = pl.read_parquet(PERIOD_DIM_PATH)
    mart = pl.read_parquet(MART_PATH)

    check(
        fact.height > 0,
        f"Source fact opened: {fact.height:,} rows.",
    )

    check(
        institutions.height > 0,
        (
            "Institution dimension opened: "
            f"{institutions.height:,} rows."
        ),
    )

    check(
        periods.height > 0,
        (
            "Period dimension opened: "
            f"{periods.height:,} rows."
        ),
    )

    check(
        mart.height > 0,
        f"CET1 mart opened: {mart.height:,} rows.",
    )

    # ------------------------------------------------------------------
    # 3. SAVED MART STRUCTURAL VALIDATION
    # ------------------------------------------------------------------

    header("3. SAVED MART STRUCTURAL VALIDATION")

    check(
        mart.height == 469,
        (
            "CET1 mart contains expected "
            f"469 rows: {mart.height:,}."
        ),
    )

    check(
        mart.width == 21,
        (
            "CET1 mart contains expected "
            f"21 columns: {mart.width}."
        ),
    )

    check(
        mart.columns == EXPECTED_COLUMNS,
        (
            "CET1 mart column names and order "
            "match the approved schema."
        ),
    )

    assert_unique_grain(
        mart,
        "Saved CET1 mart",
    )

    business_grain_nulls = mart.filter(
        pl.col("lei_code").is_null()
        | pl.col("period").is_null()
    ).height

    check(
        business_grain_nulls == 0,
        (
            "Saved CET1 mart business grain "
            "contains no NULL values."
        ),
    )

    # ------------------------------------------------------------------
    # 4. DATA-TYPE VALIDATION
    # ------------------------------------------------------------------

    header("4. SAVED MART DATA-TYPE VALIDATION")

    for column, expected_dtype in EXPECTED_DTYPES.items():

        actual_dtype = mart.schema[column]

        check(
            actual_dtype == expected_dtype,
            (
                f"{column} dtype is {expected_dtype}. "
                f"Actual: {actual_dtype}."
            ),
        )

    # ------------------------------------------------------------------
    # 5. RECONSTRUCT SOURCE ITEMS
    # ------------------------------------------------------------------

    header("5. INDEPENDENT SOURCE ITEM RECONSTRUCTION")

    cet1_capital = (
        fact
        .filter(
            pl.col("item") == CET1_CAPITAL_ITEM
        )
        .select(
            [
                "lei_code",
                "period",
                pl.col("amount")
                .alias("expected_cet1_capital"),
            ]
        )
    )

    trea = (
        fact
        .filter(
            pl.col("item")
            == TOTAL_RISK_EXPOSURE_ITEM
        )
        .select(
            [
                "lei_code",
                "period",
                pl.col("amount").alias(
                    "expected_total_risk_exposure_amount"
                ),
            ]
        )
    )

    reported_ratio = (
        fact
        .filter(
            pl.col("item") == CET1_RATIO_ITEM
        )
        .select(
            [
                "lei_code",
                "period",
                pl.col("amount")
                .alias("expected_cet1_ratio"),
            ]
        )
    )

    check(
        cet1_capital.height == 469,
        (
            "Independent CET1 Capital source "
            "contains 469 observations."
        ),
    )

    check(
        trea.height == 469,
        (
            "Independent Total Risk Exposure source "
            "contains 469 observations."
        ),
    )

    check(
        reported_ratio.height == 469,
        (
            "Independent reported CET1 Ratio source "
            "contains 469 observations."
        ),
    )

    assert_unique_grain(
        cet1_capital,
        "Independent CET1 Capital source",
    )

    assert_unique_grain(
        trea,
        "Independent Total Risk Exposure source",
    )

    assert_unique_grain(
        reported_ratio,
        "Independent reported CET1 Ratio source",
    )

    # ------------------------------------------------------------------
    # 6. RECONSTRUCT INSTITUTION METADATA
    # ------------------------------------------------------------------

    header("6. INDEPENDENT INSTITUTION METADATA RECONSTRUCTION")

    required_institution_fields = [
        "lei_code",
        "name",
        "country",
        "desc_country",
        "entity_type",
        "is_aggregate",
    ]

    missing_institution_fields = [
        column
        for column in required_institution_fields
        if column not in institutions.columns
    ]

    check(
        len(missing_institution_fields) == 0,
        (
            "Institution dimension contains all "
            "required CET1 mart metadata fields."
        ),
    )

    if missing_institution_fields:
        raise ValueError(
            "Missing institution fields: "
            f"{missing_institution_fields}"
        )

    institution_lookup = (
        institutions
        .select(required_institution_fields)
        .rename(
            {
                "name": "expected_institution_name",
                "country": (
                    "expected_institution_country_code"
                ),
                "desc_country": (
                    "expected_institution_country"
                ),
                "entity_type": (
                    "expected_entity_type"
                ),
                "is_aggregate": (
                    "expected_is_aggregate"
                ),
            }
        )
    )

    check(
        institution_lookup[
            "lei_code"
        ].n_unique()
        == institution_lookup.height,
        (
            "Institution lookup is unique by lei_code."
        ),
    )

    # ------------------------------------------------------------------
    # 7. RECONSTRUCT PERIOD METADATA
    # ------------------------------------------------------------------

    header("7. INDEPENDENT PERIOD METADATA RECONSTRUCTION")

    required_period_fields = [
        "period_code",
        "period_end_date",
        "year",
        "quarter",
        "month",
        "year_quarter",
        "is_year_end",
    ]

    missing_period_fields = [
        column
        for column in required_period_fields
        if column not in periods.columns
    ]

    check(
        len(missing_period_fields) == 0,
        (
            "Period dimension contains all "
            "required CET1 mart metadata fields."
        ),
    )

    if missing_period_fields:
        raise ValueError(
            "Missing period fields: "
            f"{missing_period_fields}"
        )

    period_lookup = (
        periods
        .select(required_period_fields)
        .rename(
            {
                "period_code": "period",
                "period_end_date": (
                    "expected_period_end_date"
                ),
                "year": "expected_year",
                "quarter": "expected_quarter",
                "month": "expected_month",
                "year_quarter": (
                    "expected_year_quarter"
                ),
                "is_year_end": (
                    "expected_is_year_end"
                ),
            }
        )
        .with_columns(
            pl.col("period").cast(pl.Int64)
        )
    )

    check(
        period_lookup[
            "period"
        ].n_unique()
        == period_lookup.height,
        (
            "Period lookup is unique by reporting period."
        ),
    )

    # ------------------------------------------------------------------
    # 8. BUILD INDEPENDENT EXPECTED MART
    # ------------------------------------------------------------------

    header("8. BUILD INDEPENDENT EXPECTED CET1 MART")

    expected = (
        reported_ratio
        .join(
            cet1_capital,
            on=["lei_code", "period"],
            how="left",
        )
        .join(
            trea,
            on=["lei_code", "period"],
            how="left",
        )
        .join(
            institution_lookup,
            on="lei_code",
            how="left",
        )
        .join(
            period_lookup,
            on="period",
            how="left",
        )
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "expected_total_risk_exposure_amount"
                    )
                    != 0
                )
                .then(
                    pl.col(
                        "expected_cet1_capital"
                    )
                    /
                    pl.col(
                        "expected_total_risk_exposure_amount"
                    )
                )
                .otherwise(None)
                .alias(
                    "expected_reconstructed_cet1_ratio"
                ),

                (
                    pl.col(
                        "expected_cet1_ratio"
                    )
                    * 100
                )
                .alias(
                    "expected_cet1_ratio_pct"
                ),

                (
                    pl.col(
                        "expected_total_risk_exposure_amount"
                    )
                    == 0
                )
                .alias(
                    "expected_zero_trea_flag"
                ),
            ]
        )
        .with_columns(
            [
                (
                    pl.col(
                        "expected_cet1_ratio"
                    )
                    -
                    pl.col(
                        "expected_reconstructed_cet1_ratio"
                    )
                )
                .alias(
                    "expected_reconciliation_difference"
                ),
            ]
        )
        .with_columns(
            [
                pl.col(
                    "expected_reconciliation_difference"
                )
                .abs()
                .alias(
                    "expected_reconciliation_difference_abs"
                ),
            ]
        )
        .sort(
            [
                "lei_code",
                "period",
            ]
        )
    )

    check(
        expected.height == 469,
        (
            "Independent reconstruction created "
            "469 CET1 observations."
        ),
    )

    assert_unique_grain(
        expected,
        "Independent CET1 reconstruction",
    )

    # ------------------------------------------------------------------
    # 9. SOURCE COVERAGE VALIDATION
    # ------------------------------------------------------------------

    header("9. SOURCE COVERAGE VALIDATION")

    coverage = expected.select(
        [
            pl.col(
                "expected_cet1_capital"
            )
            .null_count()
            .alias("missing_cet1_capital"),

            pl.col(
                "expected_total_risk_exposure_amount"
            )
            .null_count()
            .alias("missing_trea"),

            pl.col(
                "expected_cet1_ratio"
            )
            .null_count()
            .alias("missing_reported_ratio"),

            (
                pl.col(
                    "expected_total_risk_exposure_amount"
                )
                == 0
            )
            .sum()
            .alias("zero_trea"),
        ]
    )

    print(coverage)

    check(
        coverage[
            "missing_cet1_capital"
        ][0] == 0,
        "No CET1 Capital values are missing.",
    )

    check(
        coverage[
            "missing_trea"
        ][0] == 0,
        (
            "No Total Risk Exposure Amount "
            "values are missing."
        ),
    )

    check(
        coverage[
            "missing_reported_ratio"
        ][0] == 0,
        "No directly reported CET1 Ratios are missing.",
    )

    check(
        coverage[
            "zero_trea"
        ][0] == 0,
        (
            "No zero Total Risk Exposure Amount "
            "denominators exist."
        ),
    )

    # ------------------------------------------------------------------
    # 10. ALIGN SAVED MART TO EXPECTED RECONSTRUCTION
    # ------------------------------------------------------------------

    header("10. SAVED MART TO INDEPENDENT RECONSTRUCTION ALIGNMENT")

    saved_for_join = mart.rename(
        {
            column: f"saved_{column}"
            for column in mart.columns
            if column not in ["lei_code", "period"]
        }
    )

    joined = expected.join(
        saved_for_join,
        on=["lei_code", "period"],
        how="inner",
    )

    check(
        joined.height == 469,
        (
            "Saved CET1 observations align one-to-one "
            "with independent reconstruction."
        ),
    )

    # ------------------------------------------------------------------
    # 11. SOURCE VALUE RECONCILIATION
    # ------------------------------------------------------------------

    header("11. SOURCE VALUE RECONCILIATION")

    compare_float_column(
        joined,
        "expected_cet1_capital",
        "saved_cet1_capital",
        "CET1 Capital",
    )

    compare_float_column(
        joined,
        "expected_total_risk_exposure_amount",
        "saved_total_risk_exposure_amount",
        "Total Risk Exposure Amount",
    )

    compare_float_column(
        joined,
        "expected_cet1_ratio",
        "saved_cet1_ratio",
        "Direct reported CET1 Ratio",
    )

    # ------------------------------------------------------------------
    # 12. DERIVED VALUE RECONCILIATION
    # ------------------------------------------------------------------

    header("12. DERIVED CET1 FIELD RECONCILIATION")

    compare_float_column(
        joined,
        "expected_cet1_ratio_pct",
        "saved_cet1_ratio_pct",
        "CET1 Ratio percentage",
    )

    compare_float_column(
        joined,
        "expected_reconstructed_cet1_ratio",
        "saved_reconstructed_cet1_ratio",
        "Reconstructed CET1 Ratio",
    )

    compare_float_column(
        joined,
        "expected_reconciliation_difference",
        "saved_reconciliation_difference",
        "CET1 reconciliation difference",
    )

    compare_float_column(
        joined,
        "expected_reconciliation_difference_abs",
        "saved_reconciliation_difference_abs",
        "Absolute CET1 reconciliation difference",
    )

    # ------------------------------------------------------------------
    # 13. BOOLEAN / FLAG RECONCILIATION
    # ------------------------------------------------------------------

    header("13. FLAG RECONCILIATION")

    zero_flag_mismatch = joined.filter(
        pl.col("expected_zero_trea_flag")
        != pl.col("saved_zero_trea_flag")
    )

    check(
        zero_flag_mismatch.height == 0,
        (
            "Zero-denominator flag independently "
            "reconciles exactly."
        ),
    )

    aggregate_flag_mismatch = joined.filter(
        pl.col("expected_is_aggregate")
        != pl.col("saved_is_aggregate")
    )

    check(
        aggregate_flag_mismatch.height == 0,
        (
            "Aggregate/institution flag independently "
            "reconciles exactly."
        ),
    )

    # ------------------------------------------------------------------
    # 14. INSTITUTION METADATA RECONCILIATION
    # ------------------------------------------------------------------

    header("14. INSTITUTION METADATA RECONCILIATION")

    institution_comparisons = [
        (
            "expected_institution_name",
            "saved_institution_name",
            "Institution name",
        ),
        (
            "expected_institution_country_code",
            "saved_institution_country_code",
            "Institution country code",
        ),
        (
            "expected_institution_country",
            "saved_institution_country",
            "Institution country",
        ),
        (
            "expected_entity_type",
            "saved_entity_type",
            "Entity type",
        ),
    ]

    for expected_col, saved_col, description in (
        institution_comparisons
    ):

        mismatches = joined.filter(
            pl.col(expected_col)
            != pl.col(saved_col)
        )

        check(
            mismatches.height == 0,
            (
                f"{description} independently "
                "reconciles exactly."
            ),
        )

    # ------------------------------------------------------------------
    # 15. PERIOD METADATA RECONCILIATION
    # ------------------------------------------------------------------

    header("15. PERIOD METADATA RECONCILIATION")

    period_comparisons = [
        (
            "expected_period_end_date",
            "saved_period_end_date",
            "Period end date",
        ),
        (
            "expected_year",
            "saved_year",
            "Year",
        ),
        (
            "expected_quarter",
            "saved_quarter",
            "Quarter",
        ),
        (
            "expected_month",
            "saved_month",
            "Month",
        ),
        (
            "expected_year_quarter",
            "saved_year_quarter",
            "Year-quarter label",
        ),
        (
            "expected_is_year_end",
            "saved_is_year_end",
            "Year-end flag",
        ),
    ]

    for expected_col, saved_col, description in (
        period_comparisons
    ):

        mismatches = joined.filter(
            pl.col(expected_col)
            != pl.col(saved_col)
        )

        check(
            mismatches.height == 0,
            (
                f"{description} independently "
                "reconciles exactly."
            ),
        )

    # ------------------------------------------------------------------
    # 16. DIRECT-TO-RECONSTRUCTED BUSINESS RULE
    # ------------------------------------------------------------------

    header("16. CET1 RATIO RECONCILIATION BUSINESS RULE")

    max_difference = expected.select(
        pl.col(
            "expected_reconciliation_difference_abs"
        ).max()
    ).item()

    print(
        "Maximum independent direct/reconstructed "
        f"difference: {max_difference:.16g}"
    )

    check(
        max_difference <= FLOAT_TOLERANCE,
        (
            "Direct EBA CET1 Ratio reconciles to "
            "CET1 Capital / Total Risk Exposure Amount "
            f"within tolerance {FLOAT_TOLERANCE}."
        ),
    )

    # ------------------------------------------------------------------
    # 17. OUTLIER PRESERVATION CONTROL
    # ------------------------------------------------------------------

    header("17. HIGH CET1 RATIO PRESERVATION CONTROL")

    expected_high = expected.filter(
        pl.col("expected_cet1_ratio") > 1.0
    )

    saved_high = mart.filter(
        pl.col("cet1_ratio") > 1.0
    )

    print(
        f"Independent high-ratio observations : "
        f"{expected_high.height:,}"
    )
    print(
        f"Saved high-ratio observations       : "
        f"{saved_high.height:,}"
    )

    check(
        expected_high.height == 6,
        (
            "Independent source reconstruction contains "
            "the expected six CET1 Ratio observations >100%."
        ),
    )

    check(
        saved_high.height == expected_high.height,
        (
            "Saved mart preserves all high CET1 Ratio "
            "observations without deletion or capping."
        ),
    )

    high_keys_expected = set(
        expected_high
        .select(["lei_code", "period"])
        .iter_rows()
    )

    high_keys_saved = set(
        saved_high
        .select(["lei_code", "period"])
        .iter_rows()
    )

    check(
        high_keys_expected == high_keys_saved,
        (
            "High CET1 Ratio entity-period observations "
            "match source reconstruction exactly."
        ),
    )

    # ------------------------------------------------------------------
    # 18. POPULATION PROFILE
    # ------------------------------------------------------------------

    header("18. CET1 MART POPULATION PROFILE")

    population_profile = mart.select(
        [
            pl.len().alias("rows"),
            pl.col("lei_code")
            .n_unique()
            .alias("entities"),
            pl.col("period")
            .n_unique()
            .alias("periods"),
            pl.col("is_aggregate")
            .sum()
            .alias("aggregate_rows"),
            (~pl.col("is_aggregate"))
            .sum()
            .alias("institution_rows"),
        ]
    )

    print(population_profile)

    check(
        mart[
            "lei_code"
        ].n_unique() == 120,
        "CET1 mart contains 120 analytical entities.",
    )

    check(
        mart[
            "period"
        ].n_unique() == 4,
        "CET1 mart contains four reporting periods.",
    )

    check(
        mart.filter(
            pl.col("is_aggregate")
        ).height == 4,
        (
            "CET1 mart contains four aggregate "
            "entity-period observations."
        ),
    )

    check(
        mart.filter(
            ~pl.col("is_aggregate")
        ).height == 465,
        (
            "CET1 mart contains 465 institution-level "
            "observations."
        ),
    )

    # ------------------------------------------------------------------
    # 19. PERIOD PROFILE
    # ------------------------------------------------------------------

    header("19. PERIOD PROFILE")

    period_profile = (
        mart
        .group_by("period")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("lei_code")
                .n_unique()
                .alias("entities"),
                pl.col("cet1_ratio")
                .median()
                .alias("median_cet1_ratio"),
                pl.col("cet1_ratio")
                .min()
                .alias("minimum_cet1_ratio"),
                pl.col("cet1_ratio")
                .max()
                .alias("maximum_cet1_ratio"),
            ]
        )
        .sort("period")
    )

    print(period_profile)

    expected_period_counts = {
        202409: 117,
        202412: 120,
        202503: 112,
        202506: 120,
    }

    for period, expected_count in (
        expected_period_counts.items()
    ):

        actual_count = mart.filter(
            pl.col("period") == period
        ).height

        check(
            actual_count == expected_count,
            (
                f"Period {period} contains "
                f"{expected_count} observations."
            ),
        )

    # ------------------------------------------------------------------
    # 20. FINAL SUMMARY
    # ------------------------------------------------------------------

    header("CET1 RATIO MART INDEPENDENT VALIDATION SUMMARY")

    total_checks = checks_passed + checks_failed

    print(f"Total checks : {total_checks}")
    print(f"PASS         : {checks_passed}")
    print(f"FAIL         : {checks_failed}")

    print()
    print("-" * 105)

    if checks_failed == 0:

        print("OVERALL STATUS : PASS")

        print()
        print(
            "INTERPRETATION:"
        )
        print(
            "The saved CET1 Ratio analytical mart "
            "independently reconciles to the approved "
            "EBA source items and analytical dimensions."
        )

        print()
        print("Validated controls include:")
        print("  - institution-period business grain")
        print("  - exact analytical schema")
        print("  - source item coverage")
        print("  - CET1 Capital reconciliation")
        print("  - Total Risk Exposure reconciliation")
        print("  - directly reported CET1 Ratio")
        print("  - independently reconstructed CET1 Ratio")
        print("  - institution metadata")
        print("  - period metadata")
        print("  - zero-denominator control")
        print("  - aggregate entity preservation")
        print("  - >100% CET1 Ratio outlier preservation")

        print()
        print(
            "ACTION: mart_cet1_ratio can be considered "
            "for approval for downstream capital analytics "
            "and reporting."
        )

    else:

        print("OVERALL STATUS : FAIL")

        print()
        print(
            "ACTION: Do not approve mart_cet1_ratio "
            "for downstream use until failed controls "
            "are investigated and resolved."
        )

        raise SystemExit(1)

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No data was modified.")


if __name__ == "__main__":
    main()
