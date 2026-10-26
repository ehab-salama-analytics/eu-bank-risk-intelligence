from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_benchmark.parquet"
)

TREND_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_trend.parquet"
)

FLOAT_TOLERANCE = 1e-12


BASE_COLUMNS = [
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
    "benchmark_eligible",
    "eu_peer_count",
    "eu_mean_cet1_ratio",
    "eu_median_cet1_ratio",
    "eu_rank",
    "eu_percentile",
    "eu_percentile_pct",
    "country_peer_count",
    "country_descriptive_available",
    "country_benchmark_available",
    "country_benchmark_status",
    "country_mean_cet1_ratio",
    "country_median_cet1_ratio",
    "country_rank",
    "country_percentile",
    "country_percentile_pct",
    "eu_deviation_from_median",
    "country_deviation_from_median",
]


TREND_COLUMNS = [
    "previous_period",
    "quarter_gap",
    "qoq_available",
    "previous_cet1_ratio",
    "previous_cet1_ratio_pct",
    "cet1_ratio_change_qoq",
    "cet1_ratio_pp_change_qoq",
    "eu_percentile_qoq_available",
    "previous_eu_percentile",
    "eu_percentile_change_qoq",
    "country_percentile_qoq_available",
    "previous_country_percentile",
    "country_percentile_change_qoq",
    "three_period_pattern_available",
    "period_two_quarters_ago",
    "cet1_ratio_two_quarters_ago",
    "cet1_ratio_pct_two_quarters_ago",
    "cet1_ratio_change_3_period",
    "cet1_ratio_pp_change_3_period",
    "cet1_decrease_count_3_period",
    "cet1_increase_count_3_period",
    "cet1_three_period_pattern",
]


EXPECTED_COLUMNS = BASE_COLUMNS + TREND_COLUMNS


EXPECTED_TREND_DTYPES = {
    "previous_period": pl.Int64,
    "quarter_gap": pl.Int64,
    "qoq_available": pl.Boolean,
    "previous_cet1_ratio": pl.Float64,
    "previous_cet1_ratio_pct": pl.Float64,
    "cet1_ratio_change_qoq": pl.Float64,
    "cet1_ratio_pp_change_qoq": pl.Float64,
    "eu_percentile_qoq_available": pl.Boolean,
    "previous_eu_percentile": pl.Float64,
    "eu_percentile_change_qoq": pl.Float64,
    "country_percentile_qoq_available": pl.Boolean,
    "previous_country_percentile": pl.Float64,
    "country_percentile_change_qoq": pl.Float64,
    "three_period_pattern_available": pl.Boolean,
    "period_two_quarters_ago": pl.Int64,
    "cet1_ratio_two_quarters_ago": pl.Float64,
    "cet1_ratio_pct_two_quarters_ago": pl.Float64,
    "cet1_ratio_change_3_period": pl.Float64,
    "cet1_ratio_pp_change_3_period": pl.Float64,
    "cet1_decrease_count_3_period": pl.Int8,
    "cet1_increase_count_3_period": pl.Int8,
    "cet1_three_period_pattern": pl.String,
}


FLOAT_TREND_COLUMNS = {
    "previous_cet1_ratio",
    "previous_cet1_ratio_pct",
    "cet1_ratio_change_qoq",
    "cet1_ratio_pp_change_qoq",
    "previous_eu_percentile",
    "eu_percentile_change_qoq",
    "previous_country_percentile",
    "country_percentile_change_qoq",
    "cet1_ratio_two_quarters_ago",
    "cet1_ratio_pct_two_quarters_ago",
    "cet1_ratio_change_3_period",
    "cet1_ratio_pp_change_3_period",
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
) -> None:

    duplicates = (
        df.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    check(
        duplicates.height == 0,
        f"{name} is unique at grain: lei_code x period.",
    )

    if duplicates.height > 0:
        print(duplicates.head(20))


def compare_exact(
    joined: pl.DataFrame,
    expected_col: str,
    saved_col: str,
    description: str,
) -> None:

    mismatches = joined.filter(
        (
            pl.col(expected_col).is_null()
            != pl.col(saved_col).is_null()
        )
        |
        (
            pl.col(expected_col).is_not_null()
            & pl.col(saved_col).is_not_null()
            & (
                pl.col(expected_col)
                != pl.col(saved_col)
            )
        )
    )

    check(
        mismatches.height == 0,
        f"{description} reconciles exactly.",
    )

    if mismatches.height > 0:
        print(
            mismatches
            .select(
                [
                    "lei_code",
                    "period",
                    expected_col,
                    saved_col,
                ]
            )
            .head(20)
        )


def compare_float(
    joined: pl.DataFrame,
    expected_col: str,
    saved_col: str,
    description: str,
) -> None:

    diff_col = f"__diff_{saved_col}"

    comparison = joined.with_columns(
        (
            pl.col(expected_col)
            - pl.col(saved_col)
        )
        .abs()
        .alias(diff_col)
    )

    mismatches = comparison.filter(
        (
            pl.col(expected_col).is_null()
            != pl.col(saved_col).is_null()
        )
        |
        (
            pl.col(expected_col).is_not_null()
            & pl.col(saved_col).is_not_null()
            & (
                pl.col(diff_col) > FLOAT_TOLERANCE
            )
        )
    )

    check(
        mismatches.height == 0,
        f"{description} independently reconciles.",
    )

    if mismatches.height > 0:
        print(
            mismatches
            .select(
                [
                    "lei_code",
                    "period",
                    expected_col,
                    saved_col,
                    diff_col,
                ]
            )
            .sort(
                diff_col,
                descending=True,
            )
            .head(20)
        )


def main() -> None:

    global checks_passed
    global checks_failed

    header("CET1 TREND MART INDEPENDENT VALIDATION")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source mart  : {SOURCE_PATH}")
    print(f"Trend mart   : {TREND_PATH}")
    print()
    print("VALIDATION MODE: READ ONLY")
    print("No data will be modified.")
    print()
    print(
        "Expected trend values are reconstructed independently "
        "from mart_cet1_benchmark.parquet."
    )

    # ------------------------------------------------------------------
    # 1. FILE AVAILABILITY
    # ------------------------------------------------------------------

    header("1. FILE AVAILABILITY")

    for path in [SOURCE_PATH, TREND_PATH]:

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

    source = pl.read_parquet(SOURCE_PATH)
    saved = pl.read_parquet(TREND_PATH)

    check(
        source.height == 469,
        "Source CET1 benchmark mart contains 469 rows.",
    )

    check(
        source.width == 39,
        "Source CET1 benchmark mart contains 39 columns.",
    )

    check(
        saved.height == 469,
        "Saved CET1 trend mart contains 469 rows.",
    )

    check(
        saved.width == 61,
        "Saved CET1 trend mart contains 61 columns.",
    )

    assert_unique_grain(
        source,
        "Source CET1 benchmark mart",
    )

    assert_unique_grain(
        saved,
        "Saved CET1 trend mart",
    )

    # ------------------------------------------------------------------
    # 3. SCHEMA VALIDATION
    # ------------------------------------------------------------------

    header("3. SAVED TREND MART SCHEMA VALIDATION")

    check(
        saved.columns == EXPECTED_COLUMNS,
        (
            "Saved trend mart column names and order "
            "match the approved 61-column schema."
        ),
    )

    for column, expected_dtype in (
        EXPECTED_TREND_DTYPES.items()
    ):

        actual_dtype = saved.schema[column]

        check(
            actual_dtype == expected_dtype,
            (
                f"{column} dtype is {expected_dtype}. "
                f"Actual: {actual_dtype}."
            ),
        )

    # ------------------------------------------------------------------
    # 4. SOURCE FIELD PRESERVATION
    # ------------------------------------------------------------------

    header("4. BENCHMARK SOURCE FIELD PRESERVATION")

    source_sorted = source.sort(
        ["lei_code", "period"]
    )

    saved_sorted = saved.sort(
        ["lei_code", "period"]
    )

    check(
        saved_sorted.select(BASE_COLUMNS).equals(
            source_sorted.select(BASE_COLUMNS)
        ),
        (
            "All 39 approved benchmark fields are "
            "preserved exactly in the trend mart."
        ),
    )

    # ------------------------------------------------------------------
    # 5. RECONSTRUCT INSTITUTION TREND BASE
    # ------------------------------------------------------------------

    header("5. INDEPENDENT INSTITUTION TREND RECONSTRUCTION")

    institutions = (
        source
        .filter(~pl.col("is_aggregate"))
        .sort(
            [
                "lei_code",
                "period",
            ]
        )
    )

    aggregates = (
        source
        .filter(pl.col("is_aggregate"))
        .sort(
            [
                "lei_code",
                "period",
            ]
        )
    )

    check(
        institutions.height == 465,
        (
            "Independent trend population contains "
            "465 institution observations."
        ),
    )

    check(
        aggregates.height == 4,
        (
            "Independent reconstruction contains "
            "four aggregate observations."
        ),
    )

    # Use the validated year + quarter metadata to construct
    # quarter sequence independently from the transformation script.
    institutions = institutions.with_columns(
        (
            pl.col("year").cast(pl.Int64) * 4
            + pl.col("quarter").cast(pl.Int64)
        ).alias("__quarter_index")
    )

    institutions = institutions.with_columns(
        [
            pl.col("period")
            .shift(1)
            .over("lei_code")
            .alias("previous_period"),

            pl.col("__quarter_index")
            .shift(1)
            .over("lei_code")
            .alias("__previous_quarter_index"),

            pl.col("cet1_ratio")
            .shift(1)
            .over("lei_code")
            .alias("previous_cet1_ratio"),

            pl.col("cet1_ratio_pct")
            .shift(1)
            .over("lei_code")
            .alias("previous_cet1_ratio_pct"),

            pl.col("eu_percentile")
            .shift(1)
            .over("lei_code")
            .alias("previous_eu_percentile"),

            pl.col("country_percentile")
            .shift(1)
            .over("lei_code")
            .alias("previous_country_percentile"),

            pl.col("period")
            .shift(2)
            .over("lei_code")
            .alias("__period_two_ago"),

            pl.col("__quarter_index")
            .shift(2)
            .over("lei_code")
            .alias("__quarter_index_two_ago"),

            pl.col("cet1_ratio")
            .shift(2)
            .over("lei_code")
            .alias("__cet1_two_ago"),

            pl.col("cet1_ratio_pct")
            .shift(2)
            .over("lei_code")
            .alias("__cet1_pct_two_ago"),
        ]
    )

    institutions = institutions.with_columns(
        [
            (
                pl.col("__quarter_index")
                - pl.col("__previous_quarter_index")
            )
            .alias("quarter_gap"),

            (
                pl.col("__previous_quarter_index")
                - pl.col("__quarter_index_two_ago")
            )
            .alias("__previous_gap"),
        ]
    )

    institutions = institutions.with_columns(
        [
            (
                pl.col("quarter_gap") == 1
            )
            .fill_null(False)
            .alias("qoq_available"),

            (
                (
                    pl.col("quarter_gap") == 1
                )
                &
                (
                    pl.col("__previous_gap") == 1
                )
            )
            .fill_null(False)
            .alias("three_period_pattern_available"),
        ]
    )

    # ------------------------------------------------------------------
    # 6. QOQ CET1 RECONSTRUCTION
    # ------------------------------------------------------------------

    header("6. INDEPENDENT QOQ CET1 RECONSTRUCTION")

    institutions = institutions.with_columns(
        [
            pl.when(
                pl.col("qoq_available")
            )
            .then(
                pl.col("cet1_ratio")
                - pl.col("previous_cet1_ratio")
            )
            .otherwise(None)
            .alias("cet1_ratio_change_qoq"),

            pl.when(
                pl.col("qoq_available")
            )
            .then(
                pl.col("cet1_ratio_pct")
                - pl.col("previous_cet1_ratio_pct")
            )
            .otherwise(None)
            .alias("cet1_ratio_pp_change_qoq"),
        ]
    )

    prior_count = institutions.filter(
        pl.col("previous_period").is_not_null()
    ).height

    qoq_count = institutions.filter(
        pl.col("qoq_available")
    ).height

    nonconsecutive_count = institutions.filter(
        pl.col("previous_period").is_not_null()
        & (pl.col("quarter_gap") != 1)
    ).height

    check(
        prior_count == 346,
        (
            "Independent reconstruction finds "
            "346 previous-observation pairs."
        ),
    )

    check(
        qoq_count == 338,
        (
            "Independent reconstruction finds "
            "338 genuine QoQ pairs."
        ),
    )

    check(
        nonconsecutive_count == 8,
        (
            "Independent reconstruction finds "
            "eight non-consecutive previous-observation pairs."
        ),
    )

    # ------------------------------------------------------------------
    # 7. EUROPEAN PEER MOVEMENT
    # ------------------------------------------------------------------

    header("7. EUROPEAN PEER MOVEMENT RECONSTRUCTION")

    institutions = institutions.with_columns(
        (
            pl.col("qoq_available")
            & pl.col("eu_percentile").is_not_null()
            & pl.col("previous_eu_percentile").is_not_null()
        )
        .fill_null(False)
        .alias("eu_percentile_qoq_available")
    )

    institutions = institutions.with_columns(
        pl.when(
            pl.col("eu_percentile_qoq_available")
        )
        .then(
            pl.col("eu_percentile")
            - pl.col("previous_eu_percentile")
        )
        .otherwise(None)
        .alias("eu_percentile_change_qoq")
    )

    eu_count = institutions.filter(
        pl.col("eu_percentile_qoq_available")
    ).height

    check(
        eu_count == 338,
        (
            "Independent reconstruction finds "
            "338 EU-percentile QoQ comparisons."
        ),
    )

    # ------------------------------------------------------------------
    # 8. COUNTRY PEER MOVEMENT
    # ------------------------------------------------------------------

    header("8. COUNTRY PEER MOVEMENT RECONSTRUCTION")

    institutions = institutions.with_columns(
        (
            pl.col("qoq_available")
            & pl.col("country_percentile").is_not_null()
            & pl.col(
                "previous_country_percentile"
            ).is_not_null()
        )
        .fill_null(False)
        .alias("country_percentile_qoq_available")
    )

    institutions = institutions.with_columns(
        pl.when(
            pl.col(
                "country_percentile_qoq_available"
            )
        )
        .then(
            pl.col("country_percentile")
            - pl.col("previous_country_percentile")
        )
        .otherwise(None)
        .alias("country_percentile_change_qoq")
    )

    country_count = institutions.filter(
        pl.col("country_percentile_qoq_available")
    ).height

    check(
        country_count == 231,
        (
            "Independent reconstruction finds "
            "231 formal country-percentile QoQ comparisons."
        ),
    )

    # ------------------------------------------------------------------
    # 9. THREE-PERIOD PATTERN RECONSTRUCTION
    # ------------------------------------------------------------------

    header("9. THREE-PERIOD PATTERN RECONSTRUCTION")

    institutions = institutions.with_columns(
        [
            (
                pl.col("previous_cet1_ratio")
                - pl.col("__cet1_two_ago")
            )
            .alias("__first_change"),

            (
                pl.col("cet1_ratio")
                - pl.col("previous_cet1_ratio")
            )
            .alias("__second_change"),
        ]
    )

    pattern = (
        pl.when(
            (pl.col("__first_change") < 0)
            &
            (pl.col("__second_change") < 0)
        )
        .then(pl.lit("consecutive_decrease"))

        .when(
            (pl.col("__first_change") > 0)
            &
            (pl.col("__second_change") > 0)
        )
        .then(pl.lit("consecutive_increase"))

        .when(
            (pl.col("__first_change") < 0)
            &
            (pl.col("__second_change") > 0)
        )
        .then(pl.lit("decrease_then_increase"))

        .when(
            (pl.col("__first_change") > 0)
            &
            (pl.col("__second_change") < 0)
        )
        .then(pl.lit("increase_then_decrease"))

        .otherwise(
            pl.lit("stable_or_mixed")
        )
    )

    institutions = institutions.with_columns(
        [
            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                pl.col("__period_two_ago")
            )
            .otherwise(None)
            .cast(pl.Int64)
            .alias("period_two_quarters_ago"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                pl.col("__cet1_two_ago")
            )
            .otherwise(None)
            .alias("cet1_ratio_two_quarters_ago"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                pl.col("__cet1_pct_two_ago")
            )
            .otherwise(None)
            .alias("cet1_ratio_pct_two_quarters_ago"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                pl.col("cet1_ratio")
                - pl.col("__cet1_two_ago")
            )
            .otherwise(None)
            .alias("cet1_ratio_change_3_period"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                pl.col("cet1_ratio_pct")
                - pl.col("__cet1_pct_two_ago")
            )
            .otherwise(None)
            .alias("cet1_ratio_pp_change_3_period"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                (
                    (pl.col("__first_change") < 0)
                    .cast(pl.Int8)
                    +
                    (pl.col("__second_change") < 0)
                    .cast(pl.Int8)
                )
            )
            .otherwise(None)
            .cast(pl.Int8)
            .alias("cet1_decrease_count_3_period"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                (
                    (pl.col("__first_change") > 0)
                    .cast(pl.Int8)
                    +
                    (pl.col("__second_change") > 0)
                    .cast(pl.Int8)
                )
            )
            .otherwise(None)
            .cast(pl.Int8)
            .alias("cet1_increase_count_3_period"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(pattern)
            .otherwise(None)
            .alias("cet1_three_period_pattern"),
        ]
    )

    three_period_count = institutions.filter(
        pl.col("three_period_pattern_available")
    ).height

    check(
        three_period_count == 222,
        (
            "Independent reconstruction finds "
            "222 continuous three-quarter observations."
        ),
    )

    pattern_profile = (
        institutions
        .filter(
            pl.col("three_period_pattern_available")
        )
        .group_by("cet1_three_period_pattern")
        .agg(
            pl.len().alias("rows")
        )
        .sort("cet1_three_period_pattern")
    )

    print(pattern_profile)

    expected_patterns = {
        "consecutive_decrease": 42,
        "consecutive_increase": 64,
        "decrease_then_increase": 53,
        "increase_then_decrease": 63,
    }

    for pattern_name, expected_count in (
        expected_patterns.items()
    ):

        actual = institutions.filter(
            pl.col("cet1_three_period_pattern")
            == pattern_name
        ).height

        check(
            actual == expected_count,
            (
                f"{pattern_name} contains "
                f"{expected_count} observations."
            ),
        )

    stable_count = institutions.filter(
        pl.col("cet1_three_period_pattern")
        == "stable_or_mixed"
    ).height

    check(
        stable_count == 0,
        (
            "No continuous three-period CET1 chains "
            "are stable_or_mixed."
        ),
    )

    # ------------------------------------------------------------------
    # 10. DIRECTIONAL PROFILE
    # ------------------------------------------------------------------

    header("10. QOQ CET1 DIRECTIONAL PROFILE")

    genuine_qoq = institutions.filter(
        pl.col("qoq_available")
    )

    decreases = genuine_qoq.filter(
        pl.col("cet1_ratio_change_qoq") < 0
    ).height

    increases = genuine_qoq.filter(
        pl.col("cet1_ratio_change_qoq") > 0
    ).height

    unchanged = genuine_qoq.filter(
        pl.col("cet1_ratio_change_qoq") == 0
    ).height

    print(f"CET1 decreases : {decreases}")
    print(f"CET1 increases : {increases}")
    print(f"Unchanged      : {unchanged}")

    check(
        decreases == 152,
        "Independent reconstruction finds 152 CET1 decreases.",
    )

    check(
        increases == 186,
        "Independent reconstruction finds 186 CET1 increases.",
    )

    check(
        unchanged == 0,
        "Independent reconstruction finds zero unchanged QoQ observations.",
    )

    # ------------------------------------------------------------------
    # 11. REMOVE TEMPORARY FIELDS
    # ------------------------------------------------------------------

    institutions = institutions.drop(
        [
            "__quarter_index",
            "__previous_quarter_index",
            "__quarter_index_two_ago",
            "__previous_gap",
            "__period_two_ago",
            "__cet1_two_ago",
            "__cet1_pct_two_ago",
            "__first_change",
            "__second_change",
        ]
    )

    # ------------------------------------------------------------------
    # 12. RECONSTRUCT AGGREGATE TREATMENT
    # ------------------------------------------------------------------

    header("11. AGGREGATE TREND TREATMENT")

    aggregates = aggregates.with_columns(
        [
            pl.lit(None)
            .cast(pl.Int64)
            .alias("previous_period"),

            pl.lit(None)
            .cast(pl.Int64)
            .alias("quarter_gap"),

            pl.lit(False)
            .alias("qoq_available"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("previous_cet1_ratio"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("previous_cet1_ratio_pct"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("cet1_ratio_change_qoq"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("cet1_ratio_pp_change_qoq"),

            pl.lit(False)
            .alias("eu_percentile_qoq_available"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("previous_eu_percentile"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("eu_percentile_change_qoq"),

            pl.lit(False)
            .alias("country_percentile_qoq_available"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("previous_country_percentile"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("country_percentile_change_qoq"),

            pl.lit(False)
            .alias("three_period_pattern_available"),

            pl.lit(None)
            .cast(pl.Int64)
            .alias("period_two_quarters_ago"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("cet1_ratio_two_quarters_ago"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("cet1_ratio_pct_two_quarters_ago"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("cet1_ratio_change_3_period"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("cet1_ratio_pp_change_3_period"),

            pl.lit(None)
            .cast(pl.Int8)
            .alias("cet1_decrease_count_3_period"),

            pl.lit(None)
            .cast(pl.Int8)
            .alias("cet1_increase_count_3_period"),

            pl.lit(None)
            .cast(pl.String)
            .alias("cet1_three_period_pattern"),
        ]
    )

    # ------------------------------------------------------------------
    # 13. BUILD EXPECTED MART
    # ------------------------------------------------------------------

    header("12. BUILD INDEPENDENT EXPECTED CET1 TREND MART")

    expected = (
        pl.concat(
            [
                institutions.select(EXPECTED_COLUMNS),
                aggregates.select(EXPECTED_COLUMNS),
            ],
            how="vertical_relaxed",
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
            "Independent CET1 trend reconstruction "
            "contains 469 rows."
        ),
    )

    check(
        expected.width == 61,
        (
            "Independent CET1 trend reconstruction "
            "contains 61 columns."
        ),
    )

    assert_unique_grain(
        expected,
        "Independent CET1 trend reconstruction",
    )

    # ------------------------------------------------------------------
    # 14. ALIGN EXPECTED TO SAVED
    # ------------------------------------------------------------------

    header("13. EXPECTED-TO-SAVED ALIGNMENT")

    expected_prefixed = expected.rename(
        {
            column: f"expected_{column}"
            for column in TREND_COLUMNS
        }
    )

    saved_prefixed = saved.select(
        ["lei_code", "period"] + TREND_COLUMNS
    ).rename(
        {
            column: f"saved_{column}"
            for column in TREND_COLUMNS
        }
    )

    joined = expected_prefixed.join(
        saved_prefixed,
        on=["lei_code", "period"],
        how="inner",
    )

    check(
        joined.height == 469,
        (
            "Saved trend observations align one-to-one "
            "with independent reconstruction."
        ),
    )

    # ------------------------------------------------------------------
    # 15. TREND FIELD RECONCILIATION
    # ------------------------------------------------------------------

    header("14. TREND FIELD RECONCILIATION")

    for column in TREND_COLUMNS:

        expected_col = f"expected_{column}"
        saved_col = f"saved_{column}"

        if column in FLOAT_TREND_COLUMNS:
            compare_float(
                joined,
                expected_col,
                saved_col,
                column,
            )
        else:
            compare_exact(
                joined,
                expected_col,
                saved_col,
                column,
            )

    # ------------------------------------------------------------------
    # 16. NON-CONSECUTIVE QOQ CONTROL
    # ------------------------------------------------------------------

    header("15. NON-CONSECUTIVE HISTORY CONTROL")

    nonconsecutive_saved = saved.filter(
        (~pl.col("is_aggregate"))
        & pl.col("previous_period").is_not_null()
        & (pl.col("quarter_gap") != 1)
    )

    print(
        f"Non-consecutive observations: "
        f"{nonconsecutive_saved.height}"
    )

    check(
        nonconsecutive_saved.height == 8,
        (
            "Saved mart retains exactly eight "
            "non-consecutive previous observations."
        ),
    )

    check(
        nonconsecutive_saved.filter(
            pl.col("qoq_available")
        ).height == 0,
        (
            "No non-consecutive observation is "
            "classified as genuine QoQ."
        ),
    )

    check(
        nonconsecutive_saved.filter(
            pl.col("cet1_ratio_change_qoq").is_not_null()
            | pl.col(
                "cet1_ratio_pp_change_qoq"
            ).is_not_null()
        ).height == 0,
        (
            "Non-consecutive histories have no "
            "CET1 QoQ metrics."
        ),
    )

    print()
    print(
        nonconsecutive_saved
        .select(
            [
                "institution_name",
                "institution_country",
                "previous_period",
                "period",
                "quarter_gap",
                "qoq_available",
            ]
        )
        .sort(
            [
                "institution_country",
                "institution_name",
            ]
        )
    )

    # ------------------------------------------------------------------
    # 17. PERIOD AVAILABILITY PROFILE
    # ------------------------------------------------------------------

    header("16. TREND AVAILABILITY BY PERIOD")

    period_profile = (
        saved
        .filter(~pl.col("is_aggregate"))
        .group_by(
            [
                "period",
                "year_quarter",
            ]
        )
        .agg(
            [
                pl.len().alias("rows"),

                pl.col("qoq_available")
                .sum()
                .alias("qoq_available"),

                pl.col("eu_percentile_qoq_available")
                .sum()
                .alias("eu_qoq"),

                pl.col("country_percentile_qoq_available")
                .sum()
                .alias("country_qoq"),

                pl.col("three_period_pattern_available")
                .sum()
                .alias("three_period"),
            ]
        )
        .sort("period")
    )

    print(period_profile)

    expected_period_profile = {
        202409: (116, 0, 0, 0, 0),
        202412: (119, 116, 116, 77, 0),
        202503: (111, 111, 111, 77, 111),
        202506: (119, 111, 111, 77, 111),
    }

    for period, expected_values in (
        expected_period_profile.items()
    ):

        row = period_profile.filter(
            pl.col("period") == period
        )

        actual_values = (
            row["rows"][0],
            row["qoq_available"][0],
            row["eu_qoq"][0],
            row["country_qoq"][0],
            row["three_period"][0],
        )

        check(
            actual_values == expected_values,
            (
                f"Period {period} availability profile "
                f"matches {expected_values}."
            ),
        )

    # ------------------------------------------------------------------
    # 18. AGGREGATE EXCLUSION
    # ------------------------------------------------------------------

    header("17. AGGREGATE TREND EXCLUSION")

    aggregate_saved = saved.filter(
        pl.col("is_aggregate")
    )

    check(
        aggregate_saved.height == 4,
        "Trend mart contains four aggregate observations.",
    )

    check(
        aggregate_saved.filter(
            pl.col("qoq_available")
            | pl.col("eu_percentile_qoq_available")
            | pl.col("country_percentile_qoq_available")
            | pl.col("three_period_pattern_available")
        ).height == 0,
        (
            "Aggregate observations contain no "
            "institution-level trend availability."
        ),
    )

    # ------------------------------------------------------------------
    # 19. HIGH-RATIO PRESERVATION
    # ------------------------------------------------------------------

    header("18. HIGH CET1 RATIO PRESERVATION")

    high_ratio = saved.filter(
        (~pl.col("is_aggregate"))
        & (pl.col("cet1_ratio") > 1.0)
    )

    check(
        high_ratio.height == 6,
        (
            "All six source-valid CET1 Ratio observations "
            "above 100% remain present."
        ),
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    header("CET1 TREND MART INDEPENDENT VALIDATION SUMMARY")

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
            "The CET1 trend mart independently reconciles "
            "to mart_cet1_benchmark and the approved "
            "trend methodology."
        )

        print()
        print("Validated controls include:")
        print("  - exact 469-row / 61-column structure")
        print("  - preservation of all 39 benchmark fields")
        print("  - institution-only trend population")
        print("  - genuine quarter continuity")
        print("  - QoQ CET1 changes")
        print("  - European percentile movements")
        print("  - country percentile movements")
        print("  - continuous three-quarter patterns")
        print("  - directional pattern counts")
        print("  - non-consecutive history suppression")
        print("  - period-level availability")
        print("  - aggregate exclusion")
        print("  - >100% CET1 Ratio preservation")

        print()
        print(
            "ACTION: mart_cet1_trend can be considered "
            "for approval for downstream CET1 monitoring "
            "and reporting analytics."
        )

    else:

        print("OVERALL STATUS : FAIL")

        print()
        print(
            "ACTION: Do not approve mart_cet1_trend "
            "until all failed controls are resolved."
        )

        raise SystemExit(1)

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No data was modified.")


if __name__ == "__main__":
    main()
