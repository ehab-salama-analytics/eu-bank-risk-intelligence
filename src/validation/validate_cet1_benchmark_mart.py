from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_ratio.parquet"
)

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_benchmark.parquet"
)


MIN_FORMAL_COUNTRY_PEERS = 5
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
]


BENCHMARK_COLUMNS = [
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


EXPECTED_COLUMNS = BASE_COLUMNS + BENCHMARK_COLUMNS


FLOAT_COLUMNS = {
    "cet1_capital",
    "total_risk_exposure_amount",
    "cet1_ratio",
    "cet1_ratio_pct",
    "reconstructed_cet1_ratio",
    "reconciliation_difference",
    "reconciliation_difference_abs",
    "eu_mean_cet1_ratio",
    "eu_median_cet1_ratio",
    "eu_rank",
    "eu_percentile",
    "eu_percentile_pct",
    "country_mean_cet1_ratio",
    "country_median_cet1_ratio",
    "country_rank",
    "country_percentile",
    "country_percentile_pct",
    "eu_deviation_from_median",
    "country_deviation_from_median",
}


EXPECTED_DTYPES = {
    "benchmark_eligible": pl.Boolean,
    "eu_peer_count": pl.UInt32,
    "eu_mean_cet1_ratio": pl.Float64,
    "eu_median_cet1_ratio": pl.Float64,
    "eu_rank": pl.Float64,
    "eu_percentile": pl.Float64,
    "eu_percentile_pct": pl.Float64,
    "country_peer_count": pl.UInt32,
    "country_descriptive_available": pl.Boolean,
    "country_benchmark_available": pl.Boolean,
    "country_benchmark_status": pl.String,
    "country_mean_cet1_ratio": pl.Float64,
    "country_median_cet1_ratio": pl.Float64,
    "country_rank": pl.Float64,
    "country_percentile": pl.Float64,
    "country_percentile_pct": pl.Float64,
    "eu_deviation_from_median": pl.Float64,
    "country_deviation_from_median": pl.Float64,
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
                pl.col(diff_col)
                > FLOAT_TOLERANCE
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

    header("CET1 BENCHMARK MART INDEPENDENT VALIDATION")

    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Source mart    : {SOURCE_PATH}")
    print(f"Benchmark mart : {BENCHMARK_PATH}")
    print()
    print("VALIDATION MODE: READ ONLY")
    print("No data will be modified.")

    # ------------------------------------------------------------------
    # 1. FILE AVAILABILITY
    # ------------------------------------------------------------------

    header("1. FILE AVAILABILITY")

    for path in [SOURCE_PATH, BENCHMARK_PATH]:

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
    saved = pl.read_parquet(BENCHMARK_PATH)

    check(
        source.height == 469,
        f"Source CET1 Ratio mart contains 469 rows.",
    )

    check(
        saved.height == 469,
        f"Saved CET1 benchmark mart contains 469 rows.",
    )

    check(
        saved.width == 39,
        f"Saved CET1 benchmark mart contains 39 columns.",
    )

    assert_unique_grain(
        source,
        "Source CET1 Ratio mart",
    )

    assert_unique_grain(
        saved,
        "Saved CET1 benchmark mart",
    )

    # ------------------------------------------------------------------
    # 3. SCHEMA VALIDATION
    # ------------------------------------------------------------------

    header("3. BENCHMARK MART SCHEMA VALIDATION")

    check(
        saved.columns == EXPECTED_COLUMNS,
        (
            "Benchmark mart column names and order "
            "match the approved 39-column schema."
        ),
    )

    for column, expected_dtype in EXPECTED_DTYPES.items():

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

    header("4. SOURCE CET1 FIELD PRESERVATION")

    source_prefixed = source.rename(
        {
            column: f"expected_{column}"
            for column in source.columns
            if column not in ["lei_code", "period"]
        }
    )

    saved_prefixed = saved.rename(
        {
            column: f"saved_{column}"
            for column in saved.columns
            if column not in ["lei_code", "period"]
        }
    )

    source_alignment = source_prefixed.join(
        saved_prefixed,
        on=["lei_code", "period"],
        how="inner",
    )

    check(
        source_alignment.height == 469,
        (
            "All source CET1 observations align one-to-one "
            "with the benchmark mart."
        ),
    )

    for column in BASE_COLUMNS:

        if column in ["lei_code", "period"]:
            continue

        expected_col = f"expected_{column}"
        saved_col = f"saved_{column}"

        if column in FLOAT_COLUMNS:
            compare_float(
                source_alignment,
                expected_col,
                saved_col,
                f"Source field '{column}'",
            )
        else:
            compare_exact(
                source_alignment,
                expected_col,
                saved_col,
                f"Source field '{column}'",
            )

    # ------------------------------------------------------------------
    # 5. RECONSTRUCT ELIGIBLE POPULATION
    # ------------------------------------------------------------------

    header("5. INDEPENDENT BENCHMARK POPULATION")

    working = source.with_columns(
        (~pl.col("is_aggregate"))
        .alias("benchmark_eligible")
    )

    eligible = working.filter(
        pl.col("benchmark_eligible")
    )

    aggregate = working.filter(
        ~pl.col("benchmark_eligible")
    )

    check(
        eligible.height == 465,
        (
            "Independent benchmark population contains "
            "465 institution observations."
        ),
    )

    check(
        aggregate.height == 4,
        (
            "Independent reconstruction contains "
            "four aggregate observations."
        ),
    )

    # ------------------------------------------------------------------
    # 6. EUROPEAN BENCHMARK RECONSTRUCTION
    # ------------------------------------------------------------------

    header("6. EUROPEAN BENCHMARK RECONSTRUCTION")

    eu_stats = (
        eligible
        .group_by("period")
        .agg(
            [
                pl.col("lei_code")
                .n_unique()
                .alias("eu_peer_count"),

                pl.col("cet1_ratio")
                .mean()
                .alias("eu_mean_cet1_ratio"),

                pl.col("cet1_ratio")
                .median()
                .alias("eu_median_cet1_ratio"),
            ]
        )
    )

    eligible = (
        eligible
        .join(
            eu_stats,
            on="period",
            how="left",
        )
        .with_columns(
            pl.col("cet1_ratio")
            .rank(
                method="average",
                descending=False,
            )
            .over("period")
            .alias("eu_rank")
        )
        .with_columns(
            (
                (
                    pl.col("eu_rank") - 1.0
                )
                /
                (
                    pl.col("eu_peer_count") - 1
                )
            )
            .alias("eu_percentile")
        )
        .with_columns(
            [
                (
                    pl.col("eu_percentile") * 100
                ).alias("eu_percentile_pct"),

                (
                    pl.col("cet1_ratio")
                    - pl.col("eu_median_cet1_ratio")
                )
                .alias("eu_deviation_from_median"),
            ]
        )
    )

    print(eu_stats.sort("period"))

    expected_peer_counts = {
        202409: 116,
        202412: 119,
        202503: 111,
        202506: 119,
    }

    for period, expected_count in (
        expected_peer_counts.items()
    ):

        actual = (
            eu_stats
            .filter(pl.col("period") == period)
            ["eu_peer_count"][0]
        )

        check(
            actual == expected_count,
            (
                f"European peer count for {period} "
                f"is {expected_count}."
            ),
        )

    # ------------------------------------------------------------------
    # 7. AVERAGE-RANK TIE CHECK
    # ------------------------------------------------------------------

    header("7. EUROPEAN TIE PROFILE")

    eu_ties = (
        eligible
        .group_by(
            [
                "period",
                "cet1_ratio",
            ]
        )
        .agg(
            pl.len().alias("tie_size")
        )
        .filter(
            pl.col("tie_size") > 1
        )
        .sort(
            [
                "period",
                "cet1_ratio",
            ]
        )
    )

    print(
        f"European tied CET1 value groups: "
        f"{eu_ties.height:,}"
    )

    if eu_ties.height > 0:
        print(eu_ties)

    print(
        "Average ranks are independently reconstructed "
        "using Polars rank(method='average')."
    )

    # ------------------------------------------------------------------
    # 8. COUNTRY STATISTICS
    # ------------------------------------------------------------------

    header("8. COUNTRY BENCHMARK RECONSTRUCTION")

    country_stats = (
        eligible
        .group_by(
            [
                "period",
                "institution_country_code",
                "institution_country",
            ]
        )
        .agg(
            [
                pl.col("lei_code")
                .n_unique()
                .alias("country_peer_count"),

                pl.col("cet1_ratio")
                .mean()
                .alias("country_mean_cet1_ratio"),

                pl.col("cet1_ratio")
                .median()
                .alias("country_median_cet1_ratio"),
            ]
        )
    )

    eligible = eligible.join(
        country_stats,
        on=[
            "period",
            "institution_country_code",
            "institution_country",
        ],
        how="left",
    )

    eligible = eligible.with_columns(
        [
            (
                pl.col("country_peer_count") >= 2
            )
            .alias("country_descriptive_available"),

            (
                pl.col("country_peer_count")
                >= MIN_FORMAL_COUNTRY_PEERS
            )
            .alias("country_benchmark_available"),

            pl.when(
                pl.col("country_peer_count")
                >= MIN_FORMAL_COUNTRY_PEERS
            )
            .then(
                pl.lit("formal_benchmark_available")
            )
            .when(
                pl.col("country_peer_count")
                .is_between(
                    2,
                    4,
                    closed="both",
                )
            )
            .then(
                pl.lit("descriptive_only")
            )
            .otherwise(
                pl.lit("unavailable_singleton")
            )
            .alias("country_benchmark_status"),
        ]
    )

    # Suppress descriptive statistics for singleton groups.
    eligible = eligible.with_columns(
        [
            pl.when(
                pl.col("country_descriptive_available")
            )
            .then(
                pl.col("country_mean_cet1_ratio")
            )
            .otherwise(None)
            .alias("__country_mean"),

            pl.when(
                pl.col("country_descriptive_available")
            )
            .then(
                pl.col("country_median_cet1_ratio")
            )
            .otherwise(None)
            .alias("__country_median"),
        ]
    ).drop(
        [
            "country_mean_cet1_ratio",
            "country_median_cet1_ratio",
        ]
    ).rename(
        {
            "__country_mean":
                "country_mean_cet1_ratio",
            "__country_median":
                "country_median_cet1_ratio",
        }
    )

    # ------------------------------------------------------------------
    # 9. COUNTRY RANK / PERCENTILE
    # ------------------------------------------------------------------

    header("9. COUNTRY RANK AND PERCENTILE RECONSTRUCTION")

    eligible = eligible.with_columns(
        pl.col("cet1_ratio")
        .rank(
            method="average",
            descending=False,
        )
        .over(
            [
                "period",
                "institution_country_code",
            ]
        )
        .alias("__country_rank_raw")
    )

    eligible = eligible.with_columns(
        pl.when(
            pl.col("country_benchmark_available")
        )
        .then(
            pl.col("__country_rank_raw")
        )
        .otherwise(None)
        .alias("country_rank")
    ).drop("__country_rank_raw")

    eligible = eligible.with_columns(
        [
            pl.when(
                pl.col("country_benchmark_available")
            )
            .then(
                (
                    pl.col("country_rank") - 1.0
                )
                /
                (
                    pl.col("country_peer_count") - 1
                )
            )
            .otherwise(None)
            .alias("country_percentile"),
        ]
    )

    eligible = eligible.with_columns(
        [
            (
                pl.col("country_percentile") * 100
            )
            .alias("country_percentile_pct"),

            pl.when(
                pl.col("country_descriptive_available")
            )
            .then(
                pl.col("cet1_ratio")
                - pl.col("country_median_cet1_ratio")
            )
            .otherwise(None)
            .alias("country_deviation_from_median"),
        ]
    )

    # ------------------------------------------------------------------
    # 10. AGGREGATE RECONSTRUCTION
    # ------------------------------------------------------------------

    header("10. AGGREGATE BENCHMARK TREATMENT")

    aggregate = aggregate.with_columns(
        [
            pl.lit(None)
            .cast(pl.UInt32)
            .alias("eu_peer_count"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("eu_mean_cet1_ratio"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("eu_median_cet1_ratio"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("eu_rank"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("eu_percentile"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("eu_percentile_pct"),

            pl.lit(None)
            .cast(pl.UInt32)
            .alias("country_peer_count"),

            pl.lit(False)
            .alias("country_descriptive_available"),

            pl.lit(False)
            .alias("country_benchmark_available"),

            pl.lit("not_eligible_aggregate")
            .alias("country_benchmark_status"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("country_mean_cet1_ratio"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("country_median_cet1_ratio"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("country_rank"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("country_percentile"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("country_percentile_pct"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("eu_deviation_from_median"),

            pl.lit(None)
            .cast(pl.Float64)
            .alias("country_deviation_from_median"),
        ]
    )

    # ------------------------------------------------------------------
    # 11. BUILD INDEPENDENT EXPECTED MART
    # ------------------------------------------------------------------

    header("11. BUILD INDEPENDENT EXPECTED BENCHMARK MART")

    expected = (
        pl.concat(
            [
                eligible.select(EXPECTED_COLUMNS),
                aggregate.select(EXPECTED_COLUMNS),
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
            "Independent CET1 benchmark reconstruction "
            "contains 469 rows."
        ),
    )

    check(
        expected.width == 39,
        (
            "Independent CET1 benchmark reconstruction "
            "contains 39 columns."
        ),
    )

    assert_unique_grain(
        expected,
        "Independent CET1 benchmark reconstruction",
    )

    # ------------------------------------------------------------------
    # 12. ALIGN EXPECTED AND SAVED
    # ------------------------------------------------------------------

    header("12. EXPECTED-TO-SAVED ALIGNMENT")

    expected_prefixed = expected.rename(
        {
            column: f"expected_{column}"
            for column in expected.columns
            if column not in ["lei_code", "period"]
        }
    )

    saved_prefixed = saved.rename(
        {
            column: f"saved_{column}"
            for column in saved.columns
            if column not in ["lei_code", "period"]
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
            "Saved benchmark observations align one-to-one "
            "with independent reconstruction."
        ),
    )

    # ------------------------------------------------------------------
    # 13. BENCHMARK FIELD RECONCILIATION
    # ------------------------------------------------------------------

    header("13. BENCHMARK FIELD RECONCILIATION")

    for column in BENCHMARK_COLUMNS:

        expected_col = f"expected_{column}"
        saved_col = f"saved_{column}"

        if column in FLOAT_COLUMNS:
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
    # 14. COUNTRY STATUS PROFILE
    # ------------------------------------------------------------------

    header("14. COUNTRY BENCHMARK STATUS PROFILE")

    status_profile = (
        saved
        .group_by("country_benchmark_status")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("lei_code")
                .n_unique()
                .alias("entities"),
            ]
        )
        .sort("country_benchmark_status")
    )

    print(status_profile)

    expected_status_counts = {
        "formal_benchmark_available": 322,
        "descriptive_only": 131,
        "unavailable_singleton": 12,
        "not_eligible_aggregate": 4,
    }

    for status, expected_count in (
        expected_status_counts.items()
    ):

        actual = saved.filter(
            pl.col("country_benchmark_status")
            == status
        ).height

        check(
            actual == expected_count,
            (
                f"{status} contains "
                f"{expected_count} rows."
            ),
        )

    # ------------------------------------------------------------------
    # 15. BUSINESS RULE CONTROLS
    # ------------------------------------------------------------------

    header("15. BENCHMARK BUSINESS RULE CONTROLS")

    eligible_saved = saved.filter(
        pl.col("benchmark_eligible")
    )

    aggregate_saved = saved.filter(
        ~pl.col("benchmark_eligible")
    )

    check(
        eligible_saved.filter(
            pl.col("eu_rank").is_null()
            | pl.col("eu_percentile").is_null()
        ).height == 0,
        (
            "All benchmark-eligible observations "
            "have European rank and percentile."
        ),
    )

    check(
        aggregate_saved.filter(
            pl.col("eu_rank").is_not_null()
            | pl.col("eu_percentile").is_not_null()
        ).height == 0,
        (
            "Aggregate observations have no "
            "European rank or percentile."
        ),
    )

    formal = eligible_saved.filter(
        pl.col("country_benchmark_available")
    )

    check(
        formal.filter(
            pl.col("country_peer_count")
            < MIN_FORMAL_COUNTRY_PEERS
        ).height == 0,
        (
            "Formal country benchmarks contain "
            "at least five peers."
        ),
    )

    check(
        formal.filter(
            pl.col("country_rank").is_null()
            | pl.col("country_percentile").is_null()
        ).height == 0,
        (
            "All formal country benchmark observations "
            "contain rank and percentile."
        ),
    )

    nonformal = eligible_saved.filter(
        ~pl.col("country_benchmark_available")
    )

    check(
        nonformal.filter(
            pl.col("country_rank").is_not_null()
            | pl.col("country_percentile").is_not_null()
        ).height == 0,
        (
            "Country rank and percentile are suppressed "
            "where formal benchmarking is unavailable."
        ),
    )

    singleton = eligible_saved.filter(
        pl.col("country_peer_count") == 1
    )

    check(
        singleton.filter(
            pl.col("country_mean_cet1_ratio").is_not_null()
            | pl.col("country_median_cet1_ratio").is_not_null()
        ).height == 0,
        (
            "Singleton country groups suppress "
            "descriptive statistics."
        ),
    )

    # ------------------------------------------------------------------
    # 16. PERCENTILE RANGE CONTROLS
    # ------------------------------------------------------------------

    header("16. PERCENTILE RANGE CONTROLS")

    invalid_eu_percentiles = eligible_saved.filter(
        (pl.col("eu_percentile") < 0)
        | (pl.col("eu_percentile") > 1)
    )

    check(
        invalid_eu_percentiles.height == 0,
        "European percentiles remain within [0, 1].",
    )

    invalid_country_percentiles = formal.filter(
        (pl.col("country_percentile") < 0)
        | (pl.col("country_percentile") > 1)
    )

    check(
        invalid_country_percentiles.height == 0,
        "Formal country percentiles remain within [0, 1].",
    )

    # ------------------------------------------------------------------
    # 17. HIGH CET1 RATIO PRESERVATION
    # ------------------------------------------------------------------

    header("17. HIGH CET1 RATIO PRESERVATION")

    source_high = source.filter(
        (~pl.col("is_aggregate"))
        & (pl.col("cet1_ratio") > 1.0)
    )

    saved_high = saved.filter(
        pl.col("benchmark_eligible")
        & (pl.col("cet1_ratio") > 1.0)
    )

    print(
        f"Source high-ratio observations : "
        f"{source_high.height:,}"
    )

    print(
        f"Saved high-ratio observations  : "
        f"{saved_high.height:,}"
    )

    check(
        source_high.height == 6,
        (
            "Source contains the expected six "
            "institution CET1 Ratio observations >100%."
        ),
    )

    check(
        saved_high.height == 6,
        (
            "Benchmark mart preserves all six "
            "CET1 Ratio observations >100%."
        ),
    )

    source_high_keys = set(
        source_high
        .select(["lei_code", "period"])
        .iter_rows()
    )

    saved_high_keys = set(
        saved_high
        .select(["lei_code", "period"])
        .iter_rows()
    )

    check(
        source_high_keys == saved_high_keys,
        (
            "High CET1 Ratio entity-period observations "
            "match the source exactly."
        ),
    )

    # ------------------------------------------------------------------
    # 18. EUROPEAN PERIOD PROFILE
    # ------------------------------------------------------------------

    header("18. EUROPEAN BENCHMARK PERIOD PROFILE")

    period_profile = (
        saved
        .filter(
            pl.col("benchmark_eligible")
        )
        .group_by("period")
        .agg(
            [
                pl.col("eu_peer_count")
                .first()
                .alias("peer_count"),

                pl.col("eu_mean_cet1_ratio")
                .first()
                .alias("mean_cet1_ratio"),

                pl.col("eu_median_cet1_ratio")
                .first()
                .alias("median_cet1_ratio"),

                pl.col("eu_percentile")
                .min()
                .alias("minimum_percentile"),

                pl.col("eu_percentile")
                .max()
                .alias("maximum_percentile"),
            ]
        )
        .sort("period")
    )

    print(period_profile)

    # ------------------------------------------------------------------
    # 19. FINAL SUMMARY
    # ------------------------------------------------------------------

    header("CET1 BENCHMARK MART INDEPENDENT VALIDATION SUMMARY")

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
            "The CET1 benchmark mart independently "
            "reconciles to mart_cet1_ratio and the "
            "approved benchmark methodology."
        )

        print()
        print("Validated controls include:")
        print("  - exact 469-row / 39-column structure")
        print("  - source CET1 field preservation")
        print("  - benchmark eligibility")
        print("  - European peer counts")
        print("  - European mean / median")
        print("  - average-rank methodology")
        print("  - European percentiles")
        print("  - country peer counts")
        print("  - country benchmark availability")
        print("  - country mean / median suppression")
        print("  - country ranks / percentiles")
        print("  - aggregate exclusion")
        print("  - percentile range controls")
        print("  - >100% CET1 Ratio preservation")

        print()
        print(
            "ACTION: mart_cet1_benchmark can be "
            "considered for approval for downstream "
            "CET1 trend and reporting analytics."
        )

    else:

        print("OVERALL STATUS : FAIL")

        print()
        print(
            "ACTION: Do not approve mart_cet1_benchmark "
            "until failed controls are resolved."
        )

        raise SystemExit(1)

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No data was modified.")


if __name__ == "__main__":
    main()
