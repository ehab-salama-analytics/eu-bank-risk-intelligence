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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_trend.parquet"
)


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


FINAL_COLUMNS = BASE_COLUMNS + TREND_COLUMNS


def header(title: str) -> None:
    print()
    print("=" * 105)
    print(title)
    print("=" * 105)


def assert_unique_grain(
    df: pl.DataFrame,
    name: str,
) -> None:
    duplicates = (
        df.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height > 0:
        raise ValueError(
            f"{name} is not unique at lei_code x period. "
            f"Duplicate groups: {duplicates.height}"
        )


def quarter_index_expr(column: str) -> pl.Expr:
    """
    Convert YYYYMM into a sequential quarter index.

    202409 -> 2024 Q3
    202412 -> 2024 Q4
    202503 -> 2025 Q1
    202506 -> 2025 Q2
    """

    year = pl.col(column) // 100
    month = pl.col(column) % 100

    quarter = (
        pl.when(month == 3)
        .then(1)
        .when(month == 6)
        .then(2)
        .when(month == 9)
        .then(3)
        .when(month == 12)
        .then(4)
        .otherwise(None)
    )

    return (
        year * 4 + quarter
    ).cast(pl.Int64)


def main() -> None:

    header("BUILD CET1 TREND ANALYTICAL MART")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source mart  : {SOURCE_PATH}")
    print(f"Output mart  : {OUTPUT_PATH}")

    print()
    print("Approved methodology:")
    print(
        "  - institution-level trend calculations only"
    )
    print(
        "  - QoQ requires genuine quarter_gap == 1"
    )
    print(
        "  - no interpolation or synthetic observations"
    )
    print(
        "  - negative CET1 QoQ change = ratio deterioration"
    )
    print(
        "  - negative peer-percentile movement = "
        "relative peer deterioration"
    )
    print(
        "  - three-period pattern requires two "
        "consecutive quarterly transitions"
    )
    print(
        "  - no YoY metrics: current dataset has "
        "no valid 12-month pairs"
    )
    print(
        "  - no regulatory threshold or "
        "capital-strength classification"
    )

    # ------------------------------------------------------------------
    # 1. FILE AVAILABILITY
    # ------------------------------------------------------------------

    header("1. FILE AVAILABILITY")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Approved CET1 benchmark mart not found: "
            f"{SOURCE_PATH}"
        )

    print(f"[PASS] Found: {SOURCE_PATH.name}")

    # ------------------------------------------------------------------
    # 2. LOAD APPROVED BENCHMARK MART
    # ------------------------------------------------------------------

    header("2. LOAD APPROVED CET1 BENCHMARK MART")

    source = pl.read_parquet(SOURCE_PATH)

    print(
        f"[PASS] Source opened: "
        f"{source.height:,} rows x "
        f"{source.width} columns"
    )

    if source.height != 469:
        raise ValueError(
            f"Unexpected source row count: {source.height}"
        )

    if source.width != 39:
        raise ValueError(
            f"Unexpected source column count: {source.width}"
        )

    if source.columns != BASE_COLUMNS:
        raise ValueError(
            "Source CET1 benchmark schema does not match "
            "the approved 39-column contract."
        )

    assert_unique_grain(
        source,
        "Source CET1 benchmark mart",
    )

    print(
        "[PASS] Source is unique at grain: "
        "lei_code x period."
    )

    # ------------------------------------------------------------------
    # 3. SPLIT TREND POPULATION
    # ------------------------------------------------------------------

    header("3. DEFINE TREND POPULATION")

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

    print(
        f"Institution rows : {institutions.height:,}"
    )
    print(
        f"Aggregate rows   : {aggregates.height:,}"
    )

    if institutions.height != 465:
        raise ValueError(
            "Expected 465 institution-level observations."
        )

    if aggregates.height != 4:
        raise ValueError(
            "Expected four aggregate observations."
        )

    # ------------------------------------------------------------------
    # 4. CREATE PREVIOUS OBSERVATIONS
    # ------------------------------------------------------------------

    header("4. CREATE PREVIOUS OBSERVATION FIELDS")

    institutions = institutions.with_columns(
        [
            pl.col("period")
            .shift(1)
            .over("lei_code")
            .alias("previous_period"),

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

            pl.col("cet1_ratio")
            .shift(2)
            .over("lei_code")
            .alias("__cet1_ratio_two_ago"),

            pl.col("cet1_ratio_pct")
            .shift(2)
            .over("lei_code")
            .alias("__cet1_ratio_pct_two_ago"),
        ]
    )

    print("[PASS] Previous observations created.")

    # ------------------------------------------------------------------
    # 5. CALCULATE QUARTER GAPS
    # ------------------------------------------------------------------

    header("5. CALCULATE QUARTER CONTINUITY")

    institutions = institutions.with_columns(
        [
            quarter_index_expr("period")
            .alias("__current_quarter_index"),

            quarter_index_expr("previous_period")
            .alias("__previous_quarter_index"),

            quarter_index_expr("__period_two_ago")
            .alias("__two_ago_quarter_index"),
        ]
    )

    institutions = institutions.with_columns(
        [
            (
                pl.col("__current_quarter_index")
                - pl.col("__previous_quarter_index")
            )
            .alias("quarter_gap"),

            (
                pl.col("__previous_quarter_index")
                - pl.col("__two_ago_quarter_index")
            )
            .alias("__previous_quarter_gap"),
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
                    pl.col("__previous_quarter_gap") == 1
                )
            )
            .fill_null(False)
            .alias("three_period_pattern_available"),
        ]
    )

    genuine_qoq_count = institutions.filter(
        pl.col("qoq_available")
    ).height

    nonconsecutive_count = institutions.filter(
        pl.col("previous_period").is_not_null()
        & (
            pl.col("quarter_gap") != 1
        )
    ).height

    three_period_count = institutions.filter(
        pl.col("three_period_pattern_available")
    ).height

    print(
        f"Genuine consecutive-quarter pairs : "
        f"{genuine_qoq_count:,}"
    )

    print(
        f"Non-consecutive prior pairs        : "
        f"{nonconsecutive_count:,}"
    )

    print(
        f"Continuous three-quarter rows      : "
        f"{three_period_count:,}"
    )

    # ------------------------------------------------------------------
    # 6. CALCULATE QOQ CET1 MOVEMENT
    # ------------------------------------------------------------------

    header("6. CALCULATE QOQ CET1 MOVEMENT")

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

    print(
        "[PASS] QoQ CET1 changes restricted to "
        "genuine consecutive quarters."
    )

    # ------------------------------------------------------------------
    # 7. EUROPEAN PEER MOVEMENT
    # ------------------------------------------------------------------

    header("7. CALCULATE EUROPEAN PEER MOVEMENT")

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

    print(
        "EU percentile QoQ observations: "
        f"{institutions.filter(pl.col('eu_percentile_qoq_available')).height:,}"
    )

    # ------------------------------------------------------------------
    # 8. COUNTRY PEER MOVEMENT
    # ------------------------------------------------------------------

    header("8. CALCULATE COUNTRY PEER MOVEMENT")

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

    print(
        "Country percentile QoQ observations: "
        f"{institutions.filter(pl.col('country_percentile_qoq_available')).height:,}"
    )

    # ------------------------------------------------------------------
    # 9. THREE-PERIOD DIRECTIONAL PATTERN
    # ------------------------------------------------------------------

    header("9. CALCULATE THREE-PERIOD CET1 PATTERNS")

    institutions = institutions.with_columns(
        [
            (
                pl.col("previous_cet1_ratio")
                - pl.col("__cet1_ratio_two_ago")
            )
            .alias("__first_transition"),

            (
                pl.col("cet1_ratio")
                - pl.col("previous_cet1_ratio")
            )
            .alias("__second_transition"),
        ]
    )

    pattern_expr = (
        pl.when(
            (pl.col("__first_transition") < 0)
            &
            (pl.col("__second_transition") < 0)
        )
        .then(pl.lit("consecutive_decrease"))
        .when(
            (pl.col("__first_transition") > 0)
            &
            (pl.col("__second_transition") > 0)
        )
        .then(pl.lit("consecutive_increase"))
        .when(
            (pl.col("__first_transition") < 0)
            &
            (pl.col("__second_transition") > 0)
        )
        .then(pl.lit("decrease_then_increase"))
        .when(
            (pl.col("__first_transition") > 0)
            &
            (pl.col("__second_transition") < 0)
        )
        .then(pl.lit("increase_then_decrease"))
        .otherwise(pl.lit("stable_or_mixed"))
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
                pl.col("__cet1_ratio_two_ago")
            )
            .otherwise(None)
            .alias("cet1_ratio_two_quarters_ago"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                pl.col("__cet1_ratio_pct_two_ago")
            )
            .otherwise(None)
            .alias("cet1_ratio_pct_two_quarters_ago"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                pl.col("cet1_ratio")
                - pl.col("__cet1_ratio_two_ago")
            )
            .otherwise(None)
            .alias("cet1_ratio_change_3_period"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                pl.col("cet1_ratio_pct")
                - pl.col("__cet1_ratio_pct_two_ago")
            )
            .otherwise(None)
            .alias("cet1_ratio_pp_change_3_period"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(
                (
                    (pl.col("__first_transition") < 0)
                    .cast(pl.Int8)
                    +
                    (pl.col("__second_transition") < 0)
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
                    (pl.col("__first_transition") > 0)
                    .cast(pl.Int8)
                    +
                    (pl.col("__second_transition") > 0)
                    .cast(pl.Int8)
                )
            )
            .otherwise(None)
            .cast(pl.Int8)
            .alias("cet1_increase_count_3_period"),

            pl.when(
                pl.col("three_period_pattern_available")
            )
            .then(pattern_expr)
            .otherwise(None)
            .alias("cet1_three_period_pattern"),
        ]
    )

    print("Three-period pattern distribution:")

    print(
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

    # ------------------------------------------------------------------
    # 10. REMOVE TEMPORARY FIELDS
    # ------------------------------------------------------------------

    header("10. REMOVE TEMPORARY CALCULATION FIELDS")

    temporary_columns = [
        "__period_two_ago",
        "__cet1_ratio_two_ago",
        "__cet1_ratio_pct_two_ago",
        "__current_quarter_index",
        "__previous_quarter_index",
        "__two_ago_quarter_index",
        "__previous_quarter_gap",
        "__first_transition",
        "__second_transition",
    ]

    institutions = institutions.drop(
        temporary_columns
    )

    print("[PASS] Temporary calculation fields removed.")

    # ------------------------------------------------------------------
    # 11. PREPARE AGGREGATE ROWS
    # ------------------------------------------------------------------

    header("11. PREPARE AGGREGATE ROWS")

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

    print(
        "[PASS] Aggregate rows retained but excluded "
        "from institution trend calculations."
    )

    # ------------------------------------------------------------------
    # 12. BUILD FINAL MART
    # ------------------------------------------------------------------

    header("12. BUILD FINAL CET1 TREND MART")

    mart = (
        pl.concat(
            [
                institutions.select(FINAL_COLUMNS),
                aggregates.select(FINAL_COLUMNS),
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

    print(f"Rows    : {mart.height:,}")
    print(f"Columns : {mart.width}")

    # ------------------------------------------------------------------
    # 13. PRE-SAVE VALIDATION
    # ------------------------------------------------------------------

    header("13. PRE-SAVE BUSINESS VALIDATION")

    assert_unique_grain(
        mart,
        "CET1 trend mart",
    )

    print(
        "[PASS] Mart is unique at grain: "
        "lei_code x period."
    )

    if mart.height != source.height:
        raise ValueError(
            "Trend mart row count does not reconcile "
            "to CET1 benchmark mart."
        )

    if mart.width != 61:
        raise ValueError(
            f"Expected 61 columns, found {mart.width}."
        )

    print(
        "[PASS] Row count reconciles: "
        f"{mart.height:,}"
    )

    print(
        "[PASS] Trend schema contains 61 columns."
    )

    source_sorted = source.sort(
        ["lei_code", "period"]
    )

    if not mart.select(BASE_COLUMNS).equals(
        source_sorted.select(BASE_COLUMNS)
    ):
        raise ValueError(
            "One or more source benchmark fields changed "
            "during trend transformation."
        )

    print(
        "[PASS] All 39 benchmark source fields "
        "were preserved exactly."
    )

    # ------------------------------------------------------------------
    # 14. QOQ AVAILABILITY CONTROLS
    # ------------------------------------------------------------------

    header("14. QOQ AVAILABILITY CONTROLS")

    institution_check = mart.filter(
        ~pl.col("is_aggregate")
    )

    prior_count = institution_check.filter(
        pl.col("previous_period").is_not_null()
    ).height

    qoq_count = institution_check.filter(
        pl.col("qoq_available")
    ).height

    nonconsecutive_count = institution_check.filter(
        pl.col("previous_period").is_not_null()
        & (
            pl.col("quarter_gap") != 1
        )
    ).height

    eu_qoq_count = institution_check.filter(
        pl.col("eu_percentile_qoq_available")
    ).height

    country_qoq_count = institution_check.filter(
        pl.col("country_percentile_qoq_available")
    ).height

    three_period_count = institution_check.filter(
        pl.col("three_period_pattern_available")
    ).height

    print(
        f"Previous-observation pairs       : {prior_count:,}"
    )
    print(
        f"Genuine QoQ pairs                : {qoq_count:,}"
    )
    print(
        f"Non-consecutive previous pairs   : {nonconsecutive_count:,}"
    )
    print(
        f"EU percentile QoQ pairs          : {eu_qoq_count:,}"
    )
    print(
        f"Country percentile QoQ pairs     : {country_qoq_count:,}"
    )
    print(
        f"Continuous three-period rows     : {three_period_count:,}"
    )

    if prior_count != 346:
        raise ValueError(
            f"Expected 346 previous-observation pairs, "
            f"found {prior_count}."
        )

    if qoq_count != 338:
        raise ValueError(
            f"Expected 338 genuine QoQ pairs, "
            f"found {qoq_count}."
        )

    if nonconsecutive_count != 8:
        raise ValueError(
            f"Expected 8 non-consecutive pairs, "
            f"found {nonconsecutive_count}."
        )

    if eu_qoq_count != 338:
        raise ValueError(
            f"Expected 338 EU-percentile QoQ pairs, "
            f"found {eu_qoq_count}."
        )

    if country_qoq_count != 231:
        raise ValueError(
            f"Expected 231 country-percentile QoQ pairs, "
            f"found {country_qoq_count}."
        )

    if three_period_count != 222:
        raise ValueError(
            f"Expected 222 continuous three-period rows, "
            f"found {three_period_count}."
        )

    print(
        "[PASS] Trend availability reconciles exactly "
        "to readiness inspection."
    )

    invalid_qoq_values = institution_check.filter(
        ~pl.col("qoq_available")
        & (
            pl.col("cet1_ratio_change_qoq").is_not_null()
            | pl.col(
                "cet1_ratio_pp_change_qoq"
            ).is_not_null()
        )
    )

    if invalid_qoq_values.height > 0:
        raise ValueError(
            "QoQ CET1 metrics exist for observations "
            "without genuine quarterly continuity."
        )

    print(
        "[PASS] QoQ CET1 metrics are suppressed "
        "when quarter continuity is unavailable."
    )

    # ------------------------------------------------------------------
    # 15. DIRECTIONAL PROFILE
    # ------------------------------------------------------------------

    header("15. QOQ DIRECTIONAL PROFILE")

    valid_qoq = institution_check.filter(
        pl.col("qoq_available")
    )

    decreases = valid_qoq.filter(
        pl.col("cet1_ratio_change_qoq") < 0
    ).height

    increases = valid_qoq.filter(
        pl.col("cet1_ratio_change_qoq") > 0
    ).height

    unchanged = valid_qoq.filter(
        pl.col("cet1_ratio_change_qoq") == 0
    ).height

    print(f"CET1 decreases : {decreases:,}")
    print(f"CET1 increases : {increases:,}")
    print(f"Unchanged      : {unchanged:,}")

    if decreases != 152:
        raise ValueError(
            f"Expected 152 CET1 decreases, found {decreases}."
        )

    if increases != 186:
        raise ValueError(
            f"Expected 186 CET1 increases, found {increases}."
        )

    if unchanged != 0:
        raise ValueError(
            f"Expected 0 unchanged QoQ observations, "
            f"found {unchanged}."
        )

    print(
        "[PASS] QoQ CET1 direction profile reconciles "
        "to readiness inspection."
    )

    # ------------------------------------------------------------------
    # 16. PERIOD AVAILABILITY PROFILE
    # ------------------------------------------------------------------

    header("16. TREND AVAILABILITY BY PERIOD")

    availability_profile = (
        institution_check
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
                .alias("eu_percentile_qoq_available"),

                pl.col(
                    "country_percentile_qoq_available"
                )
                .sum()
                .alias(
                    "country_percentile_qoq_available"
                ),

                pl.col("three_period_pattern_available")
                .sum()
                .alias(
                    "three_period_pattern_available"
                ),
            ]
        )
        .sort("period")
    )

    print(availability_profile)

    expected_period_counts = {
        202409: (0, 0, 0, 0),
        202412: (116, 116, 77, 0),
        202503: (111, 111, 77, 111),
        202506: (111, 111, 77, 111),
    }

    for period, expected in expected_period_counts.items():

        row = availability_profile.filter(
            pl.col("period") == period
        )

        if row.height != 1:
            raise ValueError(
                f"Period {period} not found exactly once."
            )

        actual = (
            row["qoq_available"][0],
            row["eu_percentile_qoq_available"][0],
            row[
                "country_percentile_qoq_available"
            ][0],
            row[
                "three_period_pattern_available"
            ][0],
        )

        if actual != expected:
            raise ValueError(
                f"Trend availability mismatch for "
                f"{period}. Expected {expected}, "
                f"found {actual}."
            )

    print(
        "[PASS] Period-level trend availability "
        "matches readiness inspection."
    )

    # ------------------------------------------------------------------
    # 17. THREE-PERIOD BUSINESS RULES
    # ------------------------------------------------------------------

    header("17. THREE-PERIOD PATTERN CONTROLS")

    pattern_rows = institution_check.filter(
        pl.col("three_period_pattern_available")
    )

    invalid_pattern_rows = pattern_rows.filter(
        pl.col("cet1_three_period_pattern").is_null()
        | pl.col("period_two_quarters_ago").is_null()
        | pl.col(
            "cet1_ratio_two_quarters_ago"
        ).is_null()
    )

    if invalid_pattern_rows.height > 0:
        raise ValueError(
            "Continuous three-period observations contain "
            "missing pattern fields."
        )

    print(
        "[PASS] Every continuous three-quarter observation "
        "has complete pattern fields."
    )

    nonpattern_rows = institution_check.filter(
        ~pl.col("three_period_pattern_available")
    )

    invalid_nonpattern = nonpattern_rows.filter(
        pl.col("cet1_three_period_pattern").is_not_null()
        | pl.col("cet1_ratio_change_3_period").is_not_null()
        | pl.col(
            "cet1_ratio_pp_change_3_period"
        ).is_not_null()
    )

    if invalid_nonpattern.height > 0:
        raise ValueError(
            "Three-period metrics exist where continuous "
            "three-quarter history is unavailable."
        )

    print(
        "[PASS] Three-period metrics are suppressed "
        "without continuous history."
    )

    print()
    print("Pattern distribution:")

    print(
        pattern_rows
        .group_by("cet1_three_period_pattern")
        .agg(
            pl.len().alias("rows")
        )
        .sort("cet1_three_period_pattern")
    )

    # ------------------------------------------------------------------
    # 18. AGGREGATE EXCLUSION
    # ------------------------------------------------------------------

    header("18. AGGREGATE TREND EXCLUSION")

    aggregate_check = mart.filter(
        pl.col("is_aggregate")
    )

    invalid_aggregates = aggregate_check.filter(
        pl.col("previous_period").is_not_null()
        | pl.col("qoq_available")
        | pl.col("eu_percentile_qoq_available")
        | pl.col("country_percentile_qoq_available")
        | pl.col("three_period_pattern_available")
        | pl.col("cet1_ratio_change_qoq").is_not_null()
        | pl.col("cet1_three_period_pattern").is_not_null()
    )

    if invalid_aggregates.height > 0:
        raise ValueError(
            "Aggregate rows incorrectly contain "
            "institution-level trend calculations."
        )

    print(
        "[PASS] Aggregate rows are excluded from "
        "institution-level trend calculations."
    )

    # ------------------------------------------------------------------
    # 19. HIGH-RATIO PRESERVATION
    # ------------------------------------------------------------------

    header("19. HIGH CET1 RATIO PRESERVATION")

    high_ratio_rows = mart.filter(
        (~pl.col("is_aggregate"))
        & (pl.col("cet1_ratio") > 1.0)
    )

    print(
        f"CET1 Ratio observations >100%: "
        f"{high_ratio_rows.height:,}"
    )

    if high_ratio_rows.height != 6:
        raise ValueError(
            "Expected six source-valid CET1 Ratio "
            "observations above 100%."
        )

    print(
        "[PASS] All six source-valid high CET1 "
        "observations were retained."
    )

    # ------------------------------------------------------------------
    # 20. SAVE MART
    # ------------------------------------------------------------------

    header("20. SAVE CET1 TREND MART")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mart.write_parquet(OUTPUT_PATH)

    print(f"[PASS] Saved: {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 21. POST-SAVE VALIDATION
    # ------------------------------------------------------------------

    header("21. POST-SAVE VALIDATION")

    saved = pl.read_parquet(OUTPUT_PATH)

    if saved.shape != mart.shape:
        raise ValueError(
            "Saved trend mart shape does not match "
            "the in-memory mart."
        )

    print(
        f"[PASS] Saved mart reopened: "
        f"{saved.height:,} rows x "
        f"{saved.width} columns"
    )

    assert_unique_grain(
        saved,
        "Saved CET1 trend mart",
    )

    print(
        "[PASS] Saved trend mart remains unique "
        "at lei_code x period."
    )

    if saved.columns != FINAL_COLUMNS:
        raise ValueError(
            "Saved trend mart column order does not "
            "match the approved schema."
        )

    print(
        "[PASS] Saved trend column order reconciles."
    )

    if not saved.equals(mart):
        raise ValueError(
            "Saved Parquet does not exactly match "
            "the in-memory CET1 trend mart."
        )

    print(
        "[PASS] Saved Parquet exactly matches "
        "the in-memory trend mart."
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    header("CET1 TREND MART BUILD COMPLETE")

    print(f"Output  : {OUTPUT_PATH}")
    print(f"Rows    : {saved.height:,}")
    print(f"Columns : {saved.width}")

    print()
    print("Validated trend availability:")
    print("  Genuine QoQ pairs               : 338")
    print("  EU-percentile QoQ pairs         : 338")
    print("  Country-percentile QoQ pairs    : 231")
    print("  Continuous three-quarter rows   : 222")
    print("  Non-consecutive prior pairs     : 8")

    print()
    print(
        "CET1 directional interpretation:"
    )
    print(
        "  negative QoQ CET1 change -> ratio deterioration"
    )
    print(
        "  positive QoQ CET1 change -> ratio improvement"
    )
    print(
        "  negative percentile change -> relative peer deterioration"
    )

    print()
    print(
        "No YoY metrics, regulatory capital thresholds, "
        "risk flags, or capital-strength classifications "
        "were introduced."
    )


if __name__ == "__main__":
    main()
