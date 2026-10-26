from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_MART_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_benchmark.parquet"
)

TREND_MART_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_trend.parquet"
)


# =============================================================================
# VALIDATION CONTRACT
# =============================================================================

KEY_COLUMNS = [
    "lei_code",
    "period",
]

SOURCE_REQUIRED_COLUMNS = [
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
    "npe_ratio",
    "npe_ratio_pct",
    "eu_percentile",
    "country_percentile",
]

TREND_COLUMNS = [
    "previous_period",
    "quarter_gap",
    "qoq_available",
    "previous_npe_ratio",
    "previous_npe_ratio_pct",
    "npe_ratio_change_qoq",
    "npe_ratio_pp_change_qoq",
    "eu_percentile_qoq_available",
    "previous_eu_percentile",
    "eu_percentile_change_qoq",
    "country_percentile_qoq_available",
    "previous_country_percentile",
    "country_percentile_change_qoq",
    "three_period_pattern_available",
    "period_two_quarters_ago",
    "npe_ratio_two_quarters_ago",
    "npe_ratio_pct_two_quarters_ago",
    "npe_ratio_change_3_period",
    "npe_ratio_pp_change_3_period",
    "npe_increase_count_3_period",
    "npe_decrease_count_3_period",
    "npe_three_period_pattern",
]

EXPECTED_TREND_DTYPES = {
    "previous_period": pl.Int64,
    "quarter_gap": pl.Int64,
    "qoq_available": pl.Boolean,
    "previous_npe_ratio": pl.Float64,
    "previous_npe_ratio_pct": pl.Float64,
    "npe_ratio_change_qoq": pl.Float64,
    "npe_ratio_pp_change_qoq": pl.Float64,
    "eu_percentile_qoq_available": pl.Boolean,
    "previous_eu_percentile": pl.Float64,
    "eu_percentile_change_qoq": pl.Float64,
    "country_percentile_qoq_available": pl.Boolean,
    "previous_country_percentile": pl.Float64,
    "country_percentile_change_qoq": pl.Float64,
    "three_period_pattern_available": pl.Boolean,
    "period_two_quarters_ago": pl.Int64,
    "npe_ratio_two_quarters_ago": pl.Float64,
    "npe_ratio_pct_two_quarters_ago": pl.Float64,
    "npe_ratio_change_3_period": pl.Float64,
    "npe_ratio_pp_change_3_period": pl.Float64,
    "npe_increase_count_3_period": pl.Int8,
    "npe_decrease_count_3_period": pl.Int8,
    "npe_three_period_pattern": pl.String,
}

VALID_THREE_PERIOD_PATTERNS = {
    "consecutive_increase",
    "consecutive_decrease",
    "increase_then_decrease",
    "decrease_then_increase",
    "stable_or_mixed",
}

FLOAT_TOLERANCE = 1e-12


# =============================================================================
# DISPLAY / VALIDATION HELPERS
# =============================================================================

def print_header(title: str) -> None:
    line = "=" * 100
    print(f"\n{line}")
    print(title)
    print(line)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def require(condition: bool, pass_message: str, fail_message: str) -> None:
    if condition:
        print(f"[PASS] {pass_message}")
    else:
        fail(fail_message)


def count_true(df: pl.DataFrame, column: str) -> int:
    value = df.select(
        pl.col(column)
        .fill_null(False)
        .cast(pl.Int64)
        .sum()
        .alias("count")
    ).item()

    return int(value or 0)


def values_equal(
    actual: list[Any],
    expected: list[Any],
    *,
    tolerance: float = FLOAT_TOLERANCE,
) -> tuple[bool, int, float]:
    """
    Compare two Python lists with explicit NULL and Float64 handling.

    Returns:
        equal,
        mismatch_count,
        maximum_absolute_difference
    """

    if len(actual) != len(expected):
        return False, abs(len(actual) - len(expected)), math.inf

    mismatch_count = 0
    maximum_absolute_difference = 0.0

    for actual_value, expected_value in zip(actual, expected):
        if actual_value is None and expected_value is None:
            continue

        if actual_value is None or expected_value is None:
            mismatch_count += 1
            continue

        if isinstance(actual_value, float) or isinstance(expected_value, float):
            actual_float = float(actual_value)
            expected_float = float(expected_value)

            if math.isnan(actual_float) and math.isnan(expected_float):
                continue

            if math.isnan(actual_float) != math.isnan(expected_float):
                mismatch_count += 1
                continue

            difference = abs(actual_float - expected_float)
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                difference,
            )

            if difference > tolerance:
                mismatch_count += 1

        else:
            if actual_value != expected_value:
                mismatch_count += 1

    return (
        mismatch_count == 0,
        mismatch_count,
        maximum_absolute_difference,
    )


# =============================================================================
# INDEPENDENT TREND RECONSTRUCTION
# =============================================================================

def reconstruct_expected_trends(
    source: pl.DataFrame,
) -> pl.DataFrame:
    """
    Reconstruct trend fields independently from mart_npe_benchmark.

    Important methodology:
    - observations are ordered independently within LEI;
    - previous available row is not automatically considered previous quarter;
    - QoQ requires quarter_gap == 1;
    - aggregate analytical entities do not receive institution trend metrics;
    - no missing periods are interpolated;
    - no synthetic observations are created;
    - three-period patterns require two consecutive quarter transitions.
    """

    history = (
        source
        .sort(["lei_code", "period"])
        .with_columns(
            (
                pl.col("year").cast(pl.Int64) * 4
                + pl.col("quarter").cast(pl.Int64)
            ).alias("_quarter_index")
        )
    )

    # -------------------------------------------------------------------------
    # Previous and two-observation lags
    # -------------------------------------------------------------------------

    history = history.with_columns(
        [
            pl.col("period")
            .shift(1)
            .over("lei_code")
            .alias("previous_period"),

            pl.col("_quarter_index")
            .shift(1)
            .over("lei_code")
            .alias("_previous_quarter_index"),

            pl.col("npe_ratio")
            .shift(1)
            .over("lei_code")
            .alias("previous_npe_ratio"),

            pl.col("npe_ratio_pct")
            .shift(1)
            .over("lei_code")
            .alias("previous_npe_ratio_pct"),

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
            .alias("period_two_quarters_ago"),

            pl.col("npe_ratio")
            .shift(2)
            .over("lei_code")
            .alias("npe_ratio_two_quarters_ago"),

            pl.col("npe_ratio_pct")
            .shift(2)
            .over("lei_code")
            .alias("npe_ratio_pct_two_quarters_ago"),
        ]
    )

    # -------------------------------------------------------------------------
    # Quarter-gap calculation
    # -------------------------------------------------------------------------

    history = history.with_columns(
        (
            pl.col("_quarter_index")
            - pl.col("_previous_quarter_index")
        )
        .cast(pl.Int64)
        .alias("quarter_gap")
    )

    history = history.with_columns(
        pl.col("quarter_gap")
        .shift(1)
        .over("lei_code")
        .alias("_previous_quarter_gap")
    )

    # -------------------------------------------------------------------------
    # Genuine quarter-on-quarter availability
    # -------------------------------------------------------------------------

    history = history.with_columns(
        (
            (~pl.col("is_aggregate"))
            & (pl.col("quarter_gap") == 1)
        )
        .fill_null(False)
        .alias("qoq_available")
    )

    # -------------------------------------------------------------------------
    # QoQ NPE measures
    # -------------------------------------------------------------------------

    history = history.with_columns(
        [
            pl.when(pl.col("qoq_available"))
            .then(
                pl.col("npe_ratio")
                - pl.col("previous_npe_ratio")
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("npe_ratio_change_qoq"),

            pl.when(pl.col("qoq_available"))
            .then(
                pl.col("npe_ratio_pct")
                - pl.col("previous_npe_ratio_pct")
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("npe_ratio_pp_change_qoq"),
        ]
    )

    # -------------------------------------------------------------------------
    # European percentile movement
    # -------------------------------------------------------------------------

    history = history.with_columns(
        (
            pl.col("qoq_available")
            & pl.col("eu_percentile").is_not_null()
            & pl.col("previous_eu_percentile").is_not_null()
        )
        .fill_null(False)
        .alias("eu_percentile_qoq_available")
    )

    history = history.with_columns(
        pl.when(pl.col("eu_percentile_qoq_available"))
        .then(
            pl.col("eu_percentile")
            - pl.col("previous_eu_percentile")
        )
        .otherwise(None)
        .cast(pl.Float64)
        .alias("eu_percentile_change_qoq")
    )

    # -------------------------------------------------------------------------
    # Country percentile movement
    # -------------------------------------------------------------------------

    history = history.with_columns(
        (
            pl.col("qoq_available")
            & pl.col("country_percentile").is_not_null()
            & pl.col("previous_country_percentile").is_not_null()
        )
        .fill_null(False)
        .alias("country_percentile_qoq_available")
    )

    history = history.with_columns(
        pl.when(pl.col("country_percentile_qoq_available"))
        .then(
            pl.col("country_percentile")
            - pl.col("previous_country_percentile")
        )
        .otherwise(None)
        .cast(pl.Float64)
        .alias("country_percentile_change_qoq")
    )

    # -------------------------------------------------------------------------
    # Three-period consecutive-quarter availability
    # -------------------------------------------------------------------------

    history = history.with_columns(
        (
            (~pl.col("is_aggregate"))
            & (pl.col("quarter_gap") == 1)
            & (pl.col("_previous_quarter_gap") == 1)
        )
        .fill_null(False)
        .alias("three_period_pattern_available")
    )

    # Adjacent movements inside the three-observation chain.
    first_leg_change = (
        pl.col("previous_npe_ratio")
        - pl.col("npe_ratio_two_quarters_ago")
    )

    second_leg_change = (
        pl.col("npe_ratio")
        - pl.col("previous_npe_ratio")
    )

    # -------------------------------------------------------------------------
    # Three-period measures
    # -------------------------------------------------------------------------

    history = history.with_columns(
        [
            pl.when(pl.col("three_period_pattern_available"))
            .then(
                pl.col("npe_ratio")
                - pl.col("npe_ratio_two_quarters_ago")
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("npe_ratio_change_3_period"),

            pl.when(pl.col("three_period_pattern_available"))
            .then(
                pl.col("npe_ratio_pct")
                - pl.col("npe_ratio_pct_two_quarters_ago")
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("npe_ratio_pp_change_3_period"),

            pl.when(pl.col("three_period_pattern_available"))
            .then(
                (first_leg_change > 0).cast(pl.Int8)
                + (second_leg_change > 0).cast(pl.Int8)
            )
            .otherwise(None)
            .cast(pl.Int8)
            .alias("npe_increase_count_3_period"),

            pl.when(pl.col("three_period_pattern_available"))
            .then(
                (first_leg_change < 0).cast(pl.Int8)
                + (second_leg_change < 0).cast(pl.Int8)
            )
            .otherwise(None)
            .cast(pl.Int8)
            .alias("npe_decrease_count_3_period"),
        ]
    )

    # -------------------------------------------------------------------------
    # Three-period directional pattern
    # -------------------------------------------------------------------------

    history = history.with_columns(
        pl.when(~pl.col("three_period_pattern_available"))
        .then(None)

        .when(
            (first_leg_change > 0)
            & (second_leg_change > 0)
        )
        .then(pl.lit("consecutive_increase"))

        .when(
            (first_leg_change < 0)
            & (second_leg_change < 0)
        )
        .then(pl.lit("consecutive_decrease"))

        .when(
            (first_leg_change > 0)
            & (second_leg_change < 0)
        )
        .then(pl.lit("increase_then_decrease"))

        .when(
            (first_leg_change < 0)
            & (second_leg_change > 0)
        )
        .then(pl.lit("decrease_then_increase"))

        .otherwise(pl.lit("stable_or_mixed"))
        .cast(pl.String)
        .alias("npe_three_period_pattern")
    )

    return history.select(
        KEY_COLUMNS + TREND_COLUMNS
    )


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def main() -> None:
    print_header("NPE TREND MART INDEPENDENT VALIDATION")
    print("European Bank Risk Intelligence & Automated Reporting Platform")

    print(f"\nProject root        : {PROJECT_ROOT}")
    print(f"NPE Benchmark mart  : {BENCHMARK_MART_PATH}")
    print(f"NPE Trend mart      : {TREND_MART_PATH}")

    print("\nVALIDATION MODE: READ ONLY")
    print("No analytical, staging, reference, or raw data will be modified.")

    print(
        "\nINDEPENDENT VALIDATION PRINCIPLE:\n"
        "Trend measures are reconstructed directly from "
        "mart_npe_benchmark.parquet.\n"
        "Saved trend values are not used to derive their own expected values."
    )

    # =========================================================================
    # 1. FILE AVAILABILITY
    # =========================================================================

    print_header("1. FILE AVAILABILITY")

    require(
        BENCHMARK_MART_PATH.exists(),
        f"Found source mart: {BENCHMARK_MART_PATH.name}",
        f"Source mart not found: {BENCHMARK_MART_PATH}",
    )

    require(
        TREND_MART_PATH.exists(),
        f"Found trend mart: {TREND_MART_PATH.name}",
        f"Trend mart not found: {TREND_MART_PATH}",
    )

    # =========================================================================
    # 2. LOAD DATA
    # =========================================================================

    print_header("2. LOAD APPROVED ANALYTICAL DATA")

    benchmark = pl.read_parquet(BENCHMARK_MART_PATH)
    trend = pl.read_parquet(TREND_MART_PATH)

    print(
        f"[PASS] NPE Benchmark mart opened: "
        f"{benchmark.height:,} rows"
    )
    print(
        f"[PASS] NPE Trend mart opened: "
        f"{trend.height:,} rows"
    )

    # =========================================================================
    # 3. SOURCE STRUCTURE
    # =========================================================================

    print_header("3. SOURCE STRUCTURE VALIDATION")

    missing_source_columns = [
        column
        for column in SOURCE_REQUIRED_COLUMNS
        if column not in benchmark.columns
    ]

    require(
        not missing_source_columns,
        "All required source fields are present.",
        (
            "Required benchmark-source columns are missing: "
            f"{missing_source_columns}"
        ),
    )

    require(
        benchmark.height > 0,
        f"Benchmark source contains {benchmark.height:,} rows.",
        "Benchmark source is empty.",
    )

    duplicate_source_keys = (
        benchmark
        .group_by(KEY_COLUMNS)
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    require(
        duplicate_source_keys == 0,
        "Benchmark source is unique at grain: lei_code x period.",
        (
            "Benchmark source contains duplicate "
            "lei_code x period observations."
        ),
    )

    # =========================================================================
    # 4. TREND MART STRUCTURAL VALIDATION
    # =========================================================================

    print_header("4. TREND MART STRUCTURAL VALIDATION")

    require(
        trend.height == benchmark.height,
        (
            f"Trend row count reconciles: "
            f"{trend.height:,} rows."
        ),
        (
            f"Row-count mismatch: benchmark={benchmark.height:,}, "
            f"trend={trend.height:,}"
        ),
    )

    expected_column_order = benchmark.columns + TREND_COLUMNS

    require(
        trend.columns == expected_column_order,
        (
            f"Trend mart contains the approved "
            f"{len(expected_column_order)} columns in exact order."
        ),
        (
            "Trend mart columns do not match the expected "
            "benchmark-plus-trend design."
        ),
    )

    duplicate_trend_keys = (
        trend
        .group_by(KEY_COLUMNS)
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    require(
        duplicate_trend_keys == 0,
        "Trend mart remains unique at grain: lei_code x period.",
        "Trend mart contains duplicate lei_code x period rows.",
    )

    # =========================================================================
    # 5. TREND COLUMN DATA TYPES
    # =========================================================================

    print_header("5. TREND COLUMN DATA-TYPE VALIDATION")

    for column, expected_dtype in EXPECTED_TREND_DTYPES.items():
        actual_dtype = trend.schema[column]

        require(
            actual_dtype == expected_dtype,
            f"{column} dtype is {actual_dtype}.",
            (
                f"{column} dtype mismatch: "
                f"expected {expected_dtype}, "
                f"found {actual_dtype}"
            ),
        )

    # =========================================================================
    # 6. SOURCE FIELD PRESERVATION
    # =========================================================================

    print_header("6. BENCHMARK SOURCE FIELD PRESERVATION")

    source_sorted = benchmark.sort(KEY_COLUMNS)

    saved_source_sorted = (
        trend
        .select(benchmark.columns)
        .sort(KEY_COLUMNS)
    )

    require(
        source_sorted.to_dicts() == saved_source_sorted.to_dicts(),
        "All original NPE Benchmark mart fields are preserved exactly.",
        (
            "At least one original benchmark-source field differs "
            "inside the trend mart."
        ),
    )

    # =========================================================================
    # 7. INDEPENDENT TREND RECONSTRUCTION
    # =========================================================================

    print_header("7. INDEPENDENT TREND RECONSTRUCTION")

    expected = reconstruct_expected_trends(benchmark)

    require(
        expected.height == benchmark.height,
        (
            "Independent reconstruction created "
            f"{expected.height:,} observations."
        ),
        "Independent reconstruction row count does not reconcile.",
    )

    duplicate_expected_keys = (
        expected
        .group_by(KEY_COLUMNS)
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    require(
        duplicate_expected_keys == 0,
        (
            "Independent trend reconstruction is unique at grain: "
            "lei_code x period."
        ),
        "Independent reconstruction contains duplicate business keys.",
    )

    # =========================================================================
    # 8. SAVED-TO-EXPECTED KEY ALIGNMENT
    # =========================================================================

    print_header("8. SAVED MART TO INDEPENDENT RECONSTRUCTION ALIGNMENT")

    actual_keys = set(
        trend
        .select(KEY_COLUMNS)
        .iter_rows()
    )

    expected_keys = set(
        expected
        .select(KEY_COLUMNS)
        .iter_rows()
    )

    missing_expected = actual_keys - expected_keys
    missing_actual = expected_keys - actual_keys

    require(
        not missing_expected and not missing_actual,
        (
            "Saved trend observations align one-to-one with "
            "the independent reconstruction."
        ),
        (
            "Trend key mismatch detected: "
            f"{len(missing_expected)} saved-only, "
            f"{len(missing_actual)} expected-only."
        ),
    )

    # =========================================================================
    # 9. EXACT / NUMERICAL TREND FIELD RECONCILIATION
    # =========================================================================

    print_header("9. TREND FIELD RECONCILIATION")

    comparison = (
        trend
        .select(KEY_COLUMNS + TREND_COLUMNS)
        .join(
            expected,
            on=KEY_COLUMNS,
            how="inner",
            suffix="_expected",
        )
        .sort(KEY_COLUMNS)
    )

    require(
        comparison.height == trend.height,
        (
            f"Comparison contains all "
            f"{comparison.height:,} trend observations."
        ),
        "Comparison row count does not reconcile to the saved trend mart.",
    )

    for column in TREND_COLUMNS:
        actual_values = comparison[column].to_list()
        expected_values = comparison[f"{column}_expected"].to_list()

        (
            equal,
            mismatch_count,
            maximum_difference,
        ) = values_equal(
            actual_values,
            expected_values,
        )

        if EXPECTED_TREND_DTYPES[column] == pl.Float64:
            require(
                equal,
                (
                    f"{column} independently reconciles "
                    f"within tolerance {FLOAT_TOLERANCE:g}."
                ),
                (
                    f"{column} contains {mismatch_count} mismatch(es); "
                    f"maximum absolute difference="
                    f"{maximum_difference:.12g}"
                ),
            )
        else:
            require(
                equal,
                f"{column} independently reconciles exactly.",
                (
                    f"{column} contains "
                    f"{mismatch_count} mismatch(es)."
                ),
            )

    # =========================================================================
    # 10. QOQ AVAILABILITY BUSINESS RULE
    # =========================================================================

    print_header("10. GENUINE QUARTER-ON-QUARTER BUSINESS RULE")

    actual_qoq = count_true(trend, "qoq_available")
    expected_qoq = count_true(expected, "qoq_available")

    print(f"[INFO] Saved genuine QoQ rows       : {actual_qoq:,}")
    print(f"[INFO] Reconstructed genuine QoQ   : {expected_qoq:,}")

    invalid_qoq = (
        trend
        .filter(
            pl.col("qoq_available")
            & (
                (pl.col("quarter_gap") != 1)
                | pl.col("is_aggregate")
            )
        )
        .height
    )

    require(
        invalid_qoq == 0,
        (
            "Every QoQ observation represents a genuine consecutive "
            "quarter for a non-aggregate institution."
        ),
        (
            f"{invalid_qoq} observation(s) violate the "
            "consecutive-quarter QoQ rule."
        ),
    )

    unavailable_with_qoq_values = (
        trend
        .filter(
            (~pl.col("qoq_available"))
            & (
                pl.col("npe_ratio_change_qoq").is_not_null()
                | pl.col("npe_ratio_pp_change_qoq").is_not_null()
            )
        )
        .height
    )

    require(
        unavailable_with_qoq_values == 0,
        "QoQ NPE measures exist only for genuine consecutive quarters.",
        (
            f"{unavailable_with_qoq_values} non-QoQ observation(s) "
            "contain populated QoQ measures."
        ),
    )

    # =========================================================================
    # 11. EUROPEAN PERCENTILE MOVEMENT RULE
    # =========================================================================

    print_header("11. EUROPEAN PERCENTILE MOVEMENT VALIDATION")

    actual_eu_qoq = count_true(
        trend,
        "eu_percentile_qoq_available",
    )

    expected_eu_qoq = count_true(
        expected,
        "eu_percentile_qoq_available",
    )

    print(
        f"[INFO] Saved EU-percentile QoQ rows       : "
        f"{actual_eu_qoq:,}"
    )
    print(
        f"[INFO] Reconstructed EU-percentile QoQ   : "
        f"{expected_eu_qoq:,}"
    )

    invalid_eu = (
        trend
        .filter(
            pl.col("eu_percentile_qoq_available")
            & (
                (~pl.col("qoq_available"))
                | pl.col("eu_percentile").is_null()
                | pl.col("previous_eu_percentile").is_null()
            )
        )
        .height
    )

    require(
        invalid_eu == 0,
        (
            "European percentile movement is calculated only where "
            "both adjacent genuine-quarter percentile observations exist."
        ),
        (
            f"{invalid_eu} observation(s) violate the "
            "EU-percentile movement rule."
        ),
    )

    # =========================================================================
    # 12. COUNTRY PERCENTILE MOVEMENT RULE
    # =========================================================================

    print_header("12. COUNTRY PERCENTILE MOVEMENT VALIDATION")

    actual_country_qoq = count_true(
        trend,
        "country_percentile_qoq_available",
    )

    expected_country_qoq = count_true(
        expected,
        "country_percentile_qoq_available",
    )

    print(
        f"[INFO] Saved country-percentile QoQ rows       : "
        f"{actual_country_qoq:,}"
    )
    print(
        f"[INFO] Reconstructed country-percentile QoQ   : "
        f"{expected_country_qoq:,}"
    )

    invalid_country = (
        trend
        .filter(
            pl.col("country_percentile_qoq_available")
            & (
                (~pl.col("qoq_available"))
                | pl.col("country_percentile").is_null()
                | pl.col("previous_country_percentile").is_null()
            )
        )
        .height
    )

    require(
        invalid_country == 0,
        (
            "Country-percentile movement is calculated only where "
            "both adjacent formal country benchmark values exist."
        ),
        (
            f"{invalid_country} observation(s) violate the "
            "country-percentile movement rule."
        ),
    )

    # =========================================================================
    # 13. THREE-PERIOD PATTERN VALIDATION
    # =========================================================================

    print_header("13. THREE-PERIOD DIRECTIONAL PATTERN VALIDATION")

    actual_three_period = count_true(
        trend,
        "three_period_pattern_available",
    )

    expected_three_period = count_true(
        expected,
        "three_period_pattern_available",
    )

    print(
        f"[INFO] Saved three-period pattern rows       : "
        f"{actual_three_period:,}"
    )
    print(
        f"[INFO] Reconstructed three-period rows       : "
        f"{expected_three_period:,}"
    )

    populated_patterns = set(
        trend
        .filter(
            pl.col("npe_three_period_pattern").is_not_null()
        )
        .get_column("npe_three_period_pattern")
        .unique()
        .to_list()
    )

    invalid_patterns = (
        populated_patterns
        - VALID_THREE_PERIOD_PATTERNS
    )

    require(
        not invalid_patterns,
        "All populated three-period patterns use approved methodology values.",
        (
            "Unexpected three-period pattern values found: "
            f"{sorted(invalid_patterns)}"
        ),
    )

    unavailable_patterns = (
        trend
        .filter(
            (~pl.col("three_period_pattern_available"))
            & (
                pl.col("npe_ratio_change_3_period").is_not_null()
                | pl.col("npe_ratio_pp_change_3_period").is_not_null()
                | pl.col("npe_increase_count_3_period").is_not_null()
                | pl.col("npe_decrease_count_3_period").is_not_null()
                | pl.col("npe_three_period_pattern").is_not_null()
            )
        )
        .height
    )

    require(
        unavailable_patterns == 0,
        (
            "Three-period measures exist only where two consecutive "
            "quarter transitions are available."
        ),
        (
            f"{unavailable_patterns} observation(s) contain "
            "three-period metrics without valid availability."
        ),
    )

    pattern_profile = (
        trend
        .filter(
            pl.col("npe_three_period_pattern").is_not_null()
        )
        .group_by("npe_three_period_pattern")
        .agg(
            pl.len()
            .cast(pl.UInt32)
            .alias("rows")
        )
        .sort("npe_three_period_pattern")
    )

    print("\nThree-period pattern distribution:")
    print(pattern_profile)

    # =========================================================================
    # 14. AGGREGATE EXCLUSION
    # =========================================================================

    print_header("14. AGGREGATE ENTITY TREND EXCLUSION")

    aggregate_rows = trend.filter(pl.col("is_aggregate"))

    print(
        f"[INFO] Aggregate analytical rows: "
        f"{aggregate_rows.height:,}"
    )

    aggregate_trend_violation = (
        aggregate_rows
        .filter(
            pl.col("qoq_available")
            | pl.col("eu_percentile_qoq_available")
            | pl.col("country_percentile_qoq_available")
            | pl.col("three_period_pattern_available")
            | pl.col("npe_ratio_change_qoq").is_not_null()
            | pl.col("npe_ratio_pp_change_qoq").is_not_null()
            | pl.col("eu_percentile_change_qoq").is_not_null()
            | pl.col("country_percentile_change_qoq").is_not_null()
            | pl.col("npe_ratio_change_3_period").is_not_null()
            | pl.col("npe_ratio_pp_change_3_period").is_not_null()
            | pl.col("npe_three_period_pattern").is_not_null()
        )
        .height
    )

    require(
        aggregate_trend_violation == 0,
        (
            "Aggregate analytical entity is excluded from all "
            "institution-level trend calculations."
        ),
        (
            f"{aggregate_trend_violation} aggregate row(s) "
            "contain institution-level trend measures."
        ),
    )

    # =========================================================================
    # 15. PERIOD / COVERAGE PROFILE
    # =========================================================================

    print_header("15. TREND POPULATION PROFILE")

    population_profile = trend.select(
        [
            pl.len().cast(pl.UInt32).alias("rows"),
            pl.col("lei_code")
            .n_unique()
            .cast(pl.UInt32)
            .alias("entities"),
            pl.col("period")
            .n_unique()
            .cast(pl.UInt32)
            .alias("periods"),
            pl.col("is_aggregate")
            .cast(pl.UInt32)
            .sum()
            .alias("aggregate_rows"),
            pl.col("qoq_available")
            .cast(pl.UInt32)
            .sum()
            .alias("valid_qoq_rows"),
            pl.col("eu_percentile_qoq_available")
            .cast(pl.UInt32)
            .sum()
            .alias("eu_percentile_qoq_rows"),
            pl.col("country_percentile_qoq_available")
            .cast(pl.UInt32)
            .sum()
            .alias("country_percentile_qoq_rows"),
            pl.col("three_period_pattern_available")
            .cast(pl.UInt32)
            .sum()
            .alias("three_period_pattern_rows"),
        ]
    )

    print(population_profile)

    period_profile = (
        trend
        .group_by("period")
        .agg(
            [
                pl.len()
                .cast(pl.UInt32)
                .alias("rows"),

                pl.col("lei_code")
                .n_unique()
                .cast(pl.UInt32)
                .alias("entities"),

                pl.col("qoq_available")
                .cast(pl.UInt32)
                .sum()
                .alias("valid_qoq_rows"),

                pl.col("three_period_pattern_available")
                .cast(pl.UInt32)
                .sum()
                .alias("three_period_rows"),
            ]
        )
        .sort("period")
    )

    print("\nPeriod profile:")
    print(period_profile)

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print_header(
        "NPE TREND MART INDEPENDENT VALIDATION COMPLETED SUCCESSFULLY"
    )

    print("OVERALL STATUS : PASS")

    print(
        "\nINTERPRETATION : The saved NPE trend mart independently "
        "reconciles to the approved NPE benchmark mart and the "
        "documented trend methodology."
    )

    print(
        "\nValidated controls include:"
        "\n  - benchmark-source observation preservation"
        "\n  - institution-period business grain"
        "\n  - chronological lag reconstruction"
        "\n  - genuine consecutive-quarter identification"
        "\n  - QoQ NPE Ratio movement"
        "\n  - QoQ percentage-point movement"
        "\n  - European percentile movement"
        "\n  - country-percentile movement"
        "\n  - three-period consecutive-quarter eligibility"
        "\n  - three-period NPE movement"
        "\n  - directional-pattern reconstruction"
        "\n  - aggregate-entity exclusion"
        "\n  - missing-history preservation"
    )

    print(
        "\nACTION         : mart_npe_trend can be considered for "
        "approval for downstream trend analysis and management reporting."
    )

    print(
        "\nNo deterioration thresholds, traffic-light classifications, "
        "risk ratings, or early-warning flags were approved by this "
        "validation."
    )

    print("\nVALIDATION MODE : READ ONLY")
    print("No data was modified.")


if __name__ == "__main__":
    main()
