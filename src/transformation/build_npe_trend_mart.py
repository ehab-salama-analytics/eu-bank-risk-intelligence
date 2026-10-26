"""
Build the analytical NPE trend mart.

European Bank Risk Intelligence & Automated Reporting Platform

Purpose
-------
Create institution-level consecutive-quarter and short multi-period
trend measures from the independently validated NPE benchmark mart.

Approved methodology implemented
---------------------------------
1. QoQ NPE Ratio change
2. QoQ NPE Ratio percentage-point change
3. QoQ European percentile movement
4. QoQ country percentile movement where formal benchmarks exist
5. Three-observation directional pattern
6. Three-period net NPE Ratio change
7. Three-period percentage-point change
8. Count of increases/decreases in the three-observation window

Important controls
------------------
- Source benchmark observations are preserved.
- No rows are aggregated.
- No rows are deleted.
- No periods are interpolated.
- No synthetic observations are created.
- Aggregate entities do not receive institution-level trend measures.
- QoQ calculations require quarter_gap == 1.
- Three-period patterns require two consecutive quarter gaps == 1.
- No YoY calculation is performed.
- No risk thresholds or early-warning flags are applied.
"""

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_benchmark.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_trend.parquet"
)


# ---------------------------------------------------------------------
# REQUIRED SOURCE FIELDS
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "lei_code",
    "institution_name",
    "institution_country_code",
    "institution_country",
    "entity_type",
    "is_aggregate",
    "period",
    "year",
    "quarter",
    "npe_ratio",
    "npe_ratio_pct",
    "eu_percentile",
    "country_benchmark_available",
    "country_percentile",
]


# ---------------------------------------------------------------------
# APPROVED TREND OUTPUT FIELDS
# ---------------------------------------------------------------------

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


APPROVED_PATTERN_VALUES = {
    "consecutive_increase",
    "consecutive_decrease",
    "increase_then_decrease",
    "decrease_then_increase",
    "stable_or_mixed",
}


# ---------------------------------------------------------------------
# PRINT HELPERS
# ---------------------------------------------------------------------

def print_rule() -> None:
    print("=" * 92)


def print_section(title: str) -> None:
    print()
    print_rule()
    print(title)
    print_rule()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    print_rule()
    print("BUILDING ANALYTICAL NPE TREND MART")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    print_rule()

    print(f"Source : {SOURCE_PATH}")
    print(f"Output : {OUTPUT_PATH}")

    print()
    print("TREND METHODOLOGY:")
    print("  - institution trends exclude aggregate analytical entities")
    print("  - QoQ calculations require quarter_gap == 1")
    print("  - three-period patterns require two consecutive quarter gaps")
    print("  - no interpolation")
    print("  - no synthetic observations")
    print("  - no YoY calculation")
    print("  - no risk thresholds")
    print("  - no traffic lights")
    print("  - no early-warning flags")

    # -----------------------------------------------------------------
    # 1. LOAD APPROVED BENCHMARK MART
    # -----------------------------------------------------------------

    print_section("1. LOAD APPROVED NPE BENCHMARK MART")

    require(
        SOURCE_PATH.exists(),
        f"Source benchmark mart does not exist: {SOURCE_PATH}",
    )

    print(f"[PASS] Source mart exists: {SOURCE_PATH.name}")

    source = pl.read_parquet(SOURCE_PATH)

    print(
        f"[PASS] NPE benchmark mart opened successfully: "
        f"{source.height:,} rows"
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in source.columns
    ]

    require(
        len(missing_columns) == 0,
        f"Missing required source columns: {missing_columns}",
    )

    print("[PASS] All required trend source fields are present.")

    source_columns = list(source.columns)

    # -----------------------------------------------------------------
    # 2. SOURCE STRUCTURE AND GRAIN VALIDATION
    # -----------------------------------------------------------------

    print_section("2. SOURCE STRUCTURE AND GRAIN VALIDATION")

    print(f"[INFO] Source rows              : {source.height:,}")
    print(f"[INFO] Source columns           : {source.width}")
    print(
        f"[INFO] Distinct entities        : "
        f"{source.select(pl.col('lei_code').n_unique()).item():,}"
    )
    print(
        f"[INFO] Distinct periods         : "
        f"{source.select(pl.col('period').n_unique()).item():,}"
    )

    duplicate_grain = (
        source
        .group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    require(
        duplicate_grain.height == 0,
        "Source benchmark mart is not unique at grain lei_code x period.",
    )

    print("[PASS] Source is unique at grain: lei_code x period")

    null_key_rows = source.filter(
        pl.col("lei_code").is_null()
        | pl.col("period").is_null()
    ).height

    require(
        null_key_rows == 0,
        "Source contains NULL values in the analytical business key.",
    )

    print("[PASS] Business grain contains no NULL values.")

    null_npe_ratios = source.filter(
        pl.col("npe_ratio").is_null()
    ).height

    require(
        null_npe_ratios == 0,
        "Source contains NULL NPE Ratio observations.",
    )

    print("[PASS] Source NPE Ratio contains no NULL observations.")

    # Keep a deterministic source order.
    source_sorted = source.sort(["lei_code", "period"])

    # -----------------------------------------------------------------
    # 3. BUILD CHRONOLOGICAL INSTITUTION HISTORY
    # -----------------------------------------------------------------

    print_section("3. BUILD CHRONOLOGICAL INSTITUTION HISTORY")

    work = source_sorted.with_columns(
        (
            pl.col("year").cast(pl.Int64) * 4
            + pl.col("quarter").cast(pl.Int64)
        ).alias("_quarter_index")
    )

    work = work.with_columns(
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

            pl.col("country_benchmark_available")
            .shift(1)
            .over("lei_code")
            .alias("_previous_country_benchmark_available"),

            pl.col("period")
            .shift(2)
            .over("lei_code")
            .alias("period_two_quarters_ago"),

            pl.col("_quarter_index")
            .shift(2)
            .over("lei_code")
            .alias("_two_quarters_ago_index"),

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

    work = work.with_columns(
        [
            (
                pl.col("_quarter_index")
                - pl.col("_previous_quarter_index")
            ).alias("quarter_gap"),

            (
                pl.col("_previous_quarter_index")
                - pl.col("_two_quarters_ago_index")
            ).alias("_previous_quarter_gap"),
        ]
    )

    aggregate_rows = work.filter(
        pl.col("is_aggregate")
    ).height

    non_aggregate_rows = work.filter(
        ~pl.col("is_aggregate")
    ).height

    print(f"[INFO] Aggregate rows            : {aggregate_rows:,}")
    print(f"[INFO] Non-aggregate rows        : {non_aggregate_rows:,}")

    # -----------------------------------------------------------------
    # 4. DEFINE VALID CONSECUTIVE-QUARTER OBSERVATIONS
    # -----------------------------------------------------------------

    print_section("4. DEFINE VALID CONSECUTIVE-QUARTER OBSERVATIONS")

    work = work.with_columns(
        (
            (~pl.col("is_aggregate"))
            & (pl.col("quarter_gap") == 1)
        )
        .fill_null(False)
        .alias("qoq_available")
    )

    work = work.with_columns(
        [
            (
                pl.col("qoq_available")
                & pl.col("eu_percentile").is_not_null()
                & pl.col("previous_eu_percentile").is_not_null()
            )
            .fill_null(False)
            .alias("eu_percentile_qoq_available"),

            (
                pl.col("qoq_available")
                & pl.col("country_benchmark_available").fill_null(False)
                & pl.col(
                    "_previous_country_benchmark_available"
                ).fill_null(False)
                & pl.col("country_percentile").is_not_null()
                & pl.col("previous_country_percentile").is_not_null()
            )
            .fill_null(False)
            .alias("country_percentile_qoq_available"),

            (
                (~pl.col("is_aggregate"))
                & (pl.col("quarter_gap") == 1)
                & (pl.col("_previous_quarter_gap") == 1)
            )
            .fill_null(False)
            .alias("three_period_pattern_available"),
        ]
    )

    qoq_rows = work.filter(
        pl.col("qoq_available")
    ).height

    eu_qoq_rows = work.filter(
        pl.col("eu_percentile_qoq_available")
    ).height

    country_qoq_rows = work.filter(
        pl.col("country_percentile_qoq_available")
    ).height

    three_period_rows = work.filter(
        pl.col("three_period_pattern_available")
    ).height

    print(f"[INFO] Genuine QoQ observations             : {qoq_rows:,}")
    print(f"[INFO] QoQ observations with EU percentile  : {eu_qoq_rows:,}")
    print(
        f"[INFO] QoQ observations with country "
        f"percentile: {country_qoq_rows:,}"
    )
    print(
        f"[INFO] Three-period consecutive observations: "
        f"{three_period_rows:,}"
    )

    # -----------------------------------------------------------------
    # 5. CALCULATE APPROVED QoQ TREND MEASURES
    # -----------------------------------------------------------------

    print_section("5. CALCULATE APPROVED QoQ TREND MEASURES")

    work = work.with_columns(
        [
            pl.when(pl.col("qoq_available"))
            .then(
                pl.col("npe_ratio")
                - pl.col("previous_npe_ratio")
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("npe_ratio_change_qoq"),

            pl.when(pl.col("qoq_available"))
            .then(
                pl.col("npe_ratio_pct")
                - pl.col("previous_npe_ratio_pct")
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("npe_ratio_pp_change_qoq"),

            pl.when(pl.col("eu_percentile_qoq_available"))
            .then(
                pl.col("eu_percentile")
                - pl.col("previous_eu_percentile")
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("eu_percentile_change_qoq"),

            pl.when(pl.col("country_percentile_qoq_available"))
            .then(
                pl.col("country_percentile")
                - pl.col("previous_country_percentile")
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("country_percentile_change_qoq"),
        ]
    )

    print("[PASS] Consecutive-quarter NPE Ratio changes calculated.")
    print("[PASS] Consecutive-quarter percentage-point changes calculated.")
    print("[PASS] European percentile movements calculated where valid.")
    print("[PASS] Country percentile movements calculated where valid.")

    # -----------------------------------------------------------------
    # 6. CALCULATE THREE-PERIOD DIRECTIONAL PATTERN
    # -----------------------------------------------------------------

    print_section("6. CALCULATE THREE-PERIOD DIRECTIONAL PATTERN")

    work = work.with_columns(
        (
            pl.col("previous_npe_ratio")
            - pl.col("npe_ratio_two_quarters_ago")
        ).alias("_previous_qoq_change")
    )

    work = work.with_columns(
        [
            pl.when(pl.col("three_period_pattern_available"))
            .then(
                pl.col("npe_ratio")
                - pl.col("npe_ratio_two_quarters_ago")
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("npe_ratio_change_3_period"),

            pl.when(pl.col("three_period_pattern_available"))
            .then(
                pl.col("npe_ratio_pct")
                - pl.col("npe_ratio_pct_two_quarters_ago")
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("npe_ratio_pp_change_3_period"),

            pl.when(pl.col("three_period_pattern_available"))
            .then(
                (
                    (pl.col("_previous_qoq_change") > 0)
                    .cast(pl.Int8)
                    +
                    (pl.col("npe_ratio_change_qoq") > 0)
                    .cast(pl.Int8)
                )
            )
            .otherwise(pl.lit(None, dtype=pl.Int8))
            .alias("npe_increase_count_3_period"),

            pl.when(pl.col("three_period_pattern_available"))
            .then(
                (
                    (pl.col("_previous_qoq_change") < 0)
                    .cast(pl.Int8)
                    +
                    (pl.col("npe_ratio_change_qoq") < 0)
                    .cast(pl.Int8)
                )
            )
            .otherwise(pl.lit(None, dtype=pl.Int8))
            .alias("npe_decrease_count_3_period"),
        ]
    )

    work = work.with_columns(
        pl.when(
            pl.col("three_period_pattern_available")
            & (pl.col("_previous_qoq_change") > 0)
            & (pl.col("npe_ratio_change_qoq") > 0)
        )
        .then(pl.lit("consecutive_increase"))

        .when(
            pl.col("three_period_pattern_available")
            & (pl.col("_previous_qoq_change") < 0)
            & (pl.col("npe_ratio_change_qoq") < 0)
        )
        .then(pl.lit("consecutive_decrease"))

        .when(
            pl.col("three_period_pattern_available")
            & (pl.col("_previous_qoq_change") > 0)
            & (pl.col("npe_ratio_change_qoq") < 0)
        )
        .then(pl.lit("increase_then_decrease"))

        .when(
            pl.col("three_period_pattern_available")
            & (pl.col("_previous_qoq_change") < 0)
            & (pl.col("npe_ratio_change_qoq") > 0)
        )
        .then(pl.lit("decrease_then_increase"))

        .when(pl.col("three_period_pattern_available"))
        .then(pl.lit("stable_or_mixed"))

        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("npe_three_period_pattern")
    )

    pattern_profile = (
        work
        .filter(pl.col("npe_three_period_pattern").is_not_null())
        .group_by("npe_three_period_pattern")
        .len()
        .rename({"len": "rows"})
        .sort("npe_three_period_pattern")
    )

    print("Three-period pattern profile:")
    print(pattern_profile)

    # -----------------------------------------------------------------
    # 7. MATERIALISE ANALYTICAL TREND MART
    # -----------------------------------------------------------------

    print_section("7. MATERIALISE ANALYTICAL TREND MART")

    mart = (
        work
        .select(source_columns + TREND_COLUMNS)
        .sort(["lei_code", "period"])
    )

    print(
        f"[PASS] Trend mart created in memory: "
        f"{mart.height:,} rows, {mart.width} columns"
    )

    # -----------------------------------------------------------------
    # 8. PRE-SAVE VALIDATION
    # -----------------------------------------------------------------

    print_section("8. PRE-SAVE TREND MART VALIDATION")

    require(
        mart.height == source_sorted.height,
        "Trend mart row count does not reconcile to source.",
    )

    print(
        f"[PASS] Row count reconciles: "
        f"{mart.height:,} rows preserved."
    )

    require(
        mart.select(source_columns).equals(source_sorted),
        "Original benchmark mart fields were modified.",
    )

    print("[PASS] All original benchmark mart fields are preserved exactly.")

    duplicate_output_grain = (
        mart
        .group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    require(
        duplicate_output_grain.height == 0,
        "Trend mart is not unique at grain lei_code x period.",
    )

    print("[PASS] Trend mart remains unique at grain: lei_code x period")

    invalid_qoq_values = mart.filter(
        (~pl.col("qoq_available"))
        & (
            pl.col("npe_ratio_change_qoq").is_not_null()
            | pl.col("npe_ratio_pp_change_qoq").is_not_null()
        )
    ).height

    require(
        invalid_qoq_values == 0,
        "QoQ measures exist where quarter continuity is not valid.",
    )

    print(
        "[PASS] QoQ NPE measures exist only for genuine "
        "consecutive-quarter observations."
    )

    invalid_eu_movement = mart.filter(
        (~pl.col("eu_percentile_qoq_available"))
        & pl.col("eu_percentile_change_qoq").is_not_null()
    ).height

    require(
        invalid_eu_movement == 0,
        "European percentile movement exists outside valid observations.",
    )

    print(
        "[PASS] European percentile movements respect "
        "consecutive-quarter availability."
    )

    invalid_country_movement = mart.filter(
        (~pl.col("country_percentile_qoq_available"))
        & pl.col("country_percentile_change_qoq").is_not_null()
    ).height

    require(
        invalid_country_movement == 0,
        "Country percentile movement exists without valid formal benchmarks.",
    )

    print(
        "[PASS] Country percentile movements require formal "
        "benchmarks in both adjacent quarters."
    )

    invalid_three_period_rows = mart.filter(
        (~pl.col("three_period_pattern_available"))
        & (
            pl.col("npe_three_period_pattern").is_not_null()
            | pl.col("npe_ratio_change_3_period").is_not_null()
            | pl.col("npe_ratio_pp_change_3_period").is_not_null()
            | pl.col("npe_increase_count_3_period").is_not_null()
            | pl.col("npe_decrease_count_3_period").is_not_null()
        )
    ).height

    require(
        invalid_three_period_rows == 0,
        "Three-period measures exist without valid consecutive history.",
    )

    print(
        "[PASS] Three-period measures require a valid "
        "three-observation consecutive-quarter chain."
    )

    observed_patterns = set(
        mart
        .select("npe_three_period_pattern")
        .drop_nulls()
        .get_column("npe_three_period_pattern")
        .to_list()
    )

    unexpected_patterns = (
        observed_patterns - APPROVED_PATTERN_VALUES
    )

    require(
        len(unexpected_patterns) == 0,
        f"Unexpected three-period pattern values: {unexpected_patterns}",
    )

    print("[PASS] All populated trend patterns use approved methodology values.")

    aggregate_invalid = mart.filter(
        pl.col("is_aggregate")
        & (
            pl.col("qoq_available")
            | pl.col("eu_percentile_qoq_available")
            | pl.col("country_percentile_qoq_available")
            | pl.col("three_period_pattern_available")
            | pl.col("npe_ratio_change_qoq").is_not_null()
            | pl.col("npe_three_period_pattern").is_not_null()
        )
    ).height

    require(
        aggregate_invalid == 0,
        "Aggregate analytical entity received institution-level trend metrics.",
    )

    print(
        "[PASS] Aggregate analytical entity is excluded from "
        "institution-level trend calculations."
    )

    # -----------------------------------------------------------------
    # 9. SAVE TREND MART
    # -----------------------------------------------------------------

    print_section("9. SAVE ANALYTICAL NPE TREND MART")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mart.write_parquet(OUTPUT_PATH)

    print("[PASS] NPE trend mart written successfully.")
    print()
    print("Saved analytical trend mart:")
    print(f"  {OUTPUT_PATH}")

    # -----------------------------------------------------------------
    # 10. POST-SAVE VALIDATION
    # -----------------------------------------------------------------

    print_section("10. POST-SAVE VALIDATION")

    require(
        OUTPUT_PATH.exists(),
        "Saved trend mart file does not exist.",
    )

    print(f"[PASS] Saved file exists: {OUTPUT_PATH.name}")

    saved = pl.read_parquet(OUTPUT_PATH)

    print(
        f"[PASS] Saved trend mart reopened successfully: "
        f"{saved.height:,} rows"
    )

    require(
        saved.columns == mart.columns,
        "Saved trend mart column order does not reconcile.",
    )

    print("[PASS] Saved trend mart column order reconciles.")

    require(
        saved.height == mart.height,
        "Saved trend mart row count does not reconcile.",
    )

    print(
        f"[PASS] Saved row count reconciles: "
        f"{saved.height:,}"
    )

    require(
        saved.equals(mart),
        "Saved Parquet does not exactly match validated in-memory mart.",
    )

    print(
        "[PASS] Saved Parquet exactly matches the validated "
        "in-memory trend mart."
    )

    duplicate_saved_grain = (
        saved
        .group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    require(
        duplicate_saved_grain.height == 0,
        "Saved trend mart grain is not unique.",
    )

    print("[PASS] Saved institution-period grain remains unique.")

    # -----------------------------------------------------------------
    # 11. FINAL TREND MART PROFILE
    # -----------------------------------------------------------------

    print_section("11. FINAL NPE TREND MART PROFILE")

    profile = saved.select(
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

            pl.col("qoq_available")
            .sum()
            .alias("valid_qoq_rows"),

            pl.col("eu_percentile_qoq_available")
            .sum()
            .alias("eu_percentile_qoq_rows"),

            pl.col("country_percentile_qoq_available")
            .sum()
            .alias("country_percentile_qoq_rows"),

            pl.col("three_period_pattern_available")
            .sum()
            .alias("three_period_pattern_rows"),
        ]
    )

    print(profile)

    print()
    print("Three-period pattern distribution:")

    final_pattern_profile = (
        saved
        .filter(pl.col("npe_three_period_pattern").is_not_null())
        .group_by("npe_three_period_pattern")
        .len()
        .rename({"len": "rows"})
        .sort("npe_three_period_pattern")
    )

    print(final_pattern_profile)

    print_section("NPE TREND MART BUILD COMPLETED SUCCESSFULLY")

    print(f"Output: {OUTPUT_PATH}")
    print()
    print(
        "The approved NPE Ratio and benchmark observations were "
        "preserved and augmented with controlled trend measures."
    )
    print(
        "QoQ measures were populated only for genuine consecutive "
        "reporting quarters."
    )
    print(
        "Three-period directional patterns were populated only where "
        "two consecutive quarter transitions exist."
    )
    print(
        "Aggregate analytical entities were excluded from "
        "institution-level trend calculations."
    )
    print()
    print("No source analytical data was modified.")
    print("No missing periods were interpolated.")
    print("No YoY measures were calculated.")
    print("No thresholds or risk flags were applied.")
    print("The trend mart has not yet received independent validation.")


if __name__ == "__main__":
    main()
