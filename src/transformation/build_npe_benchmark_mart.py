from pathlib import Path
import sys

import polars as pl


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_ratio.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
)

OUTPUT_PATH = OUTPUT_DIR / "mart_npe_benchmark.parquet"


REQUIRED_SOURCE_COLUMNS = [
    "lei_code",
    "institution_name",
    "institution_country_code",
    "institution_country",
    "entity_type",
    "is_aggregate",
    "period",
    "npe_ratio",
    "npe_ratio_pct",
]


BUSINESS_GRAIN = [
    "lei_code",
    "period",
]


COUNTRY_FORMAL_MIN_PEERS = 5
COUNTRY_DESCRIPTIVE_MIN_PEERS = 2


# ============================================================
# OUTPUT HELPERS
# ============================================================

def separator() -> None:
    print("=" * 96)


def section(title: str) -> None:
    print()
    separator()
    print(title)
    separator()


def pass_message(message: str) -> None:
    print(f"[PASS] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def require_columns(
    df: pl.DataFrame,
    required_columns: list[str],
    object_name: str,
) -> None:
    missing = [
        column_name
        for column_name in required_columns
        if column_name not in df.columns
    ]

    if missing:
        fail(
            f"{object_name} is missing required column(s): "
            + ", ".join(missing)
        )


def validate_unique_grain(
    df: pl.DataFrame,
    grain: list[str],
    object_name: str,
) -> None:
    duplicates = (
        df
        .group_by(grain)
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height > 0:
        print()
        print("Duplicate grain sample:")
        print(duplicates.head(20))

        fail(
            f"{object_name} is not unique at grain: "
            + " x ".join(grain)
        )

    pass_message(
        f"{object_name} is unique at grain: "
        + " x ".join(grain)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    separator()
    print("BUILDING ANALYTICAL NPE BENCHMARK MART")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    separator()

    print()
    print(f"Source : {SOURCE_PATH}")
    print(f"Output : {OUTPUT_PATH}")

    print()
    print("BENCHMARK METHODOLOGY:")
    print("  European benchmark")
    print("    - peer population: non-aggregate institutions")
    print("    - partition: reporting period")
    print("    - ranking direction: ascending NPE Ratio")
    print("    - tie method: average rank")
    print(
        "    - percentile: "
        "(average_rank - 1) / (peer_count - 1)"
    )

    print()
    print("  Country benchmark")
    print("    - peer population: non-aggregate institutions")
    print("    - partition: reporting period + country")
    print(
        f"    - formal rank/percentile: "
        f"peer_count >= {COUNTRY_FORMAL_MIN_PEERS}"
    )
    print(
        f"    - descriptive statistics: "
        f"peer_count >= {COUNTRY_DESCRIPTIVE_MIN_PEERS}"
    )
    print("    - singleton groups: benchmark unavailable")

    print()
    print("  Aggregate treatment")
    print("    - EBA aggregate retained in output")
    print("    - aggregate excluded from all peer calculations")
    print("    - aggregate receives no rank or percentile")

    print()
    print("No risk thresholds, traffic lights, or early-warning flags")
    print("are calculated in this build.")

    # ========================================================
    # 1. LOAD APPROVED NPE RATIO MART
    # ========================================================

    section("1. LOAD APPROVED NPE RATIO MART")

    if not SOURCE_PATH.exists():
        fail(f"Source mart does not exist: {SOURCE_PATH}")

    pass_message(f"Source mart exists: {SOURCE_PATH.name}")

    try:
        source = pl.read_parquet(SOURCE_PATH)
    except Exception as exc:
        fail(f"Unable to open source mart: {exc}")

    pass_message(
        f"Source NPE Ratio mart opened successfully: "
        f"{source.height:,} rows"
    )

    require_columns(
        source,
        REQUIRED_SOURCE_COLUMNS,
        "mart_npe_ratio",
    )

    pass_message("All required source columns are present.")

    source_columns = source.columns

    # ========================================================
    # 2. SOURCE STRUCTURE AND GRAIN VALIDATION
    # ========================================================

    section("2. SOURCE STRUCTURE AND GRAIN VALIDATION")

    if source.height == 0:
        fail("Source NPE Ratio mart contains no rows.")

    info(f"Source rows        : {source.height:,}")
    info(f"Source columns     : {source.width:,}")
    info(
        f"Distinct entities  : "
        f"{source['lei_code'].n_unique():,}"
    )
    info(
        f"Distinct periods   : "
        f"{source['period'].n_unique():,}"
    )

    validate_unique_grain(
        source,
        BUSINESS_GRAIN,
        "mart_npe_ratio",
    )

    null_ratios = source["npe_ratio"].null_count()

    if null_ratios > 0:
        fail(
            f"Source contains {null_ratios:,} NULL NPE Ratio value(s)."
        )

    pass_message("Source NPE Ratio contains no NULL values.")

    # ========================================================
    # 3. DEFINE ELIGIBLE PEER POPULATION
    # ========================================================

    section("3. DEFINE ELIGIBLE PEER POPULATION")

    eligible = source.filter(~pl.col("is_aggregate"))
    aggregate = source.filter(pl.col("is_aggregate"))

    eligible_entity_count = eligible["lei_code"].n_unique()
    aggregate_entity_count = aggregate["lei_code"].n_unique()

    info(f"Eligible rows              : {eligible.height:,}")
    info(f"Eligible institutions      : {eligible_entity_count:,}")
    info(f"Aggregate rows             : {aggregate.height:,}")
    info(f"Aggregate analytical entity: {aggregate_entity_count:,}")

    if eligible.height == 0:
        fail("No non-aggregate institutions are available for benchmarking.")

    if aggregate.height == 0:
        fail(
            "Expected EBA aggregate entity is absent from the approved "
            "source mart."
        )

    pass_message(
        "Individual institutions are cleanly separated from aggregate "
        "observations."
    )

    # ========================================================
    # 4. EUROPEAN PEER STATISTICS
    # ========================================================

    section("4. BUILD EUROPEAN PEER STATISTICS")

    eu_statistics = (
        eligible
        .group_by("period")
        .agg(
            [
                pl.len()
                .cast(pl.UInt32)
                .alias("eu_peer_count"),

                pl.col("npe_ratio")
                .mean()
                .alias("eu_mean_npe_ratio"),

                pl.col("npe_ratio")
                .median()
                .alias("eu_median_npe_ratio"),
            ]
        )
        .sort("period")
    )

    print(eu_statistics)

    if eu_statistics["eu_peer_count"].min() <= 1:
        fail(
            "At least one European period contains fewer than two "
            "eligible institutions."
        )

    pass_message(
        "European peer statistics calculated independently by "
        "reporting period."
    )

    # ========================================================
    # 5. EUROPEAN RANK AND PERCENTILE
    # ========================================================

    section("5. BUILD EUROPEAN RANK AND PERCENTILE")

    eu_positions = (
        eligible
        .select(
            [
                "lei_code",
                "period",
                "npe_ratio",

                pl.col("npe_ratio")
                .rank(
                    method="average",
                    descending=False,
                )
                .over("period")
                .cast(pl.Float64)
                .alias("eu_rank"),

                pl.len()
                .over("period")
                .cast(pl.UInt32)
                .alias("eu_peer_count_position"),
            ]
        )
        .with_columns(
            (
                (pl.col("eu_rank") - 1.0)
                /
                (
                    pl.col("eu_peer_count_position")
                    .cast(pl.Float64)
                    - 1.0
                )
            )
            .alias("eu_percentile")
        )
        .with_columns(
            (
                pl.col("eu_percentile") * 100.0
            )
            .alias("eu_percentile_pct")
        )
        .select(
            [
                "lei_code",
                "period",
                "eu_rank",
                "eu_percentile",
                "eu_percentile_pct",
            ]
        )
    )

    eu_position_profile = eu_positions.select(
        [
            pl.len().alias("rows"),
            pl.col("eu_rank").min().alias("minimum_rank"),
            pl.col("eu_rank").max().alias("maximum_rank"),
            pl.col("eu_percentile")
            .min()
            .alias("minimum_percentile"),
            pl.col("eu_percentile")
            .max()
            .alias("maximum_percentile"),
        ]
    )

    print(eu_position_profile)

    invalid_eu_percentiles = eu_positions.filter(
        pl.col("eu_percentile").is_null()
        | (pl.col("eu_percentile") < 0)
        | (pl.col("eu_percentile") > 1)
    )

    if invalid_eu_percentiles.height > 0:
        fail(
            "European percentile construction produced invalid values."
        )

    pass_message(
        "European average-rank and percentile calculations completed."
    )

    # ========================================================
    # 6. COUNTRY PEER STATISTICS
    # ========================================================

    section("6. BUILD COUNTRY PEER STATISTICS")

    country_statistics_raw = (
        eligible
        .group_by(
            [
                "period",
                "institution_country_code",
            ]
        )
        .agg(
            [
                pl.len()
                .cast(pl.UInt32)
                .alias("country_peer_count"),

                pl.col("npe_ratio")
                .mean()
                .alias("country_mean_npe_ratio_raw"),

                pl.col("npe_ratio")
                .median()
                .alias("country_median_npe_ratio_raw"),
            ]
        )
    )

    country_statistics = (
        country_statistics_raw
        .with_columns(
            [
                (
                    pl.col("country_peer_count")
                    >= COUNTRY_DESCRIPTIVE_MIN_PEERS
                )
                .alias("country_descriptive_available"),

                (
                    pl.col("country_peer_count")
                    >= COUNTRY_FORMAL_MIN_PEERS
                )
                .alias("country_benchmark_available"),
            ]
        )
        .with_columns(
            [
                pl.when(
                    pl.col("country_descriptive_available")
                )
                .then(pl.col("country_mean_npe_ratio_raw"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("country_mean_npe_ratio"),

                pl.when(
                    pl.col("country_descriptive_available")
                )
                .then(pl.col("country_median_npe_ratio_raw"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("country_median_npe_ratio"),

                pl.when(
                    pl.col("country_peer_count") == 1
                )
                .then(pl.lit("unavailable_singleton"))

                .when(
                    (
                        pl.col("country_peer_count")
                        >= COUNTRY_DESCRIPTIVE_MIN_PEERS
                    )
                    &
                    (
                        pl.col("country_peer_count")
                        < COUNTRY_FORMAL_MIN_PEERS
                    )
                )
                .then(pl.lit("descriptive_only"))

                .otherwise(
                    pl.lit("formal_benchmark_available")
                )
                .alias("country_benchmark_status"),
            ]
        )
        .select(
            [
                "period",
                "institution_country_code",
                "country_peer_count",
                "country_descriptive_available",
                "country_benchmark_available",
                "country_benchmark_status",
                "country_mean_npe_ratio",
                "country_median_npe_ratio",
            ]
        )
        .sort(
            [
                "period",
                "country_peer_count",
                "institution_country_code",
            ]
        )
    )

    print(country_statistics)

    country_status_profile = (
        country_statistics
        .group_by("country_benchmark_status")
        .agg(
            pl.len().alias("country_period_groups")
        )
        .sort("country_benchmark_status")
    )

    print()
    print("Country benchmark availability profile:")
    print(country_status_profile)

    # ========================================================
    # 7. COUNTRY RANK AND PERCENTILE
    # ========================================================

    section("7. BUILD COUNTRY RANK AND PERCENTILE")

    country_positions_raw = (
        eligible
        .select(
            [
                "lei_code",
                "period",
                "institution_country_code",

                pl.col("npe_ratio")
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
                .cast(pl.Float64)
                .alias("country_rank_raw"),

                pl.len()
                .over(
                    [
                        "period",
                        "institution_country_code",
                    ]
                )
                .cast(pl.UInt32)
                .alias("country_peer_count_position"),
            ]
        )
        .with_columns(
            pl.when(
                pl.col("country_peer_count_position") > 1
            )
            .then(
                (
                    pl.col("country_rank_raw") - 1.0
                )
                /
                (
                    pl.col("country_peer_count_position")
                    .cast(pl.Float64)
                    - 1.0
                )
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("country_percentile_raw")
        )
    )

    country_positions = (
        country_positions_raw
        .with_columns(
            [
                pl.when(
                    pl.col("country_peer_count_position")
                    >= COUNTRY_FORMAL_MIN_PEERS
                )
                .then(pl.col("country_rank_raw"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("country_rank"),

                pl.when(
                    pl.col("country_peer_count_position")
                    >= COUNTRY_FORMAL_MIN_PEERS
                )
                .then(pl.col("country_percentile_raw"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("country_percentile"),
            ]
        )
        .with_columns(
            pl.when(
                pl.col("country_percentile").is_not_null()
            )
            .then(
                pl.col("country_percentile") * 100.0
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("country_percentile_pct")
        )
        .select(
            [
                "lei_code",
                "period",
                "country_rank",
                "country_percentile",
                "country_percentile_pct",
            ]
        )
    )

    formal_country_positions = country_positions.filter(
        pl.col("country_rank").is_not_null()
    )

    suppressed_country_positions = country_positions.filter(
        pl.col("country_rank").is_null()
    )

    info(
        f"Observations with formal country benchmark : "
        f"{formal_country_positions.height:,}"
    )

    info(
        f"Observations with country rank suppressed   : "
        f"{suppressed_country_positions.height:,}"
    )

    invalid_country_percentiles = country_positions.filter(
        pl.col("country_percentile").is_not_null()
        &
        (
            (pl.col("country_percentile") < 0)
            |
            (pl.col("country_percentile") > 1)
        )
    )

    if invalid_country_percentiles.height > 0:
        fail(
            "Country percentile construction produced invalid values."
        )

    pass_message(
        "Country rankings and percentiles are restricted to "
        f"peer groups containing at least "
        f"{COUNTRY_FORMAL_MIN_PEERS} institutions."
    )

    # ========================================================
    # 8. MATERIALISE BENCHMARK MART
    # ========================================================

    section("8. MATERIALISE ANALYTICAL BENCHMARK MART")

    benchmark = (
        source
        .with_columns(
            (~pl.col("is_aggregate"))
            .alias("benchmark_eligible")
        )
        .join(
            eu_statistics,
            on="period",
            how="left",
        )
        .join(
            eu_positions,
            on=[
                "lei_code",
                "period",
            ],
            how="left",
        )
        .join(
            country_statistics,
            on=[
                "period",
                "institution_country_code",
            ],
            how="left",
        )
        .join(
            country_positions,
            on=[
                "lei_code",
                "period",
            ],
            how="left",
        )
    )

    # --------------------------------------------------------
    # Aggregate entities are retained in the mart, but they
    # must not receive peer benchmarking metrics.
    # --------------------------------------------------------

    benchmark = benchmark.with_columns(
        [
            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("eu_peer_count"))
            .otherwise(None)
            .cast(pl.UInt32)
            .alias("eu_peer_count"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("eu_mean_npe_ratio"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("eu_mean_npe_ratio"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("eu_median_npe_ratio"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("eu_median_npe_ratio"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("eu_rank"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("eu_rank"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("eu_percentile"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("eu_percentile"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("eu_percentile_pct"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("eu_percentile_pct"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("country_peer_count"))
            .otherwise(None)
            .cast(pl.UInt32)
            .alias("country_peer_count"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("country_descriptive_available"))
            .otherwise(False)
            .alias("country_descriptive_available"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("country_benchmark_available"))
            .otherwise(False)
            .alias("country_benchmark_available"),

            pl.when(~pl.col("benchmark_eligible"))
            .then(pl.lit("not_eligible_aggregate"))
            .otherwise(pl.col("country_benchmark_status"))
            .alias("country_benchmark_status"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("country_mean_npe_ratio"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("country_mean_npe_ratio"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("country_median_npe_ratio"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("country_median_npe_ratio"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("country_rank"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("country_rank"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("country_percentile"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("country_percentile"),

            pl.when(pl.col("benchmark_eligible"))
            .then(pl.col("country_percentile_pct"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("country_percentile_pct"),
        ]
    )

    # --------------------------------------------------------
    # Median deviations
    # --------------------------------------------------------

    benchmark = benchmark.with_columns(
        [
            pl.when(pl.col("benchmark_eligible"))
            .then(
                pl.col("npe_ratio")
                - pl.col("eu_median_npe_ratio")
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("eu_deviation_from_median"),

            pl.when(
                pl.col("benchmark_eligible")
                &
                pl.col("country_descriptive_available")
            )
            .then(
                pl.col("npe_ratio")
                - pl.col("country_median_npe_ratio")
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("country_deviation_from_median"),
        ]
    )

    pass_message(
        "Benchmark mart created in memory using the approved "
        "benchmark methodology."
    )

    # ========================================================
    # 9. PRE-SAVE VALIDATION
    # ========================================================

    section("9. PRE-SAVE BENCHMARK MART VALIDATION")

    if benchmark.height != source.height:
        fail(
            "Benchmark mart row count does not reconcile to source mart."
        )

    pass_message(
        f"Row count reconciles: {benchmark.height:,} rows preserved."
    )

    validate_unique_grain(
        benchmark,
        BUSINESS_GRAIN,
        "mart_npe_benchmark",
    )

    # --------------------------------------------------------
    # Original source fields must remain unchanged
    # --------------------------------------------------------

    source_projection = (
        source
        .select(source_columns)
        .sort(BUSINESS_GRAIN)
    )

    benchmark_source_projection = (
        benchmark
        .select(source_columns)
        .sort(BUSINESS_GRAIN)
    )

    if not source_projection.equals(benchmark_source_projection):
        fail(
            "Original NPE Ratio mart fields were modified during "
            "benchmark construction."
        )

    pass_message(
        "All original NPE Ratio mart fields are preserved exactly."
    )

    # --------------------------------------------------------
    # Eligible European benchmark completeness
    # --------------------------------------------------------

    eligible_benchmark = benchmark.filter(
        pl.col("benchmark_eligible")
    )

    eu_null_metrics = eligible_benchmark.select(
        [
            pl.col("eu_peer_count")
            .null_count()
            .alias("null_peer_count"),

            pl.col("eu_rank")
            .null_count()
            .alias("null_rank"),

            pl.col("eu_percentile")
            .null_count()
            .alias("null_percentile"),

            pl.col("eu_mean_npe_ratio")
            .null_count()
            .alias("null_mean"),

            pl.col("eu_median_npe_ratio")
            .null_count()
            .alias("null_median"),
        ]
    )

    print("European benchmark completeness:")
    print(eu_null_metrics)

    if (
        eu_null_metrics.row(0)[0] > 0
        or eu_null_metrics.row(0)[1] > 0
        or eu_null_metrics.row(0)[2] > 0
        or eu_null_metrics.row(0)[3] > 0
        or eu_null_metrics.row(0)[4] > 0
    ):
        fail(
            "Eligible institutions contain incomplete European "
            "benchmark metrics."
        )

    pass_message(
        "Every eligible institution has complete European "
        "benchmark metrics."
    )

    # --------------------------------------------------------
    # Aggregate exclusion control
    # --------------------------------------------------------

    aggregate_benchmark = benchmark.filter(
        pl.col("is_aggregate")
    )

    aggregate_ranked = aggregate_benchmark.filter(
        pl.col("eu_rank").is_not_null()
        | pl.col("eu_percentile").is_not_null()
        | pl.col("country_rank").is_not_null()
        | pl.col("country_percentile").is_not_null()
    )

    if aggregate_ranked.height > 0:
        fail(
            "At least one aggregate entity incorrectly received a "
            "peer rank or percentile."
        )

    pass_message(
        "Aggregate entity receives no institution rank or percentile."
    )

    # --------------------------------------------------------
    # Country suppression controls
    # --------------------------------------------------------

    invalid_singletons = benchmark.filter(
        pl.col("benchmark_eligible")
        &
        (pl.col("country_peer_count") == 1)
        &
        (
            pl.col("country_rank").is_not_null()
            |
            pl.col("country_percentile").is_not_null()
            |
            pl.col("country_mean_npe_ratio").is_not_null()
            |
            pl.col("country_median_npe_ratio").is_not_null()
        )
    )

    if invalid_singletons.height > 0:
        fail(
            "Singleton country peer groups received benchmark metrics."
        )

    pass_message(
        "Singleton country groups contain no country benchmark statistics."
    )

    invalid_small_group_ranks = benchmark.filter(
        pl.col("benchmark_eligible")
        &
        (pl.col("country_peer_count") >= 2)
        &
        (
            pl.col("country_peer_count")
            < COUNTRY_FORMAL_MIN_PEERS
        )
        &
        (
            pl.col("country_rank").is_not_null()
            |
            pl.col("country_percentile").is_not_null()
        )
    )

    if invalid_small_group_ranks.height > 0:
        fail(
            "Country groups below the formal peer threshold received "
            "rank or percentile values."
        )

    pass_message(
        "Country peer groups with 2-4 institutions contain descriptive "
        "statistics but no formal rank or percentile."
    )

    missing_formal_country_benchmarks = benchmark.filter(
        pl.col("benchmark_eligible")
        &
        (
            pl.col("country_peer_count")
            >= COUNTRY_FORMAL_MIN_PEERS
        )
        &
        (
            pl.col("country_rank").is_null()
            |
            pl.col("country_percentile").is_null()
            |
            pl.col("country_mean_npe_ratio").is_null()
            |
            pl.col("country_median_npe_ratio").is_null()
        )
    )

    if missing_formal_country_benchmarks.height > 0:
        fail(
            "At least one eligible formal country benchmark is incomplete."
        )

    pass_message(
        "All eligible country peer groups contain complete formal "
        "benchmark metrics."
    )

    # ========================================================
    # 10. SAVE BENCHMARK MART
    # ========================================================

    section("10. SAVE ANALYTICAL BENCHMARK MART")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        benchmark.write_parquet(OUTPUT_PATH)
    except Exception as exc:
        fail(f"Unable to write benchmark mart: {exc}")

    pass_message(
        "NPE benchmark mart written successfully."
    )

    print()
    print("Saved analytical benchmark mart:")
    print(f"  {OUTPUT_PATH}")

    # ========================================================
    # 11. POST-SAVE VALIDATION
    # ========================================================

    section("11. POST-SAVE VALIDATION")

    if not OUTPUT_PATH.exists():
        fail("Saved benchmark mart cannot be found.")

    pass_message(
        f"Saved file exists: {OUTPUT_PATH.name}"
    )

    try:
        saved = pl.read_parquet(OUTPUT_PATH)
    except Exception as exc:
        fail(f"Unable to reopen saved benchmark mart: {exc}")

    pass_message(
        f"Saved benchmark mart reopened: {saved.height:,} rows"
    )

    if saved.height != benchmark.height:
        fail("Saved row count does not reconcile.")

    pass_message(
        f"Saved row count reconciles: {saved.height:,}"
    )

    if saved.columns != benchmark.columns:
        fail("Saved benchmark column order changed.")

    pass_message("Saved benchmark column order reconciles.")

    benchmark_sorted = benchmark.sort(BUSINESS_GRAIN)
    saved_sorted = saved.sort(BUSINESS_GRAIN)

    if not benchmark_sorted.equals(saved_sorted):
        fail(
            "Saved Parquet does not exactly match the validated "
            "in-memory benchmark mart."
        )

    pass_message(
        "Saved Parquet exactly matches the validated in-memory mart."
    )

    validate_unique_grain(
        saved,
        BUSINESS_GRAIN,
        "saved mart_npe_benchmark",
    )

    # ========================================================
    # 12. FINAL BENCHMARK PROFILE
    # ========================================================

    section("12. FINAL NPE BENCHMARK MART PROFILE")

    final_profile = saved.select(
        [
            pl.len().alias("rows"),

            pl.col("lei_code")
            .n_unique()
            .alias("entities"),

            pl.col("period")
            .n_unique()
            .alias("periods"),

            pl.col("benchmark_eligible")
            .sum()
            .alias("eligible_rows"),

            pl.col("is_aggregate")
            .sum()
            .alias("aggregate_rows"),

            pl.col("eu_percentile")
            .null_count()
            .alias("null_eu_percentiles"),

            pl.col("country_rank")
            .is_not_null()
            .sum()
            .alias("formal_country_rank_rows"),
        ]
    )

    print(final_profile)

    print()
    print("Country benchmark status profile:")

    print(
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

    print()
    print("European benchmark period profile:")

    print(
        saved
        .filter(pl.col("benchmark_eligible"))
        .group_by("period")
        .agg(
            [
                pl.len().alias("peer_count"),

                pl.col("npe_ratio")
                .mean()
                .alias("mean_npe_ratio"),

                pl.col("npe_ratio")
                .median()
                .alias("median_npe_ratio"),

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

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    section("NPE BENCHMARK MART BUILD COMPLETED SUCCESSFULLY")

    print(f"Output: {OUTPUT_PATH}")

    print()
    print(
        "The approved NPE Ratio observations were preserved and "
        "augmented with the approved peer-benchmark methodology."
    )

    print(
        "The EBA aggregate remains available in the mart but was "
        "excluded from institution peer calculations."
    )

    print(
        "Country rank and percentile values were suppressed for "
        f"peer groups below {COUNTRY_FORMAL_MIN_PEERS} institutions."
    )

    print()
    print(
        "No source NPE Ratio values were modified."
    )

    print(
        "No regulatory threshold, management threshold, risk rating, "
        "or early-warning flag was applied."
    )

    print(
        "The benchmark mart has not yet received independent validation."
    )


if __name__ == "__main__":
    main()
