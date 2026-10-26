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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_benchmark.parquet"
)

MIN_FORMAL_COUNTRY_PEERS = 5


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
            f"{name} is not unique at grain lei_code x period. "
            f"Duplicate groups: {duplicates.height}"
        )


def add_average_rank(
    df: pl.DataFrame,
    partition_cols: list[str],
    value_col: str,
    output_col: str,
) -> pl.DataFrame:
    """
    Create ascending average ranks.

    Lower CET1 Ratio -> lower rank.
    Higher CET1 Ratio -> higher rank.
    Ties receive the average of their occupied rank positions.
    """

    row_number_col = f"__row_number_{output_col}"
    tie_count_col = f"__tie_count_{output_col}"
    tie_min_col = f"__tie_min_{output_col}"
    tie_max_col = f"__tie_max_{output_col}"

    ranked = (
        df
        .sort(
            partition_cols + [value_col, "lei_code"]
        )
        .with_columns(
            (
                pl.col(value_col)
                .rank(
                    method="ordinal",
                    descending=False,
                )
                .over(partition_cols)
            )
            .cast(pl.Float64)
            .alias(row_number_col)
        )
        .with_columns(
            [
                pl.len()
                .over(partition_cols + [value_col])
                .alias(tie_count_col),

                pl.col(row_number_col)
                .min()
                .over(partition_cols + [value_col])
                .alias(tie_min_col),

                pl.col(row_number_col)
                .max()
                .over(partition_cols + [value_col])
                .alias(tie_max_col),
            ]
        )
        .with_columns(
            (
                (
                    pl.col(tie_min_col)
                    + pl.col(tie_max_col)
                )
                / 2.0
            )
            .alias(output_col)
        )
        .drop(
            [
                row_number_col,
                tie_count_col,
                tie_min_col,
                tie_max_col,
            ]
        )
    )

    return ranked


def main() -> None:

    header("BUILD CET1 BENCHMARK ANALYTICAL MART")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source mart  : {SOURCE_PATH}")
    print(f"Output mart  : {OUTPUT_PATH}")
    print()
    print(
        "Methodology:"
    )
    print(
        "  - non-aggregate institutions are benchmark eligible"
    )
    print(
        "  - European benchmark by reporting period"
    )
    print(
        "  - ascending CET1 Ratio rank"
    )
    print(
        "  - average-rank tie treatment"
    )
    print(
        "  - percentile = (average_rank - 1) / (peer_count - 1)"
    )
    print(
        "  - country formal benchmark requires >=5 peers"
    )
    print(
        "  - country groups of 2-4 are descriptive only"
    )
    print(
        "  - singleton country groups are unavailable"
    )
    print(
        "  - source-valid CET1 Ratio observations >100% are retained"
    )
    print(
        "  - no regulatory thresholds or capital-strength flags"
    )

    # ------------------------------------------------------------------
    # 1. FILE AVAILABILITY
    # ------------------------------------------------------------------

    header("1. FILE AVAILABILITY")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Approved CET1 Ratio mart not found: {SOURCE_PATH}"
        )

    print(f"[PASS] Found: {SOURCE_PATH.name}")

    # ------------------------------------------------------------------
    # 2. LOAD SOURCE MART
    # ------------------------------------------------------------------

    header("2. LOAD APPROVED CET1 RATIO MART")

    source = pl.read_parquet(SOURCE_PATH)

    print(
        f"[PASS] Source opened: "
        f"{source.height:,} rows x {source.width} columns"
    )

    assert_unique_grain(
        source,
        "Source CET1 Ratio mart",
    )

    print(
        "[PASS] Source is unique at grain: "
        "lei_code x period."
    )

    missing_base_columns = [
        column
        for column in BASE_COLUMNS
        if column not in source.columns
    ]

    if missing_base_columns:
        raise ValueError(
            "Source mart is missing required columns: "
            f"{missing_base_columns}"
        )

    # ------------------------------------------------------------------
    # 3. DEFINE BENCHMARK ELIGIBILITY
    # ------------------------------------------------------------------

    header("3. DEFINE BENCHMARK ELIGIBILITY")

    working = source.with_columns(
        (
            ~pl.col("is_aggregate")
        ).alias("benchmark_eligible")
    )

    eligible = working.filter(
        pl.col("benchmark_eligible")
    )

    aggregate = working.filter(
        ~pl.col("benchmark_eligible")
    )

    print(
        f"Eligible institution rows : {eligible.height:,}"
    )
    print(
        f"Aggregate rows            : {aggregate.height:,}"
    )

    # ------------------------------------------------------------------
    # 4. EUROPEAN BENCHMARK STATISTICS
    # ------------------------------------------------------------------

    header("4. EUROPEAN BENCHMARK STATISTICS")

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
        .sort("period")
    )

    print(eu_stats)

    eligible = eligible.join(
        eu_stats,
        on="period",
        how="left",
    )

    eligible = add_average_rank(
        eligible,
        partition_cols=["period"],
        value_col="cet1_ratio",
        output_col="eu_rank",
    )

    eligible = eligible.with_columns(
        [
            pl.when(
                pl.col("eu_peer_count") > 1
            )
            .then(
                (
                    pl.col("eu_rank") - 1.0
                )
                /
                (
                    pl.col("eu_peer_count") - 1
                )
            )
            .otherwise(None)
            .alias("eu_percentile"),
        ]
    )

    eligible = eligible.with_columns(
        [
            (
                pl.col("eu_percentile") * 100
            ).alias("eu_percentile_pct"),

            (
                pl.col("cet1_ratio")
                - pl.col("eu_median_cet1_ratio")
            ).alias("eu_deviation_from_median"),
        ]
    )

    # ------------------------------------------------------------------
    # 5. COUNTRY BENCHMARK STATISTICS
    # ------------------------------------------------------------------

    header("5. COUNTRY BENCHMARK STATISTICS")

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

    print(
        country_stats
        .group_by("country_peer_count")
        .agg(
            pl.len().alias("country_period_groups")
        )
        .sort("country_peer_count")
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

    # ------------------------------------------------------------------
    # 6. COUNTRY AVAILABILITY STATUS
    # ------------------------------------------------------------------

    header("6. COUNTRY BENCHMARK AVAILABILITY")

    eligible = eligible.with_columns(
        [
            (
                pl.col("country_peer_count") >= 2
            ).alias("country_descriptive_available"),

            (
                pl.col("country_peer_count")
                >= MIN_FORMAL_COUNTRY_PEERS
            ).alias("country_benchmark_available"),

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
                    MIN_FORMAL_COUNTRY_PEERS - 1,
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

    # Singleton groups must not expose descriptive statistics.
    eligible = eligible.with_columns(
        [
            pl.when(
                pl.col("country_descriptive_available")
            )
            .then(
                pl.col("country_mean_cet1_ratio")
            )
            .otherwise(None)
            .alias("country_mean_cet1_ratio_final"),

            pl.when(
                pl.col("country_descriptive_available")
            )
            .then(
                pl.col("country_median_cet1_ratio")
            )
            .otherwise(None)
            .alias("country_median_cet1_ratio_final"),
        ]
    ).drop(
        [
            "country_mean_cet1_ratio",
            "country_median_cet1_ratio",
        ]
    ).rename(
        {
            "country_mean_cet1_ratio_final":
                "country_mean_cet1_ratio",
            "country_median_cet1_ratio_final":
                "country_median_cet1_ratio",
        }
    )

    # ------------------------------------------------------------------
    # 7. COUNTRY RANK
    # ------------------------------------------------------------------

    header("7. COUNTRY RANK AND PERCENTILE")

    country_ranked = add_average_rank(
        eligible,
        partition_cols=[
            "period",
            "institution_country_code",
        ],
        value_col="cet1_ratio",
        output_col="__country_rank_raw",
    )

    country_ranked = country_ranked.with_columns(
        [
            pl.when(
                pl.col("country_benchmark_available")
            )
            .then(
                pl.col("__country_rank_raw")
            )
            .otherwise(None)
            .alias("country_rank"),
        ]
    ).drop("__country_rank_raw")

    country_ranked = country_ranked.with_columns(
        [
            pl.when(
                pl.col("country_benchmark_available")
                & (
                    pl.col("country_peer_count") > 1
                )
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

    country_ranked = country_ranked.with_columns(
        [
            (
                pl.col("country_percentile") * 100
            ).alias("country_percentile_pct"),

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

    eligible = country_ranked

    # ------------------------------------------------------------------
    # 8. PREPARE AGGREGATE ROWS
    # ------------------------------------------------------------------

    header("8. PREPARE AGGREGATE ROWS")

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
    # 9. COMBINE FINAL MART
    # ------------------------------------------------------------------

    header("9. COMBINE FINAL BENCHMARK MART")

    final_columns = BASE_COLUMNS + [
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

    mart = (
        pl.concat(
            [
                eligible.select(final_columns),
                aggregate.select(final_columns),
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

    print(
        f"Rows    : {mart.height:,}"
    )
    print(
        f"Columns : {mart.width}"
    )

    # ------------------------------------------------------------------
    # 10. PRE-SAVE VALIDATION
    # ------------------------------------------------------------------

    header("10. PRE-SAVE BUSINESS VALIDATION")

    assert_unique_grain(
        mart,
        "CET1 benchmark mart",
    )

    print(
        "[PASS] Mart is unique at grain: "
        "lei_code x period."
    )

    if mart.height != source.height:
        raise ValueError(
            "Benchmark mart row count does not match "
            "source CET1 Ratio mart."
        )

    print(
        f"[PASS] Row count reconciles: "
        f"{mart.height:,}"
    )

    eligible_check = mart.filter(
        pl.col("benchmark_eligible")
    )

    aggregate_check = mart.filter(
        ~pl.col("benchmark_eligible")
    )

    if eligible_check.filter(
        pl.col("eu_rank").is_null()
    ).height > 0:
        raise ValueError(
            "Eligible institution rows contain NULL "
            "European ranks."
        )

    if eligible_check.filter(
        pl.col("eu_percentile").is_null()
    ).height > 0:
        raise ValueError(
            "Eligible institution rows contain NULL "
            "European percentiles."
        )

    print(
        "[PASS] All eligible institution rows have "
        "European rank and percentile."
    )

    formal_country = eligible_check.filter(
        pl.col("country_benchmark_available")
    )

    invalid_formal_country = formal_country.filter(
        pl.col("country_rank").is_null()
        | pl.col("country_percentile").is_null()
    )

    if invalid_formal_country.height > 0:
        raise ValueError(
            "Formal country benchmark rows contain "
            "missing rank or percentile."
        )

    print(
        "[PASS] Formal country benchmark rows have "
        "rank and percentile."
    )

    nonformal_country = eligible_check.filter(
        ~pl.col("country_benchmark_available")
    )

    improperly_ranked = nonformal_country.filter(
        pl.col("country_rank").is_not_null()
        | pl.col("country_percentile").is_not_null()
    )

    if improperly_ranked.height > 0:
        raise ValueError(
            "Country rank/percentile exists where formal "
            "country benchmark is unavailable."
        )

    print(
        "[PASS] Country rank and percentile are suppressed "
        "for groups below five peers."
    )

    singleton_invalid = eligible_check.filter(
        pl.col("country_peer_count") == 1
    ).filter(
        pl.col("country_mean_cet1_ratio").is_not_null()
        | pl.col("country_median_cet1_ratio").is_not_null()
    )

    if singleton_invalid.height > 0:
        raise ValueError(
            "Singleton country groups expose descriptive "
            "statistics unexpectedly."
        )

    print(
        "[PASS] Singleton country groups suppress descriptive "
        "country statistics."
    )

    aggregate_invalid = aggregate_check.filter(
        pl.col("eu_rank").is_not_null()
        | pl.col("eu_percentile").is_not_null()
        | pl.col("country_rank").is_not_null()
        | pl.col("country_percentile").is_not_null()
    )

    if aggregate_invalid.height > 0:
        raise ValueError(
            "Aggregate rows incorrectly contain peer ranks."
        )

    print(
        "[PASS] Aggregate rows are excluded from peer ranks."
    )

    # ------------------------------------------------------------------
    # 11. BENCHMARK STATUS PROFILE
    # ------------------------------------------------------------------

    header("11. BENCHMARK STATUS PROFILE")

    status_profile = (
        mart
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

    print()
    print("European period profile:")

    eu_period_profile = (
        mart
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

    print(eu_period_profile)

    print()
    print(
        "High CET1 Ratio observations retained:"
    )

    print(
        mart
        .filter(
            pl.col("cet1_ratio") > 1.0
        )
        .select(
            [
                "institution_name",
                "institution_country",
                "period",
                "cet1_ratio",
                "eu_rank",
                "eu_percentile",
                "country_benchmark_status",
                "country_rank",
                "country_percentile",
            ]
        )
        .sort(
            "cet1_ratio",
            descending=True,
        )
    )

    # ------------------------------------------------------------------
    # 12. SAVE MART
    # ------------------------------------------------------------------

    header("12. SAVE CET1 BENCHMARK MART")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mart.write_parquet(OUTPUT_PATH)

    print(
        f"[PASS] Saved: {OUTPUT_PATH}"
    )

    # ------------------------------------------------------------------
    # 13. POST-SAVE VALIDATION
    # ------------------------------------------------------------------

    header("13. POST-SAVE VALIDATION")

    saved = pl.read_parquet(OUTPUT_PATH)

    if saved.shape != mart.shape:
        raise ValueError(
            "Saved benchmark mart shape does not "
            "match in-memory mart."
        )

    print(
        f"[PASS] Saved mart reopened: "
        f"{saved.height:,} rows x "
        f"{saved.width} columns"
    )

    assert_unique_grain(
        saved,
        "Saved CET1 benchmark mart",
    )

    print(
        "[PASS] Saved benchmark mart remains unique "
        "at lei_code x period."
    )

    if saved.columns != final_columns:
        raise ValueError(
            "Saved benchmark mart column order does not "
            "match approved schema."
        )

    print(
        "[PASS] Saved benchmark column order reconciles."
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    header("CET1 BENCHMARK MART BUILD COMPLETE")

    print(
        f"Output  : {OUTPUT_PATH}"
    )
    print(
        f"Rows    : {saved.height:,}"
    )
    print(
        f"Columns : {saved.width}"
    )

    print()
    print(
        "European benchmark:"
    )
    print(
        "  - non-aggregate institutions only"
    )
    print(
        "  - ascending CET1 Ratio"
    )
    print(
        "  - average-rank ties"
    )
    print(
        "  - percentile = (rank - 1) / (peer_count - 1)"
    )

    print()
    print(
        "Country benchmark:"
    )
    print(
        "  - >=5 peers: formal rank / percentile"
    )
    print(
        "  - 2-4 peers: descriptive only"
    )
    print(
        "  - 1 peer: unavailable"
    )

    print()
    print(
        "No regulatory thresholds, capital-strength "
        "classifications, or risk flags were introduced."
    )

    print()
    print(
        "Source-valid CET1 Ratio observations above 100% "
        "were retained unchanged."
    )


if __name__ == "__main__":
    main()
