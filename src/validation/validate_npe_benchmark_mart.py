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

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_benchmark.parquet"
)

BUSINESS_GRAIN = [
    "lei_code",
    "period",
]

COUNTRY_DESCRIPTIVE_MIN_PEERS = 2
COUNTRY_FORMAL_MIN_PEERS = 5

FLOAT_TOLERANCE = 1e-12


BENCHMARK_COLUMNS = [
    "benchmark_eligible",
    "eu_peer_count",
    "eu_mean_npe_ratio",
    "eu_median_npe_ratio",
    "eu_rank",
    "eu_percentile",
    "eu_percentile_pct",
    "country_peer_count",
    "country_descriptive_available",
    "country_benchmark_available",
    "country_benchmark_status",
    "country_mean_npe_ratio",
    "country_median_npe_ratio",
    "country_rank",
    "country_percentile",
    "country_percentile_pct",
    "eu_deviation_from_median",
    "country_deviation_from_median",
]


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


EXPECTED_BENCHMARK_DTYPES = {
    "benchmark_eligible": pl.Boolean,
    "eu_peer_count": pl.UInt32,
    "eu_mean_npe_ratio": pl.Float64,
    "eu_median_npe_ratio": pl.Float64,
    "eu_rank": pl.Float64,
    "eu_percentile": pl.Float64,
    "eu_percentile_pct": pl.Float64,
    "country_peer_count": pl.UInt32,
    "country_descriptive_available": pl.Boolean,
    "country_benchmark_available": pl.Boolean,
    "country_benchmark_status": pl.String,
    "country_mean_npe_ratio": pl.Float64,
    "country_median_npe_ratio": pl.Float64,
    "country_rank": pl.Float64,
    "country_percentile": pl.Float64,
    "country_percentile_pct": pl.Float64,
    "eu_deviation_from_median": pl.Float64,
    "country_deviation_from_median": pl.Float64,
}


FLOAT_COLUMNS = [
    "eu_mean_npe_ratio",
    "eu_median_npe_ratio",
    "eu_rank",
    "eu_percentile",
    "eu_percentile_pct",
    "country_mean_npe_ratio",
    "country_median_npe_ratio",
    "country_rank",
    "country_percentile",
    "country_percentile_pct",
    "eu_deviation_from_median",
    "country_deviation_from_median",
]


EXACT_COLUMNS = [
    "benchmark_eligible",
    "eu_peer_count",
    "country_peer_count",
    "country_descriptive_available",
    "country_benchmark_available",
    "country_benchmark_status",
]


# ============================================================
# OUTPUT HELPERS
# ============================================================

def separator() -> None:
    print("=" * 100)


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
# GENERAL VALIDATION HELPERS
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
        print("Duplicate business-grain sample:")
        print(duplicates.head(20))

        fail(
            f"{object_name} is not unique at grain: "
            + " x ".join(grain)
        )

    pass_message(
        f"{object_name} is unique at grain: "
        + " x ".join(grain)
    )


def validate_exact_column(
    df: pl.DataFrame,
    actual_column: str,
    expected_column: str,
) -> None:
    mismatches = df.filter(
        (
            pl.col(actual_column).is_null()
            != pl.col(expected_column).is_null()
        )
        |
        (
            pl.col(actual_column).is_not_null()
            & pl.col(expected_column).is_not_null()
            & (
                pl.col(actual_column)
                != pl.col(expected_column)
            )
        )
    )

    if mismatches.height > 0:
        print()
        print(
            f"Mismatch sample for {actual_column}:"
        )

        print(
            mismatches
            .select(
                [
                    "lei_code",
                    "period",
                    actual_column,
                    expected_column,
                ]
            )
            .head(20)
        )

        fail(
            f"{actual_column} does not reconcile to the "
            "independent reconstruction."
        )

    pass_message(
        f"{actual_column} independently reconciles."
    )


def validate_float_column(
    df: pl.DataFrame,
    actual_column: str,
    expected_column: str,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    mismatches = df.filter(
        (
            pl.col(actual_column).is_null()
            != pl.col(expected_column).is_null()
        )
        |
        (
            pl.col(actual_column).is_not_null()
            & pl.col(expected_column).is_not_null()
            & (
                (
                    pl.col(actual_column)
                    - pl.col(expected_column)
                )
                .abs()
                > tolerance
            )
        )
    )

    if mismatches.height > 0:
        mismatch_output = (
            mismatches
            .with_columns(
                (
                    pl.col(actual_column)
                    - pl.col(expected_column)
                )
                .abs()
                .alias("absolute_difference")
            )
            .select(
                [
                    "lei_code",
                    "period",
                    actual_column,
                    expected_column,
                    "absolute_difference",
                ]
            )
            .sort(
                "absolute_difference",
                descending=True,
            )
            .head(20)
        )

        print()
        print(
            f"Mismatch sample for {actual_column}:"
        )
        print(mismatch_output)

        fail(
            f"{actual_column} does not reconcile within "
            f"tolerance {tolerance}."
        )

    pass_message(
        f"{actual_column} independently reconciles "
        f"within tolerance {tolerance}."
    )


# ============================================================
# INDEPENDENT EXPECTED-BENCHMARK CONSTRUCTION
# ============================================================

def reconstruct_expected_benchmark(
    source: pl.DataFrame,
) -> pl.DataFrame:

    eligible = source.filter(
        ~pl.col("is_aggregate")
    )

    aggregate = source.filter(
        pl.col("is_aggregate")
    )

    # --------------------------------------------------------
    # Independent European and country calculations
    # --------------------------------------------------------

    eligible_expected = (
        eligible
        .with_columns(
            [
                # European peer calculations
                pl.len()
                .over("period")
                .cast(pl.UInt32)
                .alias("expected_eu_peer_count"),

                pl.col("npe_ratio")
                .mean()
                .over("period")
                .cast(pl.Float64)
                .alias("expected_eu_mean_npe_ratio"),

                pl.col("npe_ratio")
                .median()
                .over("period")
                .cast(pl.Float64)
                .alias("expected_eu_median_npe_ratio"),

                pl.col("npe_ratio")
                .rank(
                    method="average",
                    descending=False,
                )
                .over("period")
                .cast(pl.Float64)
                .alias("expected_eu_rank"),

                # Country peer calculations
                pl.len()
                .over(
                    [
                        "period",
                        "institution_country_code",
                    ]
                )
                .cast(pl.UInt32)
                .alias("expected_country_peer_count"),

                pl.col("npe_ratio")
                .mean()
                .over(
                    [
                        "period",
                        "institution_country_code",
                    ]
                )
                .cast(pl.Float64)
                .alias("raw_country_mean"),

                pl.col("npe_ratio")
                .median()
                .over(
                    [
                        "period",
                        "institution_country_code",
                    ]
                )
                .cast(pl.Float64)
                .alias("raw_country_median"),

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
                .alias("raw_country_rank"),
            ]
        )
    )

    # --------------------------------------------------------
    # European percentile
    # --------------------------------------------------------

    eligible_expected = eligible_expected.with_columns(
        [
            (
                (
                    pl.col("expected_eu_rank") - 1.0
                )
                /
                (
                    pl.col("expected_eu_peer_count")
                    .cast(pl.Float64)
                    - 1.0
                )
            )
            .cast(pl.Float64)
            .alias("expected_eu_percentile")
        ]
    )

    eligible_expected = eligible_expected.with_columns(
        [
            (
                pl.col("expected_eu_percentile")
                * 100.0
            )
            .cast(pl.Float64)
            .alias("expected_eu_percentile_pct"),

            (
                pl.col("npe_ratio")
                - pl.col("expected_eu_median_npe_ratio")
            )
            .cast(pl.Float64)
            .alias("expected_eu_deviation_from_median"),
        ]
    )

    # --------------------------------------------------------
    # Country eligibility and statistics
    # --------------------------------------------------------

    eligible_expected = eligible_expected.with_columns(
        [
            (
                pl.col("expected_country_peer_count")
                >= COUNTRY_DESCRIPTIVE_MIN_PEERS
            )
            .alias(
                "expected_country_descriptive_available"
            ),

            (
                pl.col("expected_country_peer_count")
                >= COUNTRY_FORMAL_MIN_PEERS
            )
            .alias(
                "expected_country_benchmark_available"
            ),
        ]
    )

    eligible_expected = eligible_expected.with_columns(
        [
            pl.when(
                pl.col(
                    "expected_country_descriptive_available"
                )
            )
            .then(pl.col("raw_country_mean"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("expected_country_mean_npe_ratio"),

            pl.when(
                pl.col(
                    "expected_country_descriptive_available"
                )
            )
            .then(pl.col("raw_country_median"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("expected_country_median_npe_ratio"),

            pl.when(
                pl.col("expected_country_peer_count") == 1
            )
            .then(pl.lit("unavailable_singleton"))

            .when(
                (
                    pl.col("expected_country_peer_count")
                    >= COUNTRY_DESCRIPTIVE_MIN_PEERS
                )
                &
                (
                    pl.col("expected_country_peer_count")
                    < COUNTRY_FORMAL_MIN_PEERS
                )
            )
            .then(pl.lit("descriptive_only"))

            .otherwise(
                pl.lit("formal_benchmark_available")
            )
            .alias("expected_country_benchmark_status"),
        ]
    )

    # --------------------------------------------------------
    # Country formal rank / percentile
    # --------------------------------------------------------

    eligible_expected = eligible_expected.with_columns(
        [
            pl.when(
                pl.col("expected_country_peer_count")
                >= COUNTRY_FORMAL_MIN_PEERS
            )
            .then(pl.col("raw_country_rank"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("expected_country_rank"),

            pl.when(
                pl.col("expected_country_peer_count")
                >= COUNTRY_FORMAL_MIN_PEERS
            )
            .then(
                (
                    pl.col("raw_country_rank") - 1.0
                )
                /
                (
                    pl.col("expected_country_peer_count")
                    .cast(pl.Float64)
                    - 1.0
                )
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("expected_country_percentile"),
        ]
    )

    eligible_expected = eligible_expected.with_columns(
        [
            pl.when(
                pl.col("expected_country_percentile")
                .is_not_null()
            )
            .then(
                pl.col("expected_country_percentile")
                * 100.0
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("expected_country_percentile_pct"),

            pl.when(
                pl.col(
                    "expected_country_descriptive_available"
                )
            )
            .then(
                pl.col("npe_ratio")
                - pl.col(
                    "expected_country_median_npe_ratio"
                )
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias(
                "expected_country_deviation_from_median"
            ),
        ]
    )

    eligible_expected = (
        eligible_expected
        .with_columns(
            pl.lit(True)
            .alias("expected_benchmark_eligible")
        )
        .select(
            [
                "lei_code",
                "period",
                "expected_benchmark_eligible",
                "expected_eu_peer_count",
                "expected_eu_mean_npe_ratio",
                "expected_eu_median_npe_ratio",
                "expected_eu_rank",
                "expected_eu_percentile",
                "expected_eu_percentile_pct",
                "expected_country_peer_count",
                "expected_country_descriptive_available",
                "expected_country_benchmark_available",
                "expected_country_benchmark_status",
                "expected_country_mean_npe_ratio",
                "expected_country_median_npe_ratio",
                "expected_country_rank",
                "expected_country_percentile",
                "expected_country_percentile_pct",
                "expected_eu_deviation_from_median",
                "expected_country_deviation_from_median",
            ]
        )
    )

    # --------------------------------------------------------
    # Aggregate expected treatment
    # --------------------------------------------------------

    aggregate_expected = (
        aggregate
        .select(
            [
                "lei_code",
                "period",

                pl.lit(False)
                .cast(pl.Boolean)
                .alias("expected_benchmark_eligible"),

                pl.lit(None)
                .cast(pl.UInt32)
                .alias("expected_eu_peer_count"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_eu_mean_npe_ratio"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_eu_median_npe_ratio"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_eu_rank"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_eu_percentile"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_eu_percentile_pct"),

                pl.lit(None)
                .cast(pl.UInt32)
                .alias("expected_country_peer_count"),

                pl.lit(False)
                .cast(pl.Boolean)
                .alias(
                    "expected_country_descriptive_available"
                ),

                pl.lit(False)
                .cast(pl.Boolean)
                .alias(
                    "expected_country_benchmark_available"
                ),

                pl.lit("not_eligible_aggregate")
                .cast(pl.String)
                .alias(
                    "expected_country_benchmark_status"
                ),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_country_mean_npe_ratio"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias(
                    "expected_country_median_npe_ratio"
                ),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_country_rank"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_country_percentile"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("expected_country_percentile_pct"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias(
                    "expected_eu_deviation_from_median"
                ),

                pl.lit(None)
                .cast(pl.Float64)
                .alias(
                    "expected_country_deviation_from_median"
                ),
            ]
        )
    )

    return (
        pl.concat(
            [
                eligible_expected,
                aggregate_expected,
            ],
            how="vertical",
        )
        .sort(BUSINESS_GRAIN)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    separator()
    print("NPE BENCHMARK MART INDEPENDENT VALIDATION")
    print(
        "European Bank Risk Intelligence & "
        "Automated Reporting Platform"
    )
    separator()

    print()
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"NPE Ratio mart     : {SOURCE_PATH}")
    print(f"NPE Benchmark mart : {BENCHMARK_PATH}")

    print()
    print("VALIDATION MODE: READ ONLY")
    print(
        "No analytical, staging, reference, or raw data "
        "will be modified."
    )

    print()
    print("INDEPENDENT VALIDATION PRINCIPLE:")
    print(
        "Benchmark metrics are reconstructed directly from "
        "mart_npe_ratio.parquet."
    )
    print(
        "Saved benchmark metrics are not used to derive "
        "their own expected values."
    )

    # ========================================================
    # 1. FILE AVAILABILITY
    # ========================================================

    section("1. FILE AVAILABILITY")

    if not SOURCE_PATH.exists():
        fail(
            f"Approved NPE Ratio mart not found: {SOURCE_PATH}"
        )

    pass_message(
        f"Found source mart: {SOURCE_PATH.name}"
    )

    if not BENCHMARK_PATH.exists():
        fail(
            f"NPE benchmark mart not found: {BENCHMARK_PATH}"
        )

    pass_message(
        f"Found benchmark mart: {BENCHMARK_PATH.name}"
    )

    # ========================================================
    # 2. LOAD APPROVED ANALYTICAL DATA
    # ========================================================

    section("2. LOAD APPROVED ANALYTICAL DATA")

    try:
        source = pl.read_parquet(SOURCE_PATH)
    except Exception as exc:
        fail(
            f"Unable to open NPE Ratio source mart: {exc}"
        )

    try:
        benchmark = pl.read_parquet(BENCHMARK_PATH)
    except Exception as exc:
        fail(
            f"Unable to open NPE benchmark mart: {exc}"
        )

    pass_message(
        f"NPE Ratio source opened: {source.height:,} rows"
    )

    pass_message(
        f"NPE benchmark mart opened: "
        f"{benchmark.height:,} rows"
    )

    # ========================================================
    # 3. SOURCE AND MART STRUCTURE VALIDATION
    # ========================================================

    section("3. SOURCE AND MART STRUCTURE VALIDATION")

    require_columns(
        source,
        REQUIRED_SOURCE_COLUMNS,
        "mart_npe_ratio",
    )

    pass_message(
        "All required NPE Ratio source columns are present."
    )

    require_columns(
        benchmark,
        source.columns + BENCHMARK_COLUMNS,
        "mart_npe_benchmark",
    )

    pass_message(
        "All required benchmark mart columns are present."
    )

    expected_column_order = (
        source.columns
        + BENCHMARK_COLUMNS
    )

    if benchmark.columns != expected_column_order:
        print()
        print("Expected column order:")
        print(expected_column_order)

        print()
        print("Actual column order:")
        print(benchmark.columns)

        fail(
            "Benchmark mart columns do not match the "
            "approved design and order."
        )

    pass_message(
        "Benchmark mart columns match the approved design "
        "and order."
    )

    if source.height != benchmark.height:
        fail(
            "Benchmark mart row count does not reconcile "
            "to the NPE Ratio source mart."
        )

    pass_message(
        f"Row count reconciles: {benchmark.height:,} rows."
    )

    validate_unique_grain(
        source,
        BUSINESS_GRAIN,
        "mart_npe_ratio",
    )

    validate_unique_grain(
        benchmark,
        BUSINESS_GRAIN,
        "mart_npe_benchmark",
    )

    # ========================================================
    # 4. BENCHMARK COLUMN DATA TYPES
    # ========================================================

    section("4. BENCHMARK COLUMN DATA-TYPE VALIDATION")

    schema = benchmark.schema

    for (
        column_name,
        expected_dtype,
    ) in EXPECTED_BENCHMARK_DTYPES.items():

        actual_dtype = schema[column_name]

        if actual_dtype != expected_dtype:
            fail(
                f"{column_name} dtype is {actual_dtype}; "
                f"expected {expected_dtype}."
            )

        pass_message(
            f"{column_name} dtype is {expected_dtype}."
        )

    # ========================================================
    # 5. SOURCE FIELD PRESERVATION
    # ========================================================

    section("5. SOURCE FIELD PRESERVATION")

    source_projection = (
        source
        .select(source.columns)
        .sort(BUSINESS_GRAIN)
    )

    benchmark_source_projection = (
        benchmark
        .select(source.columns)
        .sort(BUSINESS_GRAIN)
    )

    if not source_projection.equals(
        benchmark_source_projection
    ):
        fail(
            "Original NPE Ratio source fields are not "
            "preserved exactly in the benchmark mart."
        )

    pass_message(
        "All original NPE Ratio mart fields are preserved "
        "exactly."
    )

    # ========================================================
    # 6. INDEPENDENT BENCHMARK RECONSTRUCTION
    # ========================================================

    section("6. INDEPENDENT BENCHMARK RECONSTRUCTION")

    expected = reconstruct_expected_benchmark(
        source
    )

    if expected.height != source.height:
        fail(
            "Independent reconstruction row count does not "
            "reconcile to source."
        )

    pass_message(
        f"Independent reconstruction created "
        f"{expected.height:,} expected observations."
    )

    validate_unique_grain(
        expected,
        BUSINESS_GRAIN,
        "independent benchmark reconstruction",
    )

    eligible_expected = expected.filter(
        pl.col("expected_benchmark_eligible")
    )

    aggregate_expected = expected.filter(
        ~pl.col("expected_benchmark_eligible")
    )

    info(
        f"Expected eligible observations : "
        f"{eligible_expected.height:,}"
    )

    info(
        f"Expected aggregate observations: "
        f"{aggregate_expected.height:,}"
    )

    # ========================================================
    # 7. TIE-HANDLING PROFILE
    # ========================================================

    section("7. EUROPEAN TIE-HANDLING PROFILE")

    eligible_source = source.filter(
        ~pl.col("is_aggregate")
    )

    tie_groups = (
        eligible_source
        .group_by(
            [
                "period",
                "npe_ratio",
            ]
        )
        .agg(
            pl.len()
            .cast(pl.UInt32)
            .alias("tie_size")
        )
        .filter(
            pl.col("tie_size") > 1
        )
        .sort(
            [
                "period",
                "tie_size",
            ],
            descending=[
                False,
                True,
            ],
        )
    )

    tied_observations = (
        tie_groups
        .select(
            pl.col("tie_size")
            .sum()
            .alias("tied_observations")
        )
        .item()
        if tie_groups.height > 0
        else 0
    )

    info(
        f"European tied value groups : "
        f"{tie_groups.height:,}"
    )

    info(
        f"Observations participating in ties: "
        f"{tied_observations:,}"
    )

    if tie_groups.height > 0:
        print()
        print("Tie-group sample:")
        print(
            tie_groups.head(20)
        )

    pass_message(
        "Independent reconstruction uses average rank "
        "for tied NPE Ratio observations."
    )

    # ========================================================
    # 8. JOIN SAVED MART TO INDEPENDENT EXPECTATION
    # ========================================================

    section(
        "8. SAVED MART TO INDEPENDENT RECONSTRUCTION ALIGNMENT"
    )

    comparison = (
        benchmark
        .join(
            expected,
            on=BUSINESS_GRAIN,
            how="inner",
        )
        .sort(BUSINESS_GRAIN)
    )

    if comparison.height != benchmark.height:
        fail(
            "Saved benchmark observations do not align "
            "one-to-one with the independent reconstruction."
        )

    pass_message(
        "Saved benchmark observations align one-to-one "
        "with the independent reconstruction."
    )

    # ========================================================
    # 9. EXACT BENCHMARK CONTROL RECONCILIATION
    # ========================================================

    section(
        "9. EXACT BENCHMARK CONTROL RECONCILIATION"
    )

    exact_mapping = {
        "benchmark_eligible":
            "expected_benchmark_eligible",

        "eu_peer_count":
            "expected_eu_peer_count",

        "country_peer_count":
            "expected_country_peer_count",

        "country_descriptive_available":
            "expected_country_descriptive_available",

        "country_benchmark_available":
            "expected_country_benchmark_available",

        "country_benchmark_status":
            "expected_country_benchmark_status",
    }

    for (
        actual_column,
        expected_column,
    ) in exact_mapping.items():

        validate_exact_column(
            comparison,
            actual_column,
            expected_column,
        )

    # ========================================================
    # 10. FLOATING-POINT BENCHMARK RECONCILIATION
    # ========================================================

    section(
        "10. NUMERICAL BENCHMARK RECONCILIATION"
    )

    float_mapping = {
        "eu_mean_npe_ratio":
            "expected_eu_mean_npe_ratio",

        "eu_median_npe_ratio":
            "expected_eu_median_npe_ratio",

        "eu_rank":
            "expected_eu_rank",

        "eu_percentile":
            "expected_eu_percentile",

        "eu_percentile_pct":
            "expected_eu_percentile_pct",

        "country_mean_npe_ratio":
            "expected_country_mean_npe_ratio",

        "country_median_npe_ratio":
            "expected_country_median_npe_ratio",

        "country_rank":
            "expected_country_rank",

        "country_percentile":
            "expected_country_percentile",

        "country_percentile_pct":
            "expected_country_percentile_pct",

        "eu_deviation_from_median":
            "expected_eu_deviation_from_median",

        "country_deviation_from_median":
            "expected_country_deviation_from_median",
    }

    for (
        actual_column,
        expected_column,
    ) in float_mapping.items():

        validate_float_column(
            comparison,
            actual_column,
            expected_column,
        )

    pass_message(
        "All 18 benchmark fields independently reconcile "
        "to the approved NPE Ratio source."
    )

    # ========================================================
    # 11. EUROPEAN BENCHMARK BUSINESS RULES
    # ========================================================

    section("11. EUROPEAN BENCHMARK BUSINESS-RULE VALIDATION")

    eligible_benchmark = benchmark.filter(
        pl.col("benchmark_eligible")
    )

    aggregate_benchmark = benchmark.filter(
        ~pl.col("benchmark_eligible")
    )

    invalid_eu_percentiles = (
        eligible_benchmark
        .filter(
            pl.col("eu_percentile").is_null()
            |
            (pl.col("eu_percentile") < 0.0)
            |
            (pl.col("eu_percentile") > 1.0)
        )
    )

    if invalid_eu_percentiles.height > 0:
        fail(
            "Eligible institution contains invalid "
            "European percentile."
        )

    pass_message(
        "All eligible European percentiles lie within [0, 1]."
    )

    invalid_eu_pct = (
        eligible_benchmark
        .filter(
            (
                pl.col("eu_percentile_pct")
                - (
                    pl.col("eu_percentile")
                    * 100.0
                )
            )
            .abs()
            > FLOAT_TOLERANCE
        )
    )

    if invalid_eu_pct.height > 0:
        fail(
            "European percentile percentage does not "
            "reconcile to percentile × 100."
        )

    pass_message(
        "European percentile percentage reconciles to "
        "percentile × 100."
    )

    invalid_aggregate_eu = (
        aggregate_benchmark
        .filter(
            pl.col("eu_peer_count").is_not_null()
            |
            pl.col("eu_mean_npe_ratio").is_not_null()
            |
            pl.col("eu_median_npe_ratio").is_not_null()
            |
            pl.col("eu_rank").is_not_null()
            |
            pl.col("eu_percentile").is_not_null()
            |
            pl.col("eu_percentile_pct").is_not_null()
            |
            pl.col("eu_deviation_from_median").is_not_null()
        )
    )

    if invalid_aggregate_eu.height > 0:
        fail(
            "Aggregate entity incorrectly received "
            "European benchmark metrics."
        )

    pass_message(
        "Aggregate entity is excluded from European "
        "peer benchmarking."
    )

    # ========================================================
    # 12. COUNTRY BENCHMARK BUSINESS RULES
    # ========================================================

    section("12. COUNTRY BENCHMARK BUSINESS-RULE VALIDATION")

    singleton_rows = (
        benchmark
        .filter(
            pl.col("benchmark_eligible")
            &
            (pl.col("country_peer_count") == 1)
        )
    )

    invalid_singleton = singleton_rows.filter(
        pl.col("country_descriptive_available")
        |
        pl.col("country_benchmark_available")
        |
        pl.col("country_mean_npe_ratio").is_not_null()
        |
        pl.col("country_median_npe_ratio").is_not_null()
        |
        pl.col("country_rank").is_not_null()
        |
        pl.col("country_percentile").is_not_null()
        |
        pl.col("country_percentile_pct").is_not_null()
        |
        pl.col(
            "country_deviation_from_median"
        ).is_not_null()
        |
        (
            pl.col("country_benchmark_status")
            != "unavailable_singleton"
        )
    )

    if invalid_singleton.height > 0:
        fail(
            "Singleton country groups violate the approved "
            "suppression methodology."
        )

    pass_message(
        "Singleton country groups correctly suppress all "
        "country benchmark statistics."
    )

    descriptive_only_rows = (
        benchmark
        .filter(
            pl.col("benchmark_eligible")
            &
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
    )

    invalid_descriptive_only = (
        descriptive_only_rows
        .filter(
            ~pl.col("country_descriptive_available")
            |
            pl.col("country_benchmark_available")
            |
            pl.col("country_mean_npe_ratio").is_null()
            |
            pl.col("country_median_npe_ratio").is_null()
            |
            pl.col("country_rank").is_not_null()
            |
            pl.col("country_percentile").is_not_null()
            |
            pl.col("country_percentile_pct").is_not_null()
            |
            (
                pl.col("country_benchmark_status")
                != "descriptive_only"
            )
        )
    )

    if invalid_descriptive_only.height > 0:
        fail(
            "Country peer groups containing 2-4 institutions "
            "violate the approved descriptive-only methodology."
        )

    pass_message(
        "Country peer groups containing 2-4 institutions "
        "provide descriptive statistics but suppress "
        "formal rank and percentile."
    )

    formal_country_rows = (
        benchmark
        .filter(
            pl.col("benchmark_eligible")
            &
            (
                pl.col("country_peer_count")
                >= COUNTRY_FORMAL_MIN_PEERS
            )
        )
    )

    invalid_formal = formal_country_rows.filter(
        ~pl.col("country_descriptive_available")
        |
        ~pl.col("country_benchmark_available")
        |
        pl.col("country_mean_npe_ratio").is_null()
        |
        pl.col("country_median_npe_ratio").is_null()
        |
        pl.col("country_rank").is_null()
        |
        pl.col("country_percentile").is_null()
        |
        pl.col("country_percentile_pct").is_null()
        |
        (
            pl.col("country_benchmark_status")
            != "formal_benchmark_available"
        )
    )

    if invalid_formal.height > 0:
        fail(
            "Formal country benchmark observations are incomplete."
        )

    pass_message(
        f"Country peer groups containing at least "
        f"{COUNTRY_FORMAL_MIN_PEERS} institutions have "
        f"complete formal benchmark metrics."
    )

    invalid_country_percentiles = (
        formal_country_rows
        .filter(
            (pl.col("country_percentile") < 0.0)
            |
            (pl.col("country_percentile") > 1.0)
        )
    )

    if invalid_country_percentiles.height > 0:
        fail(
            "Formal country percentile lies outside [0, 1]."
        )

    pass_message(
        "All formal country percentiles lie within [0, 1]."
    )

    aggregate_country_invalid = (
        aggregate_benchmark
        .filter(
            pl.col("country_peer_count").is_not_null()
            |
            pl.col("country_descriptive_available")
            |
            pl.col("country_benchmark_available")
            |
            (
                pl.col("country_benchmark_status")
                != "not_eligible_aggregate"
            )
            |
            pl.col("country_mean_npe_ratio").is_not_null()
            |
            pl.col("country_median_npe_ratio").is_not_null()
            |
            pl.col("country_rank").is_not_null()
            |
            pl.col("country_percentile").is_not_null()
            |
            pl.col("country_percentile_pct").is_not_null()
            |
            pl.col(
                "country_deviation_from_median"
            ).is_not_null()
        )
    )

    if aggregate_country_invalid.height > 0:
        fail(
            "Aggregate entity incorrectly received country "
            "benchmark metrics."
        )

    pass_message(
        "Aggregate entity is excluded from country "
        "peer benchmarking."
    )

    # ========================================================
    # 13. BENCHMARK POPULATION PROFILE
    # ========================================================

    section("13. BENCHMARK POPULATION PROFILE")

    profile = benchmark.select(
        [
            pl.len()
            .cast(pl.UInt32)
            .alias("rows"),

            pl.col("lei_code")
            .n_unique()
            .cast(pl.UInt32)
            .alias("entities"),

            pl.col("period")
            .n_unique()
            .cast(pl.UInt32)
            .alias("periods"),

            pl.col("benchmark_eligible")
            .sum()
            .cast(pl.UInt32)
            .alias("eligible_rows"),

            pl.col("is_aggregate")
            .sum()
            .cast(pl.UInt32)
            .alias("aggregate_rows"),

            pl.col("eu_rank")
            .is_not_null()
            .sum()
            .cast(pl.UInt32)
            .alias("eu_ranked_rows"),

            pl.col("country_rank")
            .is_not_null()
            .sum()
            .cast(pl.UInt32)
            .alias("formal_country_rank_rows"),
        ]
    )

    print(profile)

    print()
    print("Country benchmark status profile:")

    country_status_profile = (
        benchmark
        .group_by("country_benchmark_status")
        .agg(
            [
                pl.len()
                .cast(pl.UInt32)
                .alias("rows"),

                pl.col("lei_code")
                .n_unique()
                .cast(pl.UInt32)
                .alias("entities"),
            ]
        )
        .sort("country_benchmark_status")
    )

    print(country_status_profile)

    print()
    print("European period benchmark profile:")

    period_profile = (
        benchmark
        .filter(
            pl.col("benchmark_eligible")
        )
        .group_by("period")
        .agg(
            [
                pl.len()
                .cast(pl.UInt32)
                .alias("peer_count"),

                pl.col("npe_ratio")
                .mean()
                .alias("mean_npe_ratio"),

                pl.col("npe_ratio")
                .median()
                .alias("median_npe_ratio"),

                pl.col("eu_rank")
                .min()
                .alias("minimum_rank"),

                pl.col("eu_rank")
                .max()
                .alias("maximum_rank"),

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

    # ========================================================
    # 14. FINAL VALIDATION RESULT
    # ========================================================

    section(
        "NPE BENCHMARK MART INDEPENDENT VALIDATION "
        "COMPLETED SUCCESSFULLY"
    )

    print("OVERALL STATUS : PASS")

    print()
    print(
        "INTERPRETATION : The saved NPE benchmark mart "
        "independently reconciles to the approved NPE Ratio "
        "mart and the documented benchmarking methodology."
    )

    print()
    print(
        "Validated controls include:"
    )
    print(
        "  - source observation preservation"
    )
    print(
        "  - institution-period business grain"
    )
    print(
        "  - benchmark eligibility"
    )
    print(
        "  - European peer counts"
    )
    print(
        "  - European mean and median"
    )
    print(
        "  - average-rank tie handling"
    )
    print(
        "  - European percentile formula"
    )
    print(
        "  - country peer counts"
    )
    print(
        "  - singleton suppression"
    )
    print(
        "  - descriptive-only country groups"
    )
    print(
        "  - formal country benchmark threshold"
    )
    print(
        "  - country mean and median"
    )
    print(
        "  - country rank and percentile"
    )
    print(
        "  - deviation-from-median calculations"
    )
    print(
        "  - aggregate exclusion"
    )

    print()
    print(
        "ACTION        : mart_npe_benchmark can be considered "
        "for approval for downstream trend analysis, "
        "statistical analysis, and management reporting."
    )

    print()
    print(
        "No thresholds, risk classifications, or early-warning "
        "flags were independently approved by this validation."
    )

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No data was modified.")


if __name__ == "__main__":
    main()
