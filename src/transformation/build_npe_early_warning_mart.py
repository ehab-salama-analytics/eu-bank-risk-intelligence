from pathlib import Path
import sys

import polars as pl


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_trend.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_npe_early_warning.parquet"
)


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
    "qoq_available",
    "npe_ratio_change_qoq",
    "npe_ratio_pp_change_qoq",
    "eu_percentile",
    "eu_percentile_qoq_available",
    "eu_percentile_change_qoq",
    "country_benchmark_status",
    "country_percentile",
    "country_percentile_qoq_available",
    "country_percentile_change_qoq",
    "three_period_pattern_available",
    "npe_three_period_pattern",
]


NEW_EARLY_WARNING_COLUMNS = [
    "early_warning_eligible",
    "npe_ratio_deterioration",
    "eu_peer_deterioration",
    "country_peer_deterioration_available",
    "country_peer_deterioration",
    "persistent_deterioration_available",
    "persistent_deterioration",
    "supporting_signal_count",
    "early_warning_status",
]


ALLOWED_STATUSES = [
    "NOT_ELIGIBLE_AGGREGATE",
    "NOT_ASSESSABLE",
    "NO_CURRENT_DETERIORATION_SIGNAL",
    "WATCH",
    "ELEVATED_ATTENTION",
    "STRONG_ATTENTION",
]


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def pass_check(message: str) -> None:
    print(f"[PASS] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 92)
    print("BUILDING ANALYTICAL NPE EARLY-WARNING MART")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    print("=" * 92)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source       : {SOURCE_PATH}")
    print(f"Output       : {OUTPUT_PATH}")

    print()
    print("METHODOLOGY:")
    print("  - institution-level monitoring only")
    print("  - aggregate analytical entity excluded")
    print("  - direct NPE Ratio deterioration is the primary gate")
    print("  - European peer deterioration is supporting evidence")
    print("  - country peer deterioration is used only when available")
    print("  - persistent deterioration requires consecutive_increase")
    print("  - unavailable evidence remains unavailable")
    print("  - no arbitrary NPE threshold")
    print("  - no regulatory risk classification")
    print("  - no predictive score")
    print("  - no traffic-light classification")

    # =========================================================================
    # 1. LOAD VALIDATED NPE TREND MART
    # =========================================================================

    section("1. LOAD VALIDATED NPE TREND MART")

    if not SOURCE_PATH.exists():
        fail(f"Source mart not found: {SOURCE_PATH}")

    pass_check(f"Source mart exists: {SOURCE_PATH.name}")

    try:
        source = pl.read_parquet(SOURCE_PATH)
    except Exception as exc:
        fail(f"Could not open source mart: {exc}")

    pass_check(
        f"NPE trend mart opened successfully: "
        f"{source.height:,} rows, {source.width} columns"
    )

    # =========================================================================
    # 2. SOURCE STRUCTURE VALIDATION
    # =========================================================================

    section("2. SOURCE STRUCTURE VALIDATION")

    missing_columns = [
        column
        for column in REQUIRED_SOURCE_COLUMNS
        if column not in source.columns
    ]

    if missing_columns:
        fail(
            "Required source columns are missing: "
            + ", ".join(missing_columns)
        )

    pass_check("All required early-warning source fields are present.")

    duplicate_source_grain = (
        source
        .group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicate_source_grain.height > 0:
        print(duplicate_source_grain)
        fail("Source is not unique at grain: lei_code x period")

    pass_check("Source is unique at grain: lei_code x period.")

    key_nulls = source.select(
        pl.col("lei_code").is_null().sum().alias("null_lei_code"),
        pl.col("period").is_null().sum().alias("null_period"),
    )

    if (
        key_nulls["null_lei_code"][0] > 0
        or key_nulls["null_period"][0] > 0
    ):
        print(key_nulls)
        fail("Business grain contains NULL values.")

    pass_check("Business grain contains no NULL values.")

    source_rows = source.height
    source_columns = source.width

    info(f"Source rows             : {source_rows:,}")
    info(f"Source columns          : {source_columns}")
    info(
        f"Distinct entities       : "
        f"{source.select(pl.col('lei_code').n_unique()).item():,}"
    )
    info(
        f"Distinct periods        : "
        f"{source.select(pl.col('period').n_unique()).item():,}"
    )

    # =========================================================================
    # 3. DEFINE EARLY-WARNING ELIGIBILITY
    # =========================================================================

    section("3. DEFINE EARLY-WARNING ELIGIBILITY")

    mart = source.with_columns(
        (~pl.col("is_aggregate"))
        .alias("early_warning_eligible")
    )

    eligibility_profile = mart.select(
        pl.len().alias("rows"),
        pl.col("early_warning_eligible")
        .sum()
        .alias("eligible_rows"),
        pl.col("is_aggregate")
        .sum()
        .alias("aggregate_rows"),
    )

    print(eligibility_profile)

    aggregate_eligible = mart.filter(
        pl.col("is_aggregate")
        & pl.col("early_warning_eligible")
    )

    if aggregate_eligible.height > 0:
        fail(
            "Aggregate analytical entities were incorrectly marked "
            "as early-warning eligible."
        )

    pass_check(
        "Aggregate analytical entities are excluded from "
        "institution-level early-warning monitoring."
    )

    # =========================================================================
    # 4. BUILD PRIMARY NPE DETERIORATION SIGNAL
    # =========================================================================

    section("4. BUILD PRIMARY NPE RATIO DETERIORATION SIGNAL")

    mart = mart.with_columns(
        pl.when(
            pl.col("early_warning_eligible")
            & pl.col("qoq_available")
        )
        .then(pl.col("npe_ratio_change_qoq") > 0)
        .otherwise(None)
        .cast(pl.Boolean)
        .alias("npe_ratio_deterioration")
    )

    primary_profile = mart.select(
        pl.col("npe_ratio_deterioration")
        .is_not_null()
        .sum()
        .alias("assessable_rows"),

        (
            pl.col("npe_ratio_deterioration") == True
        )
        .sum()
        .alias("deterioration_rows"),

        (
            pl.col("npe_ratio_deterioration") == False
        )
        .sum()
        .alias("non_deterioration_rows"),

        pl.col("npe_ratio_deterioration")
        .is_null()
        .sum()
        .alias("not_assessable_rows"),
    )

    print(primary_profile)

    pass_check(
        "Primary NPE Ratio deterioration signal created using "
        "genuine QoQ observations only."
    )

    # =========================================================================
    # 5. BUILD EUROPEAN PEER DETERIORATION SIGNAL
    # =========================================================================

    section("5. BUILD EUROPEAN PEER DETERIORATION SIGNAL")

    mart = mart.with_columns(
        pl.when(
            pl.col("early_warning_eligible")
            & pl.col("eu_percentile_qoq_available")
        )
        .then(pl.col("eu_percentile_change_qoq") > 0)
        .otherwise(None)
        .cast(pl.Boolean)
        .alias("eu_peer_deterioration")
    )

    eu_profile = mart.select(
        pl.col("eu_peer_deterioration")
        .is_not_null()
        .sum()
        .alias("assessable_rows"),

        (
            pl.col("eu_peer_deterioration") == True
        )
        .sum()
        .alias("deterioration_rows"),

        (
            pl.col("eu_peer_deterioration") == False
        )
        .sum()
        .alias("non_deterioration_rows"),
    )

    print(eu_profile)

    pass_check(
        "European peer deterioration signal created only where "
        "consecutive-quarter percentile comparison is available."
    )

    # =========================================================================
    # 6. BUILD COUNTRY PEER DETERIORATION SIGNAL
    # =========================================================================

    section("6. BUILD COUNTRY PEER DETERIORATION SIGNAL")

    mart = mart.with_columns(
        (
            pl.col("early_warning_eligible")
            & pl.col("country_percentile_qoq_available")
        )
        .alias("country_peer_deterioration_available")
    )

    mart = mart.with_columns(
        pl.when(
            pl.col("country_peer_deterioration_available")
        )
        .then(pl.col("country_percentile_change_qoq") > 0)
        .otherwise(None)
        .cast(pl.Boolean)
        .alias("country_peer_deterioration")
    )

    country_profile = mart.select(
        pl.col("country_peer_deterioration_available")
        .sum()
        .alias("available_rows"),

        (
            pl.col("country_peer_deterioration") == True
        )
        .sum()
        .alias("deterioration_rows"),

        (
            pl.col("country_peer_deterioration") == False
        )
        .sum()
        .alias("non_deterioration_rows"),

        pl.col("country_peer_deterioration")
        .is_null()
        .sum()
        .alias("unavailable_rows"),
    )

    print(country_profile)

    invalid_country_signal = mart.filter(
        (~pl.col("country_peer_deterioration_available"))
        & pl.col("country_peer_deterioration").is_not_null()
    )

    if invalid_country_signal.height > 0:
        fail(
            "Country deterioration was populated where the "
            "country comparison is unavailable."
        )

    pass_check(
        "Unavailable country benchmarks remain NOT ASSESSABLE "
        "rather than being converted to FALSE."
    )

    # =========================================================================
    # 7. BUILD PERSISTENT DETERIORATION SIGNAL
    # =========================================================================

    section("7. BUILD PERSISTENT DETERIORATION SIGNAL")

    mart = mart.with_columns(
        (
            pl.col("early_warning_eligible")
            & pl.col("three_period_pattern_available")
        )
        .alias("persistent_deterioration_available")
    )

    mart = mart.with_columns(
        pl.when(
            pl.col("persistent_deterioration_available")
        )
        .then(
            pl.col("npe_three_period_pattern")
            == "consecutive_increase"
        )
        .otherwise(None)
        .cast(pl.Boolean)
        .alias("persistent_deterioration")
    )

    persistence_profile = mart.select(
        pl.col("persistent_deterioration_available")
        .sum()
        .alias("available_rows"),

        (
            pl.col("persistent_deterioration") == True
        )
        .sum()
        .alias("persistent_deterioration_rows"),

        (
            pl.col("persistent_deterioration") == False
        )
        .sum()
        .alias("other_pattern_rows"),

        pl.col("persistent_deterioration")
        .is_null()
        .sum()
        .alias("unavailable_rows"),
    )

    print(persistence_profile)

    invalid_persistence = mart.filter(
        (~pl.col("persistent_deterioration_available"))
        & pl.col("persistent_deterioration").is_not_null()
    )

    if invalid_persistence.height > 0:
        fail(
            "Persistent deterioration was populated where "
            "three-period history is unavailable."
        )

    pass_check(
        "Persistent deterioration is identified only from "
        "approved consecutive_increase patterns."
    )

    # =========================================================================
    # 8. SUPPORTING SIGNAL COUNT
    # =========================================================================

    section("8. BUILD SUPPORTING SIGNAL COUNT")

    mart = mart.with_columns(
        pl.when(pl.col("early_warning_eligible"))
        .then(
            pl.sum_horizontal(
                [
                    pl.col("eu_peer_deterioration")
                    .fill_null(False)
                    .cast(pl.Int8),

                    pl.col("country_peer_deterioration")
                    .fill_null(False)
                    .cast(pl.Int8),

                    pl.col("persistent_deterioration")
                    .fill_null(False)
                    .cast(pl.Int8),
                ]
            )
        )
        .otherwise(None)
        .cast(pl.Int8)
        .alias("supporting_signal_count")
    )

    supporting_profile = (
        mart
        .filter(pl.col("early_warning_eligible"))
        .group_by("supporting_signal_count")
        .agg(pl.len().alias("rows"))
        .sort("supporting_signal_count")
    )

    print(supporting_profile)

    invalid_support_count = mart.filter(
        pl.col("early_warning_eligible")
        & (
            (pl.col("supporting_signal_count") < 0)
            | (pl.col("supporting_signal_count") > 3)
        )
    )

    if invalid_support_count.height > 0:
        fail("Supporting signal count falls outside expected range 0–3.")

    pass_check("Supporting signal count lies within expected range 0–3.")

    # =========================================================================
    # 9. ASSIGN EARLY-WARNING STATUS
    # =========================================================================

    section("9. ASSIGN EARLY-WARNING MONITORING STATUS")

    any_peer_deterioration = (
        pl.col("eu_peer_deterioration")
        .fill_null(False)
        |
        pl.col("country_peer_deterioration")
        .fill_null(False)
    )

    any_supporting_deterioration = (
        pl.col("eu_peer_deterioration")
        .fill_null(False)
        |
        pl.col("country_peer_deterioration")
        .fill_null(False)
        |
        pl.col("persistent_deterioration")
        .fill_null(False)
    )

    mart = mart.with_columns(
        pl.when(~pl.col("early_warning_eligible"))
        .then(pl.lit("NOT_ELIGIBLE_AGGREGATE"))

        .when(
            pl.col("early_warning_eligible")
            & (~pl.col("qoq_available"))
        )
        .then(pl.lit("NOT_ASSESSABLE"))

        .when(
            pl.col("early_warning_eligible")
            & pl.col("qoq_available")
            & (pl.col("npe_ratio_change_qoq") <= 0)
        )
        .then(pl.lit("NO_CURRENT_DETERIORATION_SIGNAL"))

        .when(
            pl.col("early_warning_eligible")
            & pl.col("npe_ratio_deterioration")
            .fill_null(False)
            & pl.col("persistent_deterioration")
            .fill_null(False)
            & any_peer_deterioration
        )
        .then(pl.lit("STRONG_ATTENTION"))

        .when(
            pl.col("early_warning_eligible")
            & pl.col("npe_ratio_deterioration")
            .fill_null(False)
            & any_supporting_deterioration
        )
        .then(pl.lit("ELEVATED_ATTENTION"))

        .when(
            pl.col("early_warning_eligible")
            & pl.col("npe_ratio_deterioration")
            .fill_null(False)
        )
        .then(pl.lit("WATCH"))

        .otherwise(pl.lit("UNEXPECTED"))
        .alias("early_warning_status")
    )

    status_profile = (
        mart
        .group_by("early_warning_status")
        .agg(
            pl.len().alias("rows"),
            pl.col("lei_code")
            .n_unique()
            .alias("institutions"),
        )
        .sort("early_warning_status")
    )

    print(status_profile)

    unexpected_statuses = mart.filter(
        ~pl.col("early_warning_status").is_in(ALLOWED_STATUSES)
    )

    if unexpected_statuses.height > 0:
        print(
            unexpected_statuses.select(
                [
                    "lei_code",
                    "period",
                    "is_aggregate",
                    "qoq_available",
                    "npe_ratio_change_qoq",
                    "npe_ratio_deterioration",
                    "eu_peer_deterioration",
                    "country_peer_deterioration",
                    "persistent_deterioration",
                    "early_warning_status",
                ]
            )
        )
        fail("Unexpected early-warning classification encountered.")

    pass_check("Every observation received an approved monitoring status.")

    # =========================================================================
    # 10. PRE-SAVE BUSINESS-RULE VALIDATION
    # =========================================================================

    section("10. PRE-SAVE BUSINESS-RULE VALIDATION")

    if mart.height != source_rows:
        fail(
            f"Row count changed from {source_rows:,} "
            f"to {mart.height:,}."
        )

    pass_check(f"Row count reconciles: {mart.height:,} rows preserved.")

    expected_column_count = source_columns + len(NEW_EARLY_WARNING_COLUMNS)

    if mart.width != expected_column_count:
        fail(
            f"Expected {expected_column_count} columns but found "
            f"{mart.width}."
        )

    pass_check(
        f"Column count reconciles: {mart.width} columns."
    )

    expected_columns = source.columns + NEW_EARLY_WARNING_COLUMNS

    if mart.columns != expected_columns:
        fail(
            "Early-warning mart columns do not match the approved "
            "source-plus-monitoring design."
        )

    pass_check(
        "All original trend fields are preserved and early-warning "
        "fields are appended in the approved order."
    )

    # -------------------------------------------------------------------------
    # Validate aggregate treatment
    # -------------------------------------------------------------------------

    invalid_aggregate_status = mart.filter(
        pl.col("is_aggregate")
        & (
            pl.col("early_warning_status")
            != "NOT_ELIGIBLE_AGGREGATE"
        )
    )

    if invalid_aggregate_status.height > 0:
        fail(
            "One or more aggregate observations received an "
            "institution-level monitoring status."
        )

    pass_check(
        "Aggregate analytical entity receives only "
        "NOT_ELIGIBLE_AGGREGATE."
    )

    # -------------------------------------------------------------------------
    # Validate NOT_ASSESSABLE
    # -------------------------------------------------------------------------

    invalid_not_assessable = mart.filter(
        pl.col("early_warning_eligible")
        & (~pl.col("qoq_available"))
        & (
            pl.col("early_warning_status")
            != "NOT_ASSESSABLE"
        )
    )

    if invalid_not_assessable.height > 0:
        fail(
            "Some institution observations without genuine QoQ "
            "history were not classified as NOT_ASSESSABLE."
        )

    pass_check(
        "Missing genuine QoQ history produces NOT_ASSESSABLE."
    )

    # -------------------------------------------------------------------------
    # Validate no-current-deterioration rule
    # -------------------------------------------------------------------------

    invalid_no_deterioration = mart.filter(
        pl.col("early_warning_eligible")
        & pl.col("qoq_available")
        & (pl.col("npe_ratio_change_qoq") <= 0)
        & (
            pl.col("early_warning_status")
            != "NO_CURRENT_DETERIORATION_SIGNAL"
        )
    )

    if invalid_no_deterioration.height > 0:
        fail(
            "Some non-increasing NPE Ratio observations received "
            "an inappropriate attention classification."
        )

    pass_check(
        "NPE Ratio observations with zero or negative QoQ movement "
        "receive NO_CURRENT_DETERIORATION_SIGNAL."
    )

    # -------------------------------------------------------------------------
    # Validate WATCH
    # -------------------------------------------------------------------------

    invalid_watch = mart.filter(
        (pl.col("early_warning_status") == "WATCH")
        & (
            (~pl.col("npe_ratio_deterioration").fill_null(False))
            |
            (pl.col("supporting_signal_count") != 0)
        )
    )

    if invalid_watch.height > 0:
        fail("WATCH observations violate the approved methodology.")

    pass_check(
        "WATCH requires direct deterioration with zero positive "
        "supporting signals."
    )

    # -------------------------------------------------------------------------
    # Validate ELEVATED_ATTENTION
    # -------------------------------------------------------------------------

    invalid_elevated = mart.filter(
        (pl.col("early_warning_status") == "ELEVATED_ATTENTION")
        & (
            (~pl.col("npe_ratio_deterioration").fill_null(False))
            |
            (pl.col("supporting_signal_count") < 1)
        )
    )

    if invalid_elevated.height > 0:
        fail(
            "ELEVATED_ATTENTION observations violate the approved "
            "methodology."
        )

    pass_check(
        "ELEVATED_ATTENTION requires direct deterioration plus "
        "at least one supporting signal."
    )

    # -------------------------------------------------------------------------
    # Validate STRONG_ATTENTION
    # -------------------------------------------------------------------------

    invalid_strong = mart.filter(
        (pl.col("early_warning_status") == "STRONG_ATTENTION")
        & (
            (~pl.col("npe_ratio_deterioration").fill_null(False))
            |
            (~pl.col("persistent_deterioration").fill_null(False))
            |
            (~any_peer_deterioration)
        )
    )

    if invalid_strong.height > 0:
        fail(
            "STRONG_ATTENTION observations violate the approved "
            "methodology."
        )

    pass_check(
        "STRONG_ATTENTION requires direct deterioration, persistent "
        "deterioration, and at least one peer-relative deterioration."
    )

    # =========================================================================
    # 11. SAVE ANALYTICAL EARLY-WARNING MART
    # =========================================================================

    section("11. SAVE ANALYTICAL NPE EARLY-WARNING MART")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        mart.write_parquet(OUTPUT_PATH)
    except Exception as exc:
        fail(f"Could not write early-warning mart: {exc}")

    pass_check("NPE early-warning mart written successfully.")

    print()
    print("Saved analytical early-warning mart:")
    print(f"  {OUTPUT_PATH}")

    # =========================================================================
    # 12. POST-SAVE VALIDATION
    # =========================================================================

    section("12. POST-SAVE VALIDATION")

    if not OUTPUT_PATH.exists():
        fail("Saved early-warning mart does not exist.")

    pass_check(f"Saved file exists: {OUTPUT_PATH.name}")

    try:
        saved = pl.read_parquet(OUTPUT_PATH)
    except Exception as exc:
        fail(f"Could not reopen saved early-warning mart: {exc}")

    pass_check(
        f"Saved early-warning mart reopened successfully: "
        f"{saved.height:,} rows"
    )

    if saved.height != mart.height:
        fail("Saved row count does not reconcile.")

    pass_check(f"Saved row count reconciles: {saved.height:,}")

    if saved.columns != mart.columns:
        fail("Saved column order does not reconcile.")

    pass_check("Saved column order reconciles.")

    try:
        if not saved.equals(mart):
            fail(
                "Saved Parquet does not exactly match the validated "
                "in-memory mart."
            )
    except Exception as exc:
        fail(f"Could not compare saved mart to in-memory mart: {exc}")

    pass_check(
        "Saved Parquet exactly matches the validated in-memory mart."
    )

    saved_duplicate_grain = (
        saved
        .group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if saved_duplicate_grain.height > 0:
        fail("Saved mart is not unique at grain: lei_code x period")

    pass_check(
        "Saved early-warning mart remains unique at grain: "
        "lei_code x period."
    )

    # =========================================================================
    # 13. FINAL MART PROFILE
    # =========================================================================

    section("13. FINAL NPE EARLY-WARNING MART PROFILE")

    final_profile = saved.select(
        pl.len().alias("rows"),

        pl.col("lei_code")
        .n_unique()
        .alias("entities"),

        pl.col("period")
        .n_unique()
        .alias("periods"),

        pl.col("early_warning_eligible")
        .sum()
        .alias("eligible_rows"),

        pl.col("is_aggregate")
        .sum()
        .alias("aggregate_rows"),

        (
            pl.col("early_warning_status")
            == "WATCH"
        )
        .sum()
        .alias("watch_rows"),

        (
            pl.col("early_warning_status")
            == "ELEVATED_ATTENTION"
        )
        .sum()
        .alias("elevated_attention_rows"),

        (
            pl.col("early_warning_status")
            == "STRONG_ATTENTION"
        )
        .sum()
        .alias("strong_attention_rows"),
    )

    print(final_profile)

    print()
    print("Monitoring-status profile:")

    final_status_profile = (
        saved
        .group_by("early_warning_status")
        .agg(
            pl.len().alias("rows"),
            pl.col("lei_code")
            .n_unique()
            .alias("institutions"),
        )
        .sort("early_warning_status")
    )

    print(final_status_profile)

    print()
    print("Supporting-signal profile:")

    final_support_profile = (
        saved
        .filter(pl.col("early_warning_eligible"))
        .group_by("supporting_signal_count")
        .agg(pl.len().alias("rows"))
        .sort("supporting_signal_count")
    )

    print(final_support_profile)

    # =========================================================================
    # COMPLETE
    # =========================================================================

    section("NPE EARLY-WARNING MART BUILD COMPLETED SUCCESSFULLY")

    print(f"Output: {OUTPUT_PATH}")

    print()
    print(
        "The approved NPE early-warning methodology has been "
        "implemented without introducing additional thresholds."
    )
    print(
        "The output represents monitoring attention, not a "
        "regulatory or credit-risk classification."
    )
    print()
    print("No raw data was modified.")
    print("No staging data was modified.")
    print("No analytical source mart was modified.")
    print("No arbitrary deterioration threshold was introduced.")
    print("No regulatory classification was assigned.")
    print("No probability or predictive score was calculated.")
    print()
    print(
        "The early-warning mart has not yet received independent "
        "validation."
    )


if __name__ == "__main__":
    main()
