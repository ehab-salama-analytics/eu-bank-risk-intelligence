"""
Independent validation of the analytical NPE early-warning mart.

Project:
European Bank Risk Intelligence & Automated Reporting Platform

Validation principle:
    Expected early-warning fields are reconstructed independently from the
    approved NPE trend mart.

    The saved early-warning fields are NOT used to derive their own expected
    values.

Source:
    data/analytical/marts/mart_npe_trend.parquet

Validated output:
    data/analytical/marts/mart_npe_early_warning.parquet

Business grain:
    lei_code x period

Early-warning methodology:
    - institution-level monitoring only
    - aggregate analytical entity excluded
    - direct NPE Ratio deterioration is the primary gate
    - European peer deterioration is supporting evidence
    - country peer deterioration is supporting evidence only when available
    - persistent deterioration requires an approved consecutive-increase pattern
    - unavailable evidence remains unavailable
    - no arbitrary NPE threshold
    - no regulatory classification
    - no predictive score
"""

from pathlib import Path
import sys

import polars as pl


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MART_DIR = PROJECT_ROOT / "data" / "analytical" / "marts"

SOURCE_PATH = MART_DIR / "mart_npe_trend.parquet"
TARGET_PATH = MART_DIR / "mart_npe_early_warning.parquet"

KEY_COLUMNS = [
    "lei_code",
    "period",
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
    "qoq_available",
    "npe_ratio_change_qoq",
    "npe_ratio_pp_change_qoq",
    "eu_percentile_qoq_available",
    "eu_percentile_change_qoq",
    "country_percentile_qoq_available",
    "country_percentile_change_qoq",
    "three_period_pattern_available",
    "npe_three_period_pattern",
]

EARLY_WARNING_COLUMNS = [
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

EXPECTED_EARLY_WARNING_DTYPES = {
    "early_warning_eligible": pl.Boolean,
    "npe_ratio_deterioration": pl.Boolean,
    "eu_peer_deterioration": pl.Boolean,
    "country_peer_deterioration_available": pl.Boolean,
    "country_peer_deterioration": pl.Boolean,
    "persistent_deterioration_available": pl.Boolean,
    "persistent_deterioration": pl.Boolean,
    "supporting_signal_count": pl.Int8,
    "early_warning_status": pl.String,
}

APPROVED_STATUSES = {
    "NOT_ELIGIBLE_AGGREGATE",
    "NOT_ASSESSABLE",
    "NO_CURRENT_DETERIORATION_SIGNAL",
    "WATCH",
    "ELEVATED_ATTENTION",
    "STRONG_ATTENTION",
}


# =============================================================================
# DISPLAY / CONTROL HELPERS
# =============================================================================

LINE = "=" * 90
SUBLINE = "-" * 90

CHECK_RESULTS: list[tuple[bool, str]] = []


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def record_check(condition: bool, message: str) -> bool:
    """Record and display a validation result."""

    CHECK_RESULTS.append((condition, message))

    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")

    return condition


def fatal(message: str) -> None:
    """Stop validation when further checks would be unreliable."""

    print(f"[FAIL] {message}")
    print()
    print("Validation cannot continue safely.")
    sys.exit(1)


def frames_equal(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    """
    Compare two Polars DataFrames exactly while allowing equivalent NULLs.

    Compatibility handling is included for different Polars versions.
    """

    if left.shape != right.shape:
        return False

    if left.columns != right.columns:
        return False

    try:
        return left.equals(right, null_equal=True)
    except TypeError:
        return left.equals(right)
    except AttributeError:
        try:
            return left.frame_equal(right, null_equal=True)
        except TypeError:
            return left.frame_equal(right)


def null_safe_mismatch_expression(
    actual_column: str,
    expected_column: str,
) -> pl.Expr:
    """
    Return TRUE when actual and expected values differ.

    NULL = NULL is considered a match.
    """

    both_null = (
        pl.col(actual_column).is_null()
        & pl.col(expected_column).is_null()
    )

    both_non_null_and_equal = (
        pl.col(actual_column).is_not_null()
        & pl.col(expected_column).is_not_null()
        & (pl.col(actual_column) == pl.col(expected_column))
    )

    return ~(both_null | both_non_null_and_equal)


# =============================================================================
# HEADER
# =============================================================================

print(LINE)
print("NPE EARLY-WARNING MART INDEPENDENT VALIDATION")
print("European Bank Risk Intelligence & Automated Reporting Platform")
print(LINE)

print()
print(f"Project root       : {PROJECT_ROOT}")
print(f"NPE Trend mart     : {SOURCE_PATH}")
print(f"Early-warning mart : {TARGET_PATH}")

print()
print("VALIDATION MODE: READ ONLY")
print("No analytical, staging, reference, or raw data will be modified.")

print()
print("INDEPENDENT VALIDATION PRINCIPLE:")
print("Early-warning fields are reconstructed directly from mart_npe_trend.parquet.")
print("Saved early-warning fields are not used to derive their expected values.")


# =============================================================================
# 1. FILE AVAILABILITY
# =============================================================================

section("1. FILE AVAILABILITY")

if not SOURCE_PATH.exists():
    fatal(f"Source mart not found: {SOURCE_PATH}")

print(f"[PASS] Found source mart: {SOURCE_PATH.name}")

if not TARGET_PATH.exists():
    fatal(f"Early-warning mart not found: {TARGET_PATH}")

print(f"[PASS] Found early-warning mart: {TARGET_PATH.name}")


# =============================================================================
# 2. LOAD APPROVED ANALYTICAL DATA
# =============================================================================

section("2. LOAD APPROVED ANALYTICAL DATA")

try:
    source = pl.read_parquet(SOURCE_PATH)
except Exception as exc:
    fatal(f"Could not open {SOURCE_PATH.name}: {exc}")

try:
    saved = pl.read_parquet(TARGET_PATH)
except Exception as exc:
    fatal(f"Could not open {TARGET_PATH.name}: {exc}")

print(f"[PASS] NPE Trend mart opened: {source.height:,} rows")
print(f"[PASS] NPE Early-warning mart opened: {saved.height:,} rows")


# =============================================================================
# 3. SOURCE STRUCTURE VALIDATION
# =============================================================================

section("3. SOURCE STRUCTURE VALIDATION")

missing_source_columns = [
    column
    for column in REQUIRED_SOURCE_COLUMNS
    if column not in source.columns
]

if missing_source_columns:
    print(f"Missing source columns: {missing_source_columns}")
    fatal("Required source fields are missing.")

record_check(
    source.height > 0,
    "NPE Trend source contains observations.",
)

source_duplicate_count = (
    source
    .group_by(KEY_COLUMNS)
    .len()
    .filter(pl.col("len") > 1)
    .height
)

record_check(
    source_duplicate_count == 0,
    "NPE Trend source is unique at grain: lei_code x period.",
)

source_null_key_count = (
    source
    .filter(
        pl.col("lei_code").is_null()
        | pl.col("period").is_null()
    )
    .height
)

record_check(
    source_null_key_count == 0,
    "NPE Trend source business grain contains no NULL values.",
)

print(f"[INFO] Source rows              : {source.height:,}")
print(f"[INFO] Source columns           : {source.width}")
print(f"[INFO] Distinct entities        : {source['lei_code'].n_unique():,}")
print(f"[INFO] Distinct reporting periods: {source['period'].n_unique():,}")


# =============================================================================
# 4. EARLY-WARNING MART STRUCTURAL VALIDATION
# =============================================================================

section("4. EARLY-WARNING MART STRUCTURAL VALIDATION")

expected_columns = source.columns + EARLY_WARNING_COLUMNS

record_check(
    saved.height == source.height,
    f"Early-warning row count reconciles to source: {saved.height:,} rows.",
)

record_check(
    saved.columns == expected_columns,
    "Early-warning mart contains the approved source columns plus the nine "
    "early-warning fields in exact order.",
)

if saved.columns != expected_columns:
    print()
    print("Expected columns:")
    for index, column in enumerate(expected_columns, start=1):
        print(f"{index:>3}. {column}")

    print()
    print("Actual columns:")
    for index, column in enumerate(saved.columns, start=1):
        print(f"{index:>3}. {column}")

saved_duplicate_count = (
    saved
    .group_by(KEY_COLUMNS)
    .len()
    .filter(pl.col("len") > 1)
    .height
)

record_check(
    saved_duplicate_count == 0,
    "Early-warning mart remains unique at grain: lei_code x period.",
)

saved_null_key_count = (
    saved
    .filter(
        pl.col("lei_code").is_null()
        | pl.col("period").is_null()
    )
    .height
)

record_check(
    saved_null_key_count == 0,
    "Early-warning mart business grain contains no NULL values.",
)


# =============================================================================
# 5. EARLY-WARNING COLUMN DATA-TYPE VALIDATION
# =============================================================================

section("5. EARLY-WARNING COLUMN DATA-TYPE VALIDATION")

for column, expected_dtype in EXPECTED_EARLY_WARNING_DTYPES.items():

    if column not in saved.columns:
        record_check(
            False,
            f"{column} is missing.",
        )
        continue

    actual_dtype = saved.schema[column]

    record_check(
        actual_dtype == expected_dtype,
        f"{column} dtype is {actual_dtype}.",
    )


# =============================================================================
# 6. SOURCE FIELD PRESERVATION
# =============================================================================

section("6. TREND SOURCE FIELD PRESERVATION")

if all(column in saved.columns for column in source.columns):

    source_sorted = (
        source
        .sort(KEY_COLUMNS)
        .select(source.columns)
    )

    saved_source_sorted = (
        saved
        .sort(KEY_COLUMNS)
        .select(source.columns)
    )

    record_check(
        frames_equal(source_sorted, saved_source_sorted),
        "All original NPE Trend mart fields are preserved exactly.",
    )

else:
    record_check(
        False,
        "Cannot validate source preservation because source fields are missing "
        "from the early-warning mart.",
    )


# =============================================================================
# 7. INDEPENDENT EARLY-WARNING RECONSTRUCTION
# =============================================================================

section("7. INDEPENDENT EARLY-WARNING RECONSTRUCTION")

expected = (
    source

    # -------------------------------------------------------------------------
    # Eligibility
    # -------------------------------------------------------------------------
    .with_columns(
        (
            ~pl.col("is_aggregate")
        )
        .cast(pl.Boolean)
        .alias("early_warning_eligible")
    )

    # -------------------------------------------------------------------------
    # Primary direct NPE deterioration
    #
    # TRUE  = genuine QoQ exists and NPE Ratio increased
    # FALSE = genuine QoQ exists and NPE Ratio did not increase
    # NULL  = comparison unavailable or aggregate analytical entity
    # -------------------------------------------------------------------------
    .with_columns(
        pl.when(
            pl.col("early_warning_eligible")
            & pl.col("qoq_available").fill_null(False)
        )
        .then(
            pl.col("npe_ratio_change_qoq") > 0
        )
        .otherwise(
            pl.lit(None, dtype=pl.Boolean)
        )
        .cast(pl.Boolean)
        .alias("npe_ratio_deterioration")
    )

    # -------------------------------------------------------------------------
    # European peer deterioration
    #
    # Higher percentile = relatively worse NPE position under the approved
    # ascending NPE Ratio benchmark methodology.
    # -------------------------------------------------------------------------
    .with_columns(
        pl.when(
            pl.col("early_warning_eligible")
            & pl.col("eu_percentile_qoq_available").fill_null(False)
        )
        .then(
            pl.col("eu_percentile_change_qoq") > 0
        )
        .otherwise(
            pl.lit(None, dtype=pl.Boolean)
        )
        .cast(pl.Boolean)
        .alias("eu_peer_deterioration")
    )

    # -------------------------------------------------------------------------
    # Country peer availability
    # -------------------------------------------------------------------------
    .with_columns(
        (
            pl.col("early_warning_eligible")
            & pl.col("country_percentile_qoq_available").fill_null(False)
        )
        .cast(pl.Boolean)
        .alias("country_peer_deterioration_available")
    )

    # -------------------------------------------------------------------------
    # Country peer deterioration
    # -------------------------------------------------------------------------
    .with_columns(
        pl.when(
            pl.col("country_peer_deterioration_available")
        )
        .then(
            pl.col("country_percentile_change_qoq") > 0
        )
        .otherwise(
            pl.lit(None, dtype=pl.Boolean)
        )
        .cast(pl.Boolean)
        .alias("country_peer_deterioration")
    )

    # -------------------------------------------------------------------------
    # Persistent deterioration availability
    # -------------------------------------------------------------------------
    .with_columns(
        (
            pl.col("early_warning_eligible")
            & pl.col("three_period_pattern_available").fill_null(False)
        )
        .cast(pl.Boolean)
        .alias("persistent_deterioration_available")
    )

    # -------------------------------------------------------------------------
    # Persistent deterioration
    #
    # Only the validated three-period pattern "consecutive_increase"
    # constitutes persistent deterioration.
    # -------------------------------------------------------------------------
    .with_columns(
        pl.when(
            pl.col("persistent_deterioration_available")
        )
        .then(
            pl.col("npe_three_period_pattern") == "consecutive_increase"
        )
        .otherwise(
            pl.lit(None, dtype=pl.Boolean)
        )
        .cast(pl.Boolean)
        .alias("persistent_deterioration")
    )

    # -------------------------------------------------------------------------
    # Supporting signal count
    #
    # Supporting evidence:
    #     1. European peer deterioration
    #     2. Country peer deterioration when available
    #     3. Persistent deterioration
    #
    # Unavailable evidence contributes zero to the count.
    # Aggregate analytical entities receive NULL because they are ineligible.
    # -------------------------------------------------------------------------
    .with_columns(
        pl.when(
            pl.col("early_warning_eligible")
        )
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
            .cast(pl.Int8)
        )
        .otherwise(
            pl.lit(None, dtype=pl.Int8)
        )
        .alias("supporting_signal_count")
    )

    # -------------------------------------------------------------------------
    # Monitoring status
    #
    # This is a monitoring-attention classification, NOT a regulatory
    # credit-risk classification.
    # -------------------------------------------------------------------------
    .with_columns(
        pl.when(
            ~pl.col("early_warning_eligible")
        )
        .then(
            pl.lit("NOT_ELIGIBLE_AGGREGATE")
        )

        .when(
            pl.col("npe_ratio_deterioration").is_null()
        )
        .then(
            pl.lit("NOT_ASSESSABLE")
        )

        .when(
            pl.col("npe_ratio_deterioration") == False
        )
        .then(
            pl.lit("NO_CURRENT_DETERIORATION_SIGNAL")
        )

        .when(
            pl.col("npe_ratio_deterioration").fill_null(False)
            & pl.col("persistent_deterioration").fill_null(False)
            & (
                pl.col("eu_peer_deterioration").fill_null(False)
                | pl.col("country_peer_deterioration").fill_null(False)
            )
        )
        .then(
            pl.lit("STRONG_ATTENTION")
        )

        .when(
            pl.col("supporting_signal_count").fill_null(0) >= 1
        )
        .then(
            pl.lit("ELEVATED_ATTENTION")
        )

        .otherwise(
            pl.lit("WATCH")
        )
        .cast(pl.String)
        .alias("early_warning_status")
    )
)

record_check(
    expected.height == source.height,
    f"Independent reconstruction created {expected.height:,} observations.",
)

expected_duplicate_count = (
    expected
    .group_by(KEY_COLUMNS)
    .len()
    .filter(pl.col("len") > 1)
    .height
)

record_check(
    expected_duplicate_count == 0,
    "Independent reconstruction is unique at grain: lei_code x period.",
)


# =============================================================================
# 8. SAVED MART TO INDEPENDENT RECONSTRUCTION ALIGNMENT
# =============================================================================

section("8. SAVED MART TO INDEPENDENT RECONSTRUCTION ALIGNMENT")

expected_renamed = expected.select(
    KEY_COLUMNS
    + EARLY_WARNING_COLUMNS
).rename(
    {
        column: f"expected_{column}"
        for column in EARLY_WARNING_COLUMNS
    }
)

comparison = (
    saved
    .select(
        KEY_COLUMNS
        + EARLY_WARNING_COLUMNS
    )
    .join(
        expected_renamed,
        on=KEY_COLUMNS,
        how="inner",
        validate="1:1",
    )
)

record_check(
    comparison.height == source.height,
    "Saved early-warning observations align one-to-one with the independent "
    "reconstruction.",
)


# =============================================================================
# 9. EARLY-WARNING FIELD RECONCILIATION
# =============================================================================

section("9. EARLY-WARNING FIELD RECONCILIATION")

field_reconciliation_passed = True

for column in EARLY_WARNING_COLUMNS:

    expected_column = f"expected_{column}"

    mismatch_count = (
        comparison
        .filter(
            null_safe_mismatch_expression(
                column,
                expected_column,
            )
        )
        .height
    )

    passed = mismatch_count == 0

    record_check(
        passed,
        (
            f"{column} independently reconciles exactly."
            if passed
            else f"{column} has {mismatch_count:,} mismatch(es)."
        ),
    )

    if not passed:
        field_reconciliation_passed = False

        print()
        print(f"Mismatch sample for {column}:")

        print(
            comparison
            .filter(
                null_safe_mismatch_expression(
                    column,
                    expected_column,
                )
            )
            .select(
                [
                    "lei_code",
                    "period",
                    column,
                    expected_column,
                ]
            )
            .head(20)
        )


# =============================================================================
# 10. PRIMARY DETERIORATION BUSINESS RULE
# =============================================================================

section("10. PRIMARY NPE RATIO DETERIORATION BUSINESS RULE")

aggregate_direct_values = (
    saved
    .filter(pl.col("is_aggregate"))
    .filter(pl.col("npe_ratio_deterioration").is_not_null())
    .height
)

record_check(
    aggregate_direct_values == 0,
    "Aggregate analytical entity receives no institution-level direct "
    "deterioration assessment.",
)

invalid_direct_positive = (
    saved
    .filter(
        pl.col("npe_ratio_deterioration").fill_null(False)
        & (
            (~pl.col("qoq_available").fill_null(False))
            | (pl.col("npe_ratio_change_qoq") <= 0)
        )
    )
    .height
)

record_check(
    invalid_direct_positive == 0,
    "Direct deterioration is TRUE only for genuine QoQ observations with "
    "positive NPE Ratio movement.",
)

invalid_direct_negative = (
    saved
    .filter(
        pl.col("early_warning_eligible")
        & pl.col("qoq_available").fill_null(False)
        & (pl.col("npe_ratio_change_qoq") <= 0)
        & (pl.col("npe_ratio_deterioration") != False)
    )
    .height
)

record_check(
    invalid_direct_negative == 0,
    "Zero or negative QoQ NPE Ratio movement is not classified as deterioration.",
)


# =============================================================================
# 11. PEER-SIGNAL BUSINESS RULES
# =============================================================================

section("11. PEER DETERIORATION BUSINESS RULES")

invalid_eu_signal = (
    saved
    .filter(
        pl.col("eu_peer_deterioration").fill_null(False)
        & (
            (~pl.col("eu_percentile_qoq_available").fill_null(False))
            | (pl.col("eu_percentile_change_qoq") <= 0)
        )
    )
    .height
)

record_check(
    invalid_eu_signal == 0,
    "European peer deterioration requires an available positive "
    "European-percentile movement.",
)

invalid_country_availability = (
    saved
    .filter(
        pl.col("country_peer_deterioration_available")
        != (
            pl.col("early_warning_eligible")
            & pl.col("country_percentile_qoq_available").fill_null(False)
        )
    )
    .height
)

record_check(
    invalid_country_availability == 0,
    "Country peer deterioration availability reconciles to formal "
    "country-percentile QoQ availability.",
)

invalid_country_signal = (
    saved
    .filter(
        pl.col("country_peer_deterioration").fill_null(False)
        & (
            (~pl.col("country_peer_deterioration_available"))
            | (pl.col("country_percentile_change_qoq") <= 0)
        )
    )
    .height
)

record_check(
    invalid_country_signal == 0,
    "Country peer deterioration is TRUE only when formal country comparison "
    "is available and percentile position worsened.",
)

unavailable_country_non_null = (
    saved
    .filter(
        (~pl.col("country_peer_deterioration_available"))
        & pl.col("country_peer_deterioration").is_not_null()
    )
    .height
)

record_check(
    unavailable_country_non_null == 0,
    "Unavailable country peer evidence remains NULL rather than FALSE.",
)


# =============================================================================
# 12. PERSISTENT DETERIORATION BUSINESS RULE
# =============================================================================

section("12. PERSISTENT DETERIORATION BUSINESS RULE")

invalid_persistent_availability = (
    saved
    .filter(
        pl.col("persistent_deterioration_available")
        != (
            pl.col("early_warning_eligible")
            & pl.col("three_period_pattern_available").fill_null(False)
        )
    )
    .height
)

record_check(
    invalid_persistent_availability == 0,
    "Persistent-deterioration availability reconciles to approved "
    "three-period-pattern availability.",
)

invalid_persistent_true = (
    saved
    .filter(
        pl.col("persistent_deterioration").fill_null(False)
        & (
            (~pl.col("persistent_deterioration_available"))
            | (
                pl.col("npe_three_period_pattern")
                != "consecutive_increase"
            )
        )
    )
    .height
)

record_check(
    invalid_persistent_true == 0,
    "Persistent deterioration is TRUE only for the approved "
    "'consecutive_increase' pattern.",
)

unavailable_persistent_non_null = (
    saved
    .filter(
        (~pl.col("persistent_deterioration_available"))
        & pl.col("persistent_deterioration").is_not_null()
    )
    .height
)

record_check(
    unavailable_persistent_non_null == 0,
    "Unavailable persistent-deterioration evidence remains NULL rather "
    "than FALSE.",
)


# =============================================================================
# 13. SUPPORTING SIGNAL COUNT VALIDATION
# =============================================================================

section("13. SUPPORTING SIGNAL COUNT VALIDATION")

expected_supporting_count = (
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
    .cast(pl.Int8)
)

invalid_support_count = (
    saved
    .filter(
        pl.col("early_warning_eligible")
        & (
            pl.col("supporting_signal_count")
            != expected_supporting_count
        )
    )
    .height
)

record_check(
    invalid_support_count == 0,
    "Supporting signal count equals the number of positive approved "
    "supporting signals.",
)

aggregate_support_count = (
    saved
    .filter(
        pl.col("is_aggregate")
        & pl.col("supporting_signal_count").is_not_null()
    )
    .height
)

record_check(
    aggregate_support_count == 0,
    "Aggregate analytical entity receives no supporting-signal count.",
)

out_of_range_support_count = (
    saved
    .filter(
        pl.col("supporting_signal_count").is_not_null()
        & (
            (pl.col("supporting_signal_count") < 0)
            | (pl.col("supporting_signal_count") > 3)
        )
    )
    .height
)

record_check(
    out_of_range_support_count == 0,
    "Supporting signal count lies within the approved range 0-3.",
)

print()
print("Supporting-signal profile:")

print(
    saved
    .group_by("supporting_signal_count")
    .agg(
        pl.len().alias("rows")
    )
    .sort(
        "supporting_signal_count",
        nulls_last=False,
    )
)


# =============================================================================
# 14. MONITORING STATUS BUSINESS-RULE VALIDATION
# =============================================================================

section("14. EARLY-WARNING MONITORING STATUS BUSINESS RULES")

observed_statuses = set(
    saved
    .get_column("early_warning_status")
    .drop_nulls()
    .unique()
    .to_list()
)

record_check(
    observed_statuses.issubset(APPROVED_STATUSES),
    "All monitoring statuses belong to the approved status domain.",
)

invalid_aggregate_status = (
    saved
    .filter(
        pl.col("is_aggregate")
        & (
            pl.col("early_warning_status")
            != "NOT_ELIGIBLE_AGGREGATE"
        )
    )
    .height
)

record_check(
    invalid_aggregate_status == 0,
    "Aggregate analytical entity receives only NOT_ELIGIBLE_AGGREGATE.",
)

invalid_not_assessable = (
    saved
    .filter(
        (
            pl.col("early_warning_status")
            == "NOT_ASSESSABLE"
        )
        & (
            (~pl.col("early_warning_eligible"))
            | pl.col("npe_ratio_deterioration").is_not_null()
        )
    )
    .height
)

record_check(
    invalid_not_assessable == 0,
    "NOT_ASSESSABLE is used only for eligible institutions lacking a genuine "
    "direct QoQ deterioration assessment.",
)

invalid_no_signal = (
    saved
    .filter(
        (
            pl.col("early_warning_status")
            == "NO_CURRENT_DETERIORATION_SIGNAL"
        )
        & (
            pl.col("npe_ratio_deterioration") != False
        )
    )
    .height
)

record_check(
    invalid_no_signal == 0,
    "NO_CURRENT_DETERIORATION_SIGNAL requires no direct NPE Ratio deterioration.",
)

invalid_watch = (
    saved
    .filter(
        (
            pl.col("early_warning_status")
            == "WATCH"
        )
        & (
            (pl.col("npe_ratio_deterioration") != True)
            | (pl.col("supporting_signal_count") != 0)
        )
    )
    .height
)

record_check(
    invalid_watch == 0,
    "WATCH requires direct deterioration with zero positive supporting signals.",
)

invalid_elevated = (
    saved
    .filter(
        (
            pl.col("early_warning_status")
            == "ELEVATED_ATTENTION"
        )
        & (
            (pl.col("npe_ratio_deterioration") != True)
            | (pl.col("supporting_signal_count") < 1)
        )
    )
    .height
)

record_check(
    invalid_elevated == 0,
    "ELEVATED_ATTENTION requires direct deterioration plus at least one "
    "supporting signal.",
)

invalid_strong = (
    saved
    .filter(
        (
            pl.col("early_warning_status")
            == "STRONG_ATTENTION"
        )
        & (
            (pl.col("npe_ratio_deterioration") != True)
            | (pl.col("persistent_deterioration") != True)
            | ~(
                pl.col("eu_peer_deterioration").fill_null(False)
                | pl.col("country_peer_deterioration").fill_null(False)
            )
        )
    )
    .height
)

record_check(
    invalid_strong == 0,
    "STRONG_ATTENTION requires direct deterioration, persistent deterioration, "
    "and at least one peer-relative deterioration signal.",
)


# =============================================================================
# 15. EARLY-WARNING POPULATION PROFILE
# =============================================================================

section("15. EARLY-WARNING POPULATION PROFILE")

population_profile = saved.select(
    [
        pl.len().alias("rows"),
        pl.col("lei_code").n_unique().alias("entities"),
        pl.col("period").n_unique().alias("periods"),
        pl.col("is_aggregate").sum().alias("aggregate_rows"),
        pl.col("early_warning_eligible").sum().alias("eligible_rows"),
        pl.col("npe_ratio_deterioration")
        .fill_null(False)
        .sum()
        .alias("direct_deterioration_rows"),
        pl.col("persistent_deterioration")
        .fill_null(False)
        .sum()
        .alias("persistent_deterioration_rows"),
    ]
)

print(population_profile)

print()
print("Monitoring-status profile:")

status_profile = (
    saved
    .group_by("early_warning_status")
    .agg(
        [
            pl.len().alias("rows"),
            pl.col("lei_code").n_unique().alias("institutions"),
        ]
    )
    .sort("early_warning_status")
)

print(status_profile)

print()
print("Period profile:")

period_profile = (
    saved
    .group_by("period")
    .agg(
        [
            pl.len().alias("rows"),
            pl.col("lei_code").n_unique().alias("entities"),
            pl.col("qoq_available")
            .fill_null(False)
            .sum()
            .alias("qoq_rows"),
            pl.col("npe_ratio_deterioration")
            .fill_null(False)
            .sum()
            .alias("deterioration_rows"),
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
        ]
    )
    .sort("period")
)

print(period_profile)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("NPE EARLY-WARNING MART INDEPENDENT VALIDATION SUMMARY")

total_checks = len(CHECK_RESULTS)
passed_checks = sum(1 for result, _ in CHECK_RESULTS if result)
failed_checks = total_checks - passed_checks

print(f"Total checks : {total_checks}")
print(f"PASS         : {passed_checks}")
print(f"FAIL         : {failed_checks}")

print(SUBLINE)

if failed_checks == 0:

    print("OVERALL STATUS : PASS")

    print()
    print(
        "INTERPRETATION : The saved NPE early-warning mart independently "
        "reconciles to the validated NPE trend mart and the documented "
        "early-warning methodology."
    )

    print()
    print("Validated controls include:")
    print("  - trend-source observation preservation")
    print("  - institution-period business grain")
    print("  - early-warning eligibility")
    print("  - genuine QoQ direct deterioration")
    print("  - European peer-relative deterioration")
    print("  - country peer availability and deterioration")
    print("  - persistent three-period deterioration")
    print("  - unavailable-evidence NULL preservation")
    print("  - supporting-signal count")
    print("  - monitoring-status assignment")
    print("  - aggregate analytical entity exclusion")

    print()
    print(
        "ACTION         : mart_npe_early_warning can be considered for "
        "approval for downstream monitoring dashboards, management reporting, "
        "and further analytical development."
    )

    print()
    print(
        "The monitoring statuses are analytical attention categories. "
        "They are not regulatory classifications, default assessments, "
        "credit ratings, or predictive probabilities."
    )

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No data was modified.")

    sys.exit(0)

else:

    print("OVERALL STATUS : FAIL")

    print()
    print(
        "INTERPRETATION : One or more independently reconstructed "
        "early-warning controls do not reconcile to the saved analytical mart."
    )

    print()
    print(
        "ACTION         : Do not approve mart_npe_early_warning for downstream "
        "reporting until the failed controls are investigated."
    )

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No data was modified.")

    sys.exit(1)
