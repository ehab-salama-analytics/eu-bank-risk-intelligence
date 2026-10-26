from pathlib import Path
import sys

import polars as pl


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "facts"
    / "fact_credit_risk.parquet"
)

INSTITUTION_DIM_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
    / "dim_institution.parquet"
)

PERIOD_DIM_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
    / "dim_period.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
)

OUTPUT_PATH = OUTPUT_DIR / "mart_npe_ratio.parquet"


# Approved KPI coordinates
TARGET_ITEM = 2520603

TOTAL_PORTFOLIO = 0
TOTAL_COUNTRY = 0
TOTAL_EXPOSURE = 0
TOTAL_STATUS = 0

TOTAL_PERF_STATUS = 0
NPE_PERF_STATUS = 2

# Based on the independently validated source population
EXPECTED_ROWS = 426

# Technical tolerance only for validating numerator <= denominator.
# This is NOT a KPI threshold.
NUMERIC_EPSILON = 1e-9


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

SEPARATOR = "=" * 100
SUB_SEPARATOR = "-" * 100


def section(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def passed(message: str) -> None:
    print(f"[PASS] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


# =============================================================================
# FILE VALIDATION
# =============================================================================

def validate_input_file(path: Path, description: str) -> None:
    if not path.exists():
        fail(f"{description} not found: {path}")

    passed(f"{description} exists: {path.name}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    section("1. LOAD APPROVED ANALYTICAL DATA")

    validate_input_file(FACT_PATH, "Credit-risk analytical fact")
    validate_input_file(INSTITUTION_DIM_PATH, "Institution dimension")
    validate_input_file(PERIOD_DIM_PATH, "Period dimension")

    try:
        fact = pl.read_parquet(FACT_PATH)
        institutions = pl.read_parquet(INSTITUTION_DIM_PATH)
        periods = pl.read_parquet(PERIOD_DIM_PATH)
    except Exception as exc:
        fail(f"Unable to open analytical Parquet files: {exc}")

    passed(f"fact_credit_risk opened: {fact.height:,} rows")
    passed(f"dim_institution opened: {institutions.height:,} rows")
    passed(f"dim_period opened: {periods.height:,} rows")

    return fact, institutions, periods


# =============================================================================
# SOURCE STRUCTURE VALIDATION
# =============================================================================

def validate_source_columns(
    fact: pl.DataFrame,
    institutions: pl.DataFrame,
    periods: pl.DataFrame,
) -> None:
    section("2. SOURCE STRUCTURE VALIDATION")

    required_fact_columns = {
        "lei_code",
        "period",
        "item",
        "portfolio",
        "country",
        "exposure",
        "status",
        "perf_status",
        "amount",
    }

    required_institution_columns = {
        "lei_code",
        "name",
        "country",
        "desc_country",
        "entity_type",
        "is_aggregate",
    }

    required_period_columns = {
        "period_code",
        "period_end_date",
        "year",
        "quarter",
        "month",
        "year_quarter",
        "is_year_end",
    }

    missing_fact = required_fact_columns - set(fact.columns)
    missing_institution = required_institution_columns - set(institutions.columns)
    missing_period = required_period_columns - set(periods.columns)

    if missing_fact:
        fail(
            "fact_credit_risk is missing required column(s): "
            + ", ".join(sorted(missing_fact))
        )

    if missing_institution:
        fail(
            "dim_institution is missing required column(s): "
            + ", ".join(sorted(missing_institution))
        )

    if missing_period:
        fail(
            "dim_period is missing required column(s): "
            + ", ".join(sorted(missing_period))
        )

    passed("All required fact columns are present.")
    passed("All required institution-dimension columns are present.")
    passed("All required period-dimension columns are present.")


# =============================================================================
# KPI COMPONENT EXTRACTION
# =============================================================================

def extract_denominator(fact: pl.DataFrame) -> pl.DataFrame:
    section("3. EXTRACT APPROVED DENOMINATOR")

    denominator = (
        fact.filter(
            (pl.col("item") == TARGET_ITEM)
            & (pl.col("portfolio") == TOTAL_PORTFOLIO)
            & (pl.col("country") == TOTAL_COUNTRY)
            & (pl.col("exposure") == TOTAL_EXPOSURE)
            & (pl.col("status") == TOTAL_STATUS)
            & (pl.col("perf_status") == TOTAL_PERF_STATUS)
        )
        .select(
            [
                "lei_code",
                "period",
                pl.col("amount").alias("total_gross_carrying_amount"),
            ]
        )
        .sort(["lei_code", "period"])
    )

    info(f"Denominator rows: {denominator.height:,}")
    info(
        "Distinct institutions: "
        f"{denominator.select(pl.col('lei_code').n_unique()).item():,}"
    )
    info(
        "Distinct periods: "
        f"{denominator.select(pl.col('period').n_unique()).item():,}"
    )

    duplicate_count = (
        denominator.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_count != 0:
        fail(
            "Denominator contains duplicate institution-period coordinates. "
            f"Duplicate groups: {duplicate_count}"
        )

    null_amounts = denominator["total_gross_carrying_amount"].null_count()

    if null_amounts != 0:
        fail(
            "Denominator contains NULL Amount values. "
            f"NULL count: {null_amounts}"
        )

    passed("Denominator contains exactly one observation per institution-period.")
    passed("Denominator contains no NULL Amount values.")

    return denominator


def extract_numerator(fact: pl.DataFrame) -> pl.DataFrame:
    section("4. EXTRACT APPROVED NPE NUMERATOR")

    numerator = (
        fact.filter(
            (pl.col("item") == TARGET_ITEM)
            & (pl.col("portfolio") == TOTAL_PORTFOLIO)
            & (pl.col("country") == TOTAL_COUNTRY)
            & (pl.col("exposure") == TOTAL_EXPOSURE)
            & (pl.col("status") == TOTAL_STATUS)
            & (pl.col("perf_status") == NPE_PERF_STATUS)
        )
        .select(
            [
                "lei_code",
                "period",
                pl.col("amount").alias("non_performing_amount"),
            ]
        )
        .sort(["lei_code", "period"])
    )

    info(f"Numerator rows: {numerator.height:,}")
    info(
        "Distinct institutions: "
        f"{numerator.select(pl.col('lei_code').n_unique()).item():,}"
    )
    info(
        "Distinct periods: "
        f"{numerator.select(pl.col('period').n_unique()).item():,}"
    )

    duplicate_count = (
        numerator.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_count != 0:
        fail(
            "Numerator contains duplicate institution-period coordinates. "
            f"Duplicate groups: {duplicate_count}"
        )

    null_amounts = numerator["non_performing_amount"].null_count()

    if null_amounts != 0:
        fail(
            "Numerator contains NULL Amount values. "
            f"NULL count: {null_amounts}"
        )

    passed("Numerator contains exactly one observation per institution-period.")
    passed("Numerator contains no NULL Amount values.")

    return numerator


# =============================================================================
# COVERAGE VALIDATION
# =============================================================================

def validate_component_coverage(
    denominator: pl.DataFrame,
    numerator: pl.DataFrame,
) -> None:
    section("5. NUMERATOR / DENOMINATOR COVERAGE VALIDATION")

    denominator_keys = denominator.select(["lei_code", "period"])
    numerator_keys = numerator.select(["lei_code", "period"])

    missing_numerator = denominator_keys.join(
        numerator_keys,
        on=["lei_code", "period"],
        how="anti",
    )

    missing_denominator = numerator_keys.join(
        denominator_keys,
        on=["lei_code", "period"],
        how="anti",
    )

    info(f"Denominator institution-period rows: {denominator.height:,}")
    info(f"Numerator institution-period rows:   {numerator.height:,}")
    info(f"Missing numerator observations:       {missing_numerator.height:,}")
    info(f"Missing denominator observations:     {missing_denominator.height:,}")

    if missing_numerator.height != 0:
        fail("Some denominator observations do not have a numerator observation.")

    if missing_denominator.height != 0:
        fail("Some numerator observations do not have a denominator observation.")

    passed("Numerator and denominator institution-period coverage is identical.")


# =============================================================================
# BUILD KPI MART
# =============================================================================

def build_kpi_mart(
    denominator: pl.DataFrame,
    numerator: pl.DataFrame,
    institutions: pl.DataFrame,
    periods: pl.DataFrame,
) -> pl.DataFrame:
    section("6. BUILD NPE RATIO MART")

    kpi = denominator.join(
        numerator,
        on=["lei_code", "period"],
        how="inner",
        validate="1:1",
    )

    # -------------------------------------------------------------------------
    # Validate numerator <= denominator before ratio construction.
    # NUMERIC_EPSILON exists only to accommodate insignificant Float64 noise.
    # -------------------------------------------------------------------------

    numerator_exceeds_denominator = kpi.filter(
        pl.col("non_performing_amount")
        > pl.col("total_gross_carrying_amount") + NUMERIC_EPSILON
    )

    if numerator_exceeds_denominator.height != 0:
        print()
        print("Observations where numerator exceeds denominator:")
        print(numerator_exceeds_denominator.head(20))

        fail(
            "NPE numerator exceeds denominator for "
            f"{numerator_exceeds_denominator.height} observation(s)."
        )

    passed("No NPE numerator materially exceeds its denominator.")

    # -------------------------------------------------------------------------
    # Compute KPI.
    #
    # Zero denominator -> NULL ratio.
    # NULL is deliberately used rather than 0, infinity or an arbitrary value.
    # -------------------------------------------------------------------------

    kpi = kpi.with_columns(
        [
            (pl.col("total_gross_carrying_amount") == 0)
            .alias("zero_denominator_flag"),

            pl.when(pl.col("total_gross_carrying_amount") == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("non_performing_amount")
                / pl.col("total_gross_carrying_amount")
            )
            .alias("npe_ratio"),
        ]
    )

    kpi = kpi.with_columns(
        (pl.col("npe_ratio") * 100.0).alias("npe_ratio_pct")
    )

    # -------------------------------------------------------------------------
    # Institution enrichment
    # -------------------------------------------------------------------------

    institution_lookup = institutions.select(
        [
            "lei_code",
            pl.col("name").alias("institution_name"),
            pl.col("country").alias("institution_country_code"),
            pl.col("desc_country").alias("institution_country"),
            "entity_type",
            "is_aggregate",
        ]
    )

    kpi = kpi.join(
        institution_lookup,
        on="lei_code",
        how="left",
        validate="m:1",
    )

    # -------------------------------------------------------------------------
    # Period enrichment
    # -------------------------------------------------------------------------

    period_lookup = periods.select(
        [
            pl.col("period_code").alias("period"),
            "period_end_date",
            "year",
            "quarter",
            "month",
            "year_quarter",
            "is_year_end",
        ]
    )

    kpi = kpi.join(
        period_lookup,
        on="period",
        how="left",
        validate="m:1",
    )

    # -------------------------------------------------------------------------
    # Reorder final analytical fields
    # -------------------------------------------------------------------------

    kpi = kpi.select(
        [
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
            "total_gross_carrying_amount",
            "non_performing_amount",
            "npe_ratio",
            "npe_ratio_pct",
            "zero_denominator_flag",
        ]
    ).sort(["lei_code", "period"])

    passed("NPE Ratio mart created in memory.")

    return kpi


# =============================================================================
# PRE-SAVE VALIDATION
# =============================================================================

def validate_kpi_mart(kpi: pl.DataFrame) -> None:
    section("7. PRE-SAVE KPI MART VALIDATION")

    if kpi.height != EXPECTED_ROWS:
        fail(
            f"Unexpected KPI row count: {kpi.height:,}. "
            f"Expected {EXPECTED_ROWS:,}."
        )

    passed(f"Row count reconciles: {kpi.height:,} rows.")

    duplicate_groups = (
        kpi.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicate_groups.height != 0:
        fail(
            "KPI mart contains duplicate institution-period observations."
        )

    passed("Institution-period business grain is unique.")

    required_columns = [
        "lei_code",
        "institution_name",
        "period",
        "period_end_date",
        "total_gross_carrying_amount",
        "non_performing_amount",
        "npe_ratio",
        "npe_ratio_pct",
        "zero_denominator_flag",
        "is_aggregate",
    ]

    for column in required_columns:
        if column not in kpi.columns:
            fail(f"Required KPI column missing: {column}")

    passed("All required KPI mart columns are present.")

    null_institution_names = kpi["institution_name"].null_count()
    null_period_dates = kpi["period_end_date"].null_count()

    if null_institution_names != 0:
        fail(
            "Institution enrichment failed for "
            f"{null_institution_names} observation(s)."
        )

    if null_period_dates != 0:
        fail(
            "Period enrichment failed for "
            f"{null_period_dates} observation(s)."
        )

    passed("All KPI observations match the institution dimension.")
    passed("All KPI observations match the period dimension.")

    zero_denominator_count = (
        kpi.filter(pl.col("zero_denominator_flag")).height
    )

    ratio_null_count = kpi["npe_ratio"].null_count()

    info(f"Zero denominator observations: {zero_denominator_count:,}")
    info(f"NULL NPE ratio observations:   {ratio_null_count:,}")

    if ratio_null_count != zero_denominator_count:
        fail(
            "NULL NPE ratio count does not reconcile to "
            "zero-denominator observations."
        )

    passed("Zero-denominator handling reconciles correctly.")

    invalid_ratio_rows = kpi.filter(
        pl.col("npe_ratio").is_not_null()
        & (
            (pl.col("npe_ratio") < -NUMERIC_EPSILON)
            | (pl.col("npe_ratio") > 1.0 + NUMERIC_EPSILON)
        )
    )

    if invalid_ratio_rows.height != 0:
        print()
        print("Invalid ratio observations:")
        print(invalid_ratio_rows.head(20))

        fail(
            f"{invalid_ratio_rows.height} NPE ratio observation(s) "
            "fall materially outside [0, 1]."
        )

    passed("All non-NULL NPE ratios fall within the expected [0, 1] range.")

    negative_numerator_count = (
        kpi.filter(pl.col("non_performing_amount") < 0).height
    )

    negative_denominator_count = (
        kpi.filter(pl.col("total_gross_carrying_amount") < 0).height
    )

    info(f"Negative numerator observations:   {negative_numerator_count:,}")
    info(f"Negative denominator observations: {negative_denominator_count:,}")

    if negative_numerator_count != 0:
        fail(
            "Negative NPE numerator values require investigation before "
            "approving the KPI mart."
        )

    if negative_denominator_count != 0:
        fail(
            "Negative denominator values require investigation before "
            "approving the KPI mart."
        )

    passed("Numerator and denominator contain no negative values.")


# =============================================================================
# SAVE
# =============================================================================

def save_kpi_mart(kpi: pl.DataFrame) -> None:
    section("8. SAVE ANALYTICAL KPI MART")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        kpi.write_parquet(OUTPUT_PATH)
    except Exception as exc:
        fail(f"Unable to save NPE Ratio mart: {exc}")

    passed("NPE Ratio mart written successfully.")

    print()
    print(f"Saved analytical KPI mart:")
    print(f"  {OUTPUT_PATH}")


# =============================================================================
# POST-SAVE VALIDATION
# =============================================================================

def validate_saved_output(kpi: pl.DataFrame) -> None:
    section("9. POST-SAVE VALIDATION")

    if not OUTPUT_PATH.exists():
        fail("Saved NPE Ratio mart cannot be found.")

    passed(f"Saved file exists: {OUTPUT_PATH.name}")

    try:
        saved = pl.read_parquet(OUTPUT_PATH)
    except Exception as exc:
        fail(f"Unable to reopen saved NPE Ratio mart: {exc}")

    passed("Saved NPE Ratio mart reopened successfully.")

    if saved.height != kpi.height:
        fail(
            f"Saved row count {saved.height:,} does not match "
            f"in-memory row count {kpi.height:,}."
        )

    passed(f"Saved row count reconciles: {saved.height:,}")

    if saved.columns != kpi.columns:
        fail("Saved column order differs from validated in-memory mart.")

    passed("Saved column order reconciles.")

    if not saved.equals(kpi):
        fail("Saved Parquet does not exactly match validated in-memory mart.")

    passed("Saved Parquet exactly matches the validated in-memory mart.")

    duplicate_groups = (
        saved.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_groups != 0:
        fail("Saved mart contains duplicate institution-period rows.")

    passed("Saved institution-period grain remains unique.")

    # Summary profile
    profile = saved.select(
        [
            pl.len().alias("rows"),
            pl.col("lei_code").n_unique().alias("institutions"),
            pl.col("period").n_unique().alias("periods"),
            pl.col("zero_denominator_flag")
            .sum()
            .alias("zero_denominators"),
            pl.col("npe_ratio").null_count().alias("null_ratios"),
            pl.col("npe_ratio").min().alias("minimum_npe_ratio"),
            pl.col("npe_ratio").mean().alias("mean_npe_ratio"),
            pl.col("npe_ratio").median().alias("median_npe_ratio"),
            pl.col("npe_ratio").max().alias("maximum_npe_ratio"),
        ]
    )

    print()
    print("Final NPE Ratio mart profile:")
    print(profile)

    print()
    print("Sample observations:")
    print(
        saved.select(
            [
                "lei_code",
                "institution_name",
                "period",
                "total_gross_carrying_amount",
                "non_performing_amount",
                "npe_ratio",
                "npe_ratio_pct",
                "is_aggregate",
            ]
        ).head(15)
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print(SEPARATOR)
    print("BUILDING ANALYTICAL NPE RATIO MART")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    print(SEPARATOR)

    print(f"Credit-risk fact : {FACT_PATH}")
    print(f"Institution dim  : {INSTITUTION_DIM_PATH}")
    print(f"Period dim       : {PERIOD_DIM_PATH}")
    print(f"Output           : {OUTPUT_PATH}")

    print()
    print("KPI DEFINITION:")
    print(f"  Item                    : {TARGET_ITEM}")
    print(f"  Denominator perf_status : {TOTAL_PERF_STATUS}")
    print(f"  Numerator perf_status   : {NPE_PERF_STATUS}")
    print("  Grain                   : lei_code x period")
    print("  Zero denominator        : NULL ratio")
    print("  No thresholds or risk flags are calculated.")

    fact, institutions, periods = load_data()

    validate_source_columns(
        fact=fact,
        institutions=institutions,
        periods=periods,
    )

    denominator = extract_denominator(fact)
    numerator = extract_numerator(fact)

    validate_component_coverage(
        denominator=denominator,
        numerator=numerator,
    )

    kpi = build_kpi_mart(
        denominator=denominator,
        numerator=numerator,
        institutions=institutions,
        periods=periods,
    )

    validate_kpi_mart(kpi)

    save_kpi_mart(kpi)

    validate_saved_output(kpi)

    section("NPE RATIO MART BUILD COMPLETED SUCCESSFULLY")

    print(f"Output: {OUTPUT_PATH}")
    print()
    print("The approved numerator and denominator definitions were applied.")
    print("No raw, staging, fact, or dimension data was modified.")
    print("No management threshold, benchmark, ranking, or risk flag was applied.")
    print("The KPI mart has not yet received independent validation.")


if __name__ == "__main__":
    main()
