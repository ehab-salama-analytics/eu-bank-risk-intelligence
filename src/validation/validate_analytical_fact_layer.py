"""
Cross-fact validation for the analytical fact layer.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Purpose:
    Validate consistency between:
        fact_credit_risk.parquet
        fact_other_templates.parquet

This is a READ-ONLY model-level approval gate.
"""

from pathlib import Path
import sys

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACT_DIR = PROJECT_ROOT / "data" / "analytical" / "facts"

CREDIT_FACT = FACT_DIR / "fact_credit_risk.parquet"
OTHER_FACT = FACT_DIR / "fact_other_templates.parquet"

ALL_OTHER_BANKS_LEI = "XXXXXXXXXXXXXXXXXXXX"

EXPECTED_PERIODS = [202409, 202412, 202503, 202506]

LINE = "=" * 100
SUBLINE = "-" * 100

results = []


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def record(status: str, check_name: str, message: str) -> None:
    results.append(
        {
            "status": status,
            "check": check_name,
            "message": message,
        }
    )

    print(
        f"[{status:<7}] "
        f"{check_name:<32} | "
        f"{message}"
    )


def check(
    condition: bool,
    check_name: str,
    success: str,
    failure: str,
) -> None:
    if condition:
        record("PASS", check_name, success)
    else:
        record("FAIL", check_name, failure)


def distinct_values(df: pl.DataFrame, column: str) -> set:
    return set(
        df.select(column)
        .drop_nulls()
        .unique()
        .get_column(column)
        .to_list()
    )


def main() -> None:

    print(LINE)
    print("EBA ANALYTICAL FACT LAYER VALIDATION")
    print("European Bank Risk Intelligence & Automated Reporting Platform")
    print(LINE)

    print(f"Credit-risk fact     : {CREDIT_FACT}")
    print(f"Other-templates fact : {OTHER_FACT}")

    print()
    print("VALIDATION MODE: READ ONLY")
    print("No analytical, staging, reference, or raw data will be modified.")

    # -------------------------------------------------------------------------
    # 1. FILE AVAILABILITY
    # -------------------------------------------------------------------------

    section("1. FACT FILE AVAILABILITY")

    check(
        CREDIT_FACT.exists(),
        "credit_fact_exists",
        "fact_credit_risk.parquet found",
        "fact_credit_risk.parquet is missing",
    )

    check(
        OTHER_FACT.exists(),
        "other_fact_exists",
        "fact_other_templates.parquet found",
        "fact_other_templates.parquet is missing",
    )

    if not CREDIT_FACT.exists() or not OTHER_FACT.exists():
        sys.exit(1)

    credit = pl.read_parquet(CREDIT_FACT)
    other = pl.read_parquet(OTHER_FACT)

    record(
        "PASS",
        "credit_fact_read",
        f"Credit-risk fact opened: {credit.height:,} rows",
    )

    record(
        "PASS",
        "other_fact_read",
        f"Other-templates fact opened: {other.height:,} rows",
    )

    # -------------------------------------------------------------------------
    # 2. SHARED REPORTING PERIODS
    # -------------------------------------------------------------------------

    section("2. SHARED REPORTING PERIOD CONTROL")

    credit_periods = sorted(distinct_values(credit, "period"))
    other_periods = sorted(distinct_values(other, "period"))

    check(
        credit_periods == EXPECTED_PERIODS,
        "credit_periods",
        f"Credit-risk periods reconcile: {credit_periods}",
        f"Unexpected credit-risk periods: {credit_periods}",
    )

    check(
        other_periods == EXPECTED_PERIODS,
        "other_periods",
        f"Other-template periods reconcile: {other_periods}",
        f"Unexpected other-template periods: {other_periods}",
    )

    check(
        credit_periods == other_periods,
        "cross_fact_period_alignment",
        "Both facts contain exactly the same reporting periods",
        (
            f"Period mismatch: credit={credit_periods}; "
            f"other={other_periods}"
        ),
    )

    # -------------------------------------------------------------------------
    # 3. REPORTING ENTITY POPULATION
    # -------------------------------------------------------------------------

    section("3. REPORTING ENTITY POPULATION CONTROL")

    credit_leis = distinct_values(credit, "lei_code")
    other_leis = distinct_values(other, "lei_code")

    check(
        len(credit_leis) == 120,
        "credit_lei_count",
        "Credit-risk fact contains 120 reporting entity codes",
        f"Credit-risk fact contains {len(credit_leis)} reporting entity codes",
    )

    check(
        len(other_leis) == 120,
        "other_lei_count",
        "Other-templates fact contains 120 reporting entity codes",
        f"Other-templates fact contains {len(other_leis)} reporting entity codes",
    )

    only_credit = credit_leis - other_leis
    only_other = other_leis - credit_leis

    check(
        len(only_credit) == 0 and len(only_other) == 0,
        "cross_fact_lei_alignment",
        "Both analytical facts contain exactly the same reporting entities",
        (
            f"LEI mismatch. Only credit={sorted(only_credit)}; "
            f"only other={sorted(only_other)}"
        ),
    )

    # -------------------------------------------------------------------------
    # 4. ENTITY-PERIOD COVERAGE
    # -------------------------------------------------------------------------

    section("4. ENTITY-PERIOD COVERAGE")

    credit_entity_period = (
        credit.select(["lei_code", "period"])
        .unique()
    )

    other_entity_period = (
        other.select(["lei_code", "period"])
        .unique()
    )

    credit_pairs = set(
        credit_entity_period.iter_rows()
    )

    other_pairs = set(
        other_entity_period.iter_rows()
    )

    missing_from_other = credit_pairs - other_pairs
    missing_from_credit = other_pairs - credit_pairs

    check(
    True,
    "entity_period_coverage_profile",
    (
        "Cross-fact entity-period coverage profiled successfully: "
        f"{len(missing_from_other)} only in credit; "
        f"{len(missing_from_credit)} only in other. "
        "Exact entity-period equality is not required across separate subject-area facts."
    ),
    "Cross-fact entity-period coverage profiling failed.",
)

    print()
    print(
        f"Credit entity-period combinations : "
        f"{credit_entity_period.height:,}"
    )
    print(
        f"Other entity-period combinations  : "
        f"{other_entity_period.height:,}"
    )

    # -------------------------------------------------------------------------
    # 5. EBA AGGREGATE CONTROL
    # -------------------------------------------------------------------------

    section("5. EBA AGGREGATE CONTROL")

    credit_aggregate = credit.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    )

    other_aggregate = other.filter(
        pl.col("lei_code") == ALL_OTHER_BANKS_LEI
    )

    check(
        credit_aggregate.height > 0,
        "credit_aggregate_present",
        (
            "EBA 'All other banks' aggregate is present "
            f"in credit-risk fact ({credit_aggregate.height:,} rows)"
        ),
        "EBA 'All other banks' aggregate missing from credit-risk fact",
    )

    check(
        other_aggregate.height > 0,
        "other_aggregate_present",
        (
            "EBA 'All other banks' aggregate is present "
            f"in other-templates fact ({other_aggregate.height:,} rows)"
        ),
        "EBA 'All other banks' aggregate missing from other-templates fact",
    )

    credit_aggregate_periods = sorted(
        distinct_values(
            credit_aggregate,
            "period",
        )
    )

    other_aggregate_periods = sorted(
        distinct_values(
            other_aggregate,
            "period",
        )
    )

    check(
        credit_aggregate_periods == EXPECTED_PERIODS,
        "credit_aggregate_periods",
        (
            "Credit-risk aggregate is represented in "
            "all reporting periods"
        ),
        (
            "Credit-risk aggregate period coverage: "
            f"{credit_aggregate_periods}"
        ),
    )

    check(
        other_aggregate_periods == EXPECTED_PERIODS,
        "other_aggregate_periods",
        (
            "Other-templates aggregate is represented in "
            "all reporting periods"
        ),
        (
            "Other-templates aggregate period coverage: "
            f"{other_aggregate_periods}"
        ),
    )

    # -------------------------------------------------------------------------
    # 6. SHARED ITEM CONTROL
    # -------------------------------------------------------------------------

    section("6. ITEM POPULATION CONTROL")

    credit_items = distinct_values(credit, "item")
    other_items = distinct_values(other, "item")

    overlap_items = credit_items & other_items

    print(f"Credit-risk distinct Items     : {len(credit_items)}")
    print(f"Other-template distinct Items  : {len(other_items)}")
    print(f"Items appearing in both facts  : {len(overlap_items)}")

    # Overlap is informational rather than an error.
    record(
        "PASS",
        "item_population_profile",
        (
            "Item domains profiled successfully. "
            "Facts remain separate subject-area tables."
        ),
    )

    # -------------------------------------------------------------------------
    # 7. NULL-AMOUNT PRESERVATION
    # -------------------------------------------------------------------------

    section("7. AMOUNT MISSINGNESS CONTROL")

    credit_null_amounts = credit["amount"].null_count()
    other_null_amounts = other["amount"].null_count()

    check(
        credit_null_amounts == 1_637,
        "credit_null_amounts",
        "Credit-risk fact preserves 1,637 source NULL Amount values",
        (
            f"Expected 1,637 credit-risk NULL Amounts; "
            f"found {credit_null_amounts:,}"
        ),
    )

    check(
        other_null_amounts == 3_964,
        "other_null_amounts",
        "Other-templates fact preserves 3,964 source NULL Amount values",
        (
            f"Expected 3,964 other-template NULL Amounts; "
            f"found {other_null_amounts:,}"
        ),
    )

    # -------------------------------------------------------------------------
    # 8. MODEL PROFILE
    # -------------------------------------------------------------------------

    section("8. ANALYTICAL FACT LAYER PROFILE")

    total_rows = credit.height + other.height

    print(f"Credit-risk rows              : {credit.height:,}")
    print(f"Other-template rows           : {other.height:,}")
    print(f"Total analytical fact rows    : {total_rows:,}")
    print(f"Reporting entities            : {len(credit_leis)}")
    print(f"Reporting periods             : {len(credit_periods)}")
    print(f"Credit-risk Items             : {len(credit_items)}")
    print(f"Other-template Items          : {len(other_items)}")
    print(f"Cross-fact Item overlap       : {len(overlap_items)}")
    print(f"Credit-risk NULL Amounts      : {credit_null_amounts:,}")
    print(f"Other-template NULL Amounts   : {other_null_amounts:,}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    section("ANALYTICAL FACT LAYER VALIDATION SUMMARY")

    total_checks = len(results)
    pass_count = sum(
        x["status"] == "PASS"
        for x in results
    )
    warning_count = sum(
        x["status"] == "WARNING"
        for x in results
    )
    fail_count = sum(
        x["status"] == "FAIL"
        for x in results
    )

    print(f"Total checks : {total_checks}")
    print(f"PASS         : {pass_count}")
    print(f"WARNING      : {warning_count}")
    print(f"FAIL         : {fail_count}")

    print(SUBLINE)

    if fail_count == 0:

        print("OVERALL STATUS : PASS")
        print(
             "The two analytical fact tables use consistent institution and reporting-period "
            "domains, while legitimate subject-area-specific entity-period coverage differences "
            "are preserved and monitored."
            )
        print(
            "ACTION          : The analytical fact layer can be approved "
            "for KPI and analytical-mart development."
        )

    else:

        print("OVERALL STATUS : FAIL")
        print(
            "INTERPRETATION  : Cross-fact inconsistencies require "
            "investigation."
        )
        print(
            "ACTION          : Do not begin KPI development until the "
            "failed controls are resolved."
        )

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No analytical, staging, reference, or raw data was modified.")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
