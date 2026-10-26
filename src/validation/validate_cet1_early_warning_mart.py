from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_trend.parquet"
)

MART_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_early_warning.parquet"
)


EW_COLUMNS = [
    "early_warning_eligible",
    "direct_cet1_deterioration",
    "eu_peer_deterioration",
    "country_peer_deterioration_available",
    "country_peer_deterioration",
    "persistent_deterioration_available",
    "persistent_deterioration",
    "supporting_signal_count",
    "early_warning_status",
]


EXPECTED_DTYPES = {
    "early_warning_eligible": pl.Boolean,
    "direct_cet1_deterioration": pl.Boolean,
    "eu_peer_deterioration": pl.Boolean,
    "country_peer_deterioration_available": pl.Boolean,
    "country_peer_deterioration": pl.Boolean,
    "persistent_deterioration_available": pl.Boolean,
    "persistent_deterioration": pl.Boolean,
    "supporting_signal_count": pl.Int8,
    "early_warning_status": pl.String,
}


checks_passed = 0
checks_failed = 0


def header(title: str) -> None:
    print()
    print("=" * 105)
    print(title)
    print("=" * 105)


def check(condition: bool, message: str) -> None:
    global checks_passed
    global checks_failed

    if condition:
        checks_passed += 1
        print(f"[PASS] {message}")
    else:
        checks_failed += 1
        print(f"[FAIL] {message}")


def main() -> None:

    global checks_passed
    global checks_failed

    header("CET1 EARLY-WARNING MART INDEPENDENT VALIDATION")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source trend : {SOURCE_PATH}")
    print(f"EW mart      : {MART_PATH}")

    print()
    print("VALIDATION MODE: READ ONLY")
    print("No data will be modified.")

    # --------------------------------------------------------------
    # 1. FILE AVAILABILITY
    # --------------------------------------------------------------

    header("1. FILE AVAILABILITY")

    for path in [SOURCE_PATH, MART_PATH]:

        exists = path.exists()

        check(
            exists,
            f"Found: {path.name}",
        )

        if not exists:
            raise FileNotFoundError(path)

    # --------------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------------

    header("2. LOAD VALIDATION INPUTS")

    source = pl.read_parquet(SOURCE_PATH)
    saved = pl.read_parquet(MART_PATH)

    check(
        source.height == 469,
        "Source CET1 trend mart contains 469 rows.",
    )

    check(
        source.width == 61,
        "Source CET1 trend mart contains 61 columns.",
    )

    check(
        saved.height == 469,
        "Saved CET1 early-warning mart contains 469 rows.",
    )

    check(
        saved.width == 70,
        "Saved CET1 early-warning mart contains 70 columns.",
    )

    # --------------------------------------------------------------
    # 3. GRAIN VALIDATION
    # --------------------------------------------------------------

    header("3. GRAIN VALIDATION")

    source_dupes = (
        source.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    saved_dupes = (
        saved.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    check(
        source_dupes.height == 0,
        "Source trend mart is unique at lei_code x period.",
    )

    check(
        saved_dupes.height == 0,
        "Saved early-warning mart is unique at lei_code x period.",
    )

    # --------------------------------------------------------------
    # 4. SOURCE FIELD PRESERVATION
    # --------------------------------------------------------------

    header("4. SOURCE TREND FIELD PRESERVATION")

    source_columns = source.columns

    check(
        saved.columns[:61] == source_columns,
        "All 61 source trend columns are preserved in original order.",
    )

    source_sorted = source.sort(
        ["lei_code", "period"]
    )

    saved_sorted = saved.sort(
        ["lei_code", "period"]
    )

    check(
        saved_sorted.select(source_columns).equals(
            source_sorted.select(source_columns)
        ),
        "All 61 source trend fields are preserved exactly.",
    )

    # --------------------------------------------------------------
    # 5. EARLY-WARNING SCHEMA
    # --------------------------------------------------------------

    header("5. EARLY-WARNING FIELD SCHEMA")

    check(
        saved.columns[61:] == EW_COLUMNS,
        "All nine early-warning fields are appended in approved order.",
    )

    for column, expected_dtype in EXPECTED_DTYPES.items():

        actual_dtype = saved.schema[column]

        check(
            actual_dtype == expected_dtype,
            f"{column} dtype is {expected_dtype}. Actual: {actual_dtype}.",
        )

    # --------------------------------------------------------------
    # 6. INDEPENDENT SIGNAL RECONSTRUCTION
    # --------------------------------------------------------------

    header("6. INDEPENDENT SIGNAL RECONSTRUCTION")

    expected = source.with_columns(
        [
            (~pl.col("is_aggregate"))
            .alias("expected_early_warning_eligible"),

            pl.when(
                (~pl.col("is_aggregate"))
                & pl.col("qoq_available")
            )
            .then(
                pl.col("cet1_ratio_change_qoq") < 0
            )
            .otherwise(None)
            .alias(
                "expected_direct_cet1_deterioration"
            ),

            pl.when(
                (~pl.col("is_aggregate"))
                & pl.col("eu_percentile_qoq_available")
            )
            .then(
                pl.col("eu_percentile_change_qoq") < 0
            )
            .otherwise(None)
            .alias(
                "expected_eu_peer_deterioration"
            ),

            (
                (~pl.col("is_aggregate"))
                & pl.col(
                    "country_percentile_qoq_available"
                )
            )
            .alias(
                "expected_country_peer_deterioration_available"
            ),

            pl.when(
                (~pl.col("is_aggregate"))
                & pl.col(
                    "country_percentile_qoq_available"
                )
            )
            .then(
                pl.col(
                    "country_percentile_change_qoq"
                ) < 0
            )
            .otherwise(None)
            .alias(
                "expected_country_peer_deterioration"
            ),

            (
                (~pl.col("is_aggregate"))
                & pl.col(
                    "three_period_pattern_available"
                )
            )
            .alias(
                "expected_persistent_deterioration_available"
            ),

            pl.when(
                (~pl.col("is_aggregate"))
                & pl.col(
                    "three_period_pattern_available"
                )
            )
            .then(
                pl.col("cet1_three_period_pattern")
                == "consecutive_decrease"
            )
            .otherwise(None)
            .alias(
                "expected_persistent_deterioration"
            ),
        ]
    )

    expected = expected.with_columns(
        pl.when(
            (~pl.col("is_aggregate"))
            & pl.col("qoq_available")
        )
        .then(
            pl.col(
                "expected_eu_peer_deterioration"
            )
            .fill_null(False)
            .cast(pl.Int8)
            +
            pl.col(
                "expected_country_peer_deterioration"
            )
            .fill_null(False)
            .cast(pl.Int8)
            +
            pl.col(
                "expected_persistent_deterioration"
            )
            .fill_null(False)
            .cast(pl.Int8)
        )
        .otherwise(None)
        .cast(pl.Int8)
        .alias(
            "expected_supporting_signal_count"
        )
    )

    # --------------------------------------------------------------
    # 7. INDEPENDENT STATUS RECONSTRUCTION
    # --------------------------------------------------------------

    header("7. INDEPENDENT STATUS RECONSTRUCTION")

    expected = expected.with_columns(
        pl.when(
            pl.col("is_aggregate")
        )
        .then(
            pl.lit("NOT_ELIGIBLE_AGGREGATE")
        )

        .when(
            ~pl.col("qoq_available")
        )
        .then(
            pl.lit("NOT_ASSESSABLE")
        )

        .when(
            pl.col(
                "expected_direct_cet1_deterioration"
            ) == False
        )
        .then(
            pl.lit(
                "NO_CURRENT_DETERIORATION_SIGNAL"
            )
        )

        .when(
            (
                pl.col(
                    "expected_direct_cet1_deterioration"
                ) == True
            )
            &
            (
                pl.col(
                    "expected_persistent_deterioration"
                ) == True
            )
            &
            (
                (
                    pl.col(
                        "expected_eu_peer_deterioration"
                    ) == True
                )
                |
                (
                    pl.col(
                        "expected_country_peer_deterioration"
                    ) == True
                )
            )
        )
        .then(
            pl.lit("STRONG_ATTENTION")
        )

        .when(
            (
                pl.col(
                    "expected_direct_cet1_deterioration"
                ) == True
            )
            &
            (
                pl.col(
                    "expected_supporting_signal_count"
                ) >= 1
            )
        )
        .then(
            pl.lit("ELEVATED_ATTENTION")
        )

        .when(
            (
                pl.col(
                    "expected_direct_cet1_deterioration"
                ) == True
            )
            &
            (
                pl.col(
                    "expected_supporting_signal_count"
                ) == 0
            )
        )
        .then(
            pl.lit("WATCH")
        )

        .otherwise(
            pl.lit("NOT_ASSESSABLE")
        )
        .alias(
            "expected_early_warning_status"
        )
    )

    # --------------------------------------------------------------
    # 8. ALIGN EXPECTED AND SAVED
    # --------------------------------------------------------------

    header("8. EXPECTED-TO-SAVED ALIGNMENT")

    expected_fields = expected.select(
        [
            "lei_code",
            "period",
            "expected_early_warning_eligible",
            "expected_direct_cet1_deterioration",
            "expected_eu_peer_deterioration",
            "expected_country_peer_deterioration_available",
            "expected_country_peer_deterioration",
            "expected_persistent_deterioration_available",
            "expected_persistent_deterioration",
            "expected_supporting_signal_count",
            "expected_early_warning_status",
        ]
    )

    saved_fields = saved.select(
        [
            "lei_code",
            "period",
            *EW_COLUMNS,
        ]
    )

    joined = expected_fields.join(
        saved_fields,
        on=["lei_code", "period"],
        how="inner",
    )

    check(
        joined.height == 469,
        "Expected observations align one-to-one with saved mart.",
    )

    # --------------------------------------------------------------
    # 9. FIELD RECONCILIATION
    # --------------------------------------------------------------

    header("9. EARLY-WARNING FIELD RECONCILIATION")

    mapping = {
        "early_warning_eligible":
            "expected_early_warning_eligible",

        "direct_cet1_deterioration":
            "expected_direct_cet1_deterioration",

        "eu_peer_deterioration":
            "expected_eu_peer_deterioration",

        "country_peer_deterioration_available":
            "expected_country_peer_deterioration_available",

        "country_peer_deterioration":
            "expected_country_peer_deterioration",

        "persistent_deterioration_available":
            "expected_persistent_deterioration_available",

        "persistent_deterioration":
            "expected_persistent_deterioration",

        "supporting_signal_count":
            "expected_supporting_signal_count",

        "early_warning_status":
            "expected_early_warning_status",
    }

    for saved_col, expected_col in mapping.items():

        mismatches = joined.filter(
            (
                pl.col(saved_col).is_null()
                != pl.col(expected_col).is_null()
            )
            |
            (
                pl.col(saved_col).is_not_null()
                & pl.col(expected_col).is_not_null()
                & (
                    pl.col(saved_col)
                    != pl.col(expected_col)
                )
            )
        )

        check(
            mismatches.height == 0,
            f"{saved_col} independently reconciles exactly.",
        )

        if mismatches.height > 0:
            print(
                mismatches
                .select(
                    [
                        "lei_code",
                        "period",
                        expected_col,
                        saved_col,
                    ]
                )
                .head(20)
            )

    # --------------------------------------------------------------
    # 10. SIGNAL PROFILE
    # --------------------------------------------------------------

    header("10. SIGNAL PROFILE")

    institutions = saved.filter(
        ~pl.col("is_aggregate")
    )

    direct_count = institutions.filter(
        pl.col("direct_cet1_deterioration")
        == True
    ).height

    eu_count = institutions.filter(
        pl.col("eu_peer_deterioration")
        == True
    ).height

    country_count = institutions.filter(
        pl.col("country_peer_deterioration")
        == True
    ).height

    persistent_count = institutions.filter(
        pl.col("persistent_deterioration")
        == True
    ).height

    print(
        f"Direct CET1 deterioration     : {direct_count}"
    )
    print(
        f"EU peer deterioration         : {eu_count}"
    )
    print(
        f"Country peer deterioration    : {country_count}"
    )
    print(
        f"Persistent deterioration      : {persistent_count}"
    )

    check(
        direct_count == 152,
        "Direct CET1 deterioration count is 152.",
    )

    check(
        eu_count == 182,
        "EU peer deterioration count is 182.",
    )

    check(
        country_count == 47,
        "Country peer deterioration count is 47.",
    )

    check(
        persistent_count == 42,
        "Persistent deterioration count is 42.",
    )

    # --------------------------------------------------------------
    # 11. STATUS PROFILE
    # --------------------------------------------------------------

    header("11. MONITORING STATUS PROFILE")

    status_profile = (
        saved.group_by("early_warning_status")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("lei_code")
                .n_unique()
                .alias("institutions"),
            ]
        )
        .sort("early_warning_status")
    )

    print(status_profile)

    expected_status_counts = {
        "ELEVATED_ATTENTION": 106,
        "NOT_ASSESSABLE": 127,
        "NOT_ELIGIBLE_AGGREGATE": 4,
        "NO_CURRENT_DETERIORATION_SIGNAL": 186,
        "STRONG_ATTENTION": 37,
        "WATCH": 9,
    }

    for status, expected_count in (
        expected_status_counts.items()
    ):

        actual = saved.filter(
            pl.col("early_warning_status")
            == status
        ).height

        check(
            actual == expected_count,
            f"{status} contains {expected_count} rows.",
        )

    # --------------------------------------------------------------
    # 12. SUPPORTING SIGNAL COUNT CONTROL
    # --------------------------------------------------------------

    header("12. SUPPORTING SIGNAL COUNT CONTROL")

    assessable = institutions.filter(
        pl.col("qoq_available")
    )

    check(
        assessable.filter(
            pl.col("supporting_signal_count").is_null()
        ).height == 0,
        "All QoQ-assessable institutions have supporting signal count.",
    )

    check(
        assessable.filter(
            ~pl.col("supporting_signal_count")
            .is_between(0, 3)
        ).height == 0,
        "Supporting signal count remains between 0 and 3.",
    )

    not_assessable = institutions.filter(
        ~pl.col("qoq_available")
    )

    check(
        not_assessable.filter(
            pl.col("supporting_signal_count").is_not_null()
        ).height == 0,
        "Supporting signal count is NULL where QoQ is unavailable.",
    )

    print()
    print(
        assessable
        .group_by("supporting_signal_count")
        .agg(
            pl.len().alias("rows")
        )
        .sort("supporting_signal_count")
    )

    # --------------------------------------------------------------
    # 13. COUNTRY AVAILABILITY SEMANTICS
    # --------------------------------------------------------------

    header("13. COUNTRY SIGNAL AVAILABILITY")

    check(
        institutions.filter(
            ~pl.col(
                "country_peer_deterioration_available"
            )
            & pl.col(
                "country_peer_deterioration"
            ).is_not_null()
        ).height == 0,
        (
            "Country deterioration remains NULL "
            "where formal country comparison is unavailable."
        ),
    )

    # --------------------------------------------------------------
    # 14. PERSISTENCE AVAILABILITY SEMANTICS
    # --------------------------------------------------------------

    header("14. PERSISTENT SIGNAL AVAILABILITY")

    check(
        institutions.filter(
            ~pl.col(
                "persistent_deterioration_available"
            )
            & pl.col(
                "persistent_deterioration"
            ).is_not_null()
        ).height == 0,
        (
            "Persistent deterioration remains NULL "
            "where three-period history is unavailable."
        ),
    )

    # --------------------------------------------------------------
    # 15. BUSINESS RULE CONTROLS
    # --------------------------------------------------------------

    header("15. BUSINESS RULE CONTROLS")

    check(
        saved.filter(
            pl.col("is_aggregate")
            & (
                pl.col("early_warning_status")
                != "NOT_ELIGIBLE_AGGREGATE"
            )
        ).height == 0,
        "Aggregate status rule is correct.",
    )

    check(
        institutions.filter(
            (~pl.col("qoq_available"))
            & (
                pl.col("early_warning_status")
                != "NOT_ASSESSABLE"
            )
        ).height == 0,
        "Missing QoQ history maps to NOT_ASSESSABLE.",
    )

    check(
        institutions.filter(
            pl.col("qoq_available")
            & (
                pl.col("direct_cet1_deterioration")
                == False
            )
            & (
                pl.col("early_warning_status")
                != "NO_CURRENT_DETERIORATION_SIGNAL"
            )
        ).height == 0,
        "No direct deterioration maps to NO_CURRENT_DETERIORATION_SIGNAL.",
    )

    check(
        saved.filter(
            pl.col("early_warning_status")
            == "WATCH"
        ).filter(
            ~(
                (
                    pl.col(
                        "direct_cet1_deterioration"
                    ) == True
                )
                &
                (
                    pl.col(
                        "supporting_signal_count"
                    ) == 0
                )
            )
        ).height == 0,
        "WATCH rule is correct.",
    )

    check(
        saved.filter(
            pl.col("early_warning_status")
            == "ELEVATED_ATTENTION"
        ).filter(
            ~(
                (
                    pl.col(
                        "direct_cet1_deterioration"
                    ) == True
                )
                &
                (
                    pl.col(
                        "supporting_signal_count"
                    ) >= 1
                )
            )
        ).height == 0,
        "ELEVATED_ATTENTION rule is correct.",
    )

    check(
        saved.filter(
            pl.col("early_warning_status")
            == "STRONG_ATTENTION"
        ).filter(
            ~(
                (
                    pl.col(
                        "direct_cet1_deterioration"
                    ) == True
                )
                &
                (
                    pl.col(
                        "persistent_deterioration"
                    ) == True
                )
                &
                (
                    (
                        pl.col(
                            "eu_peer_deterioration"
                        ) == True
                    )
                    |
                    (
                        pl.col(
                            "country_peer_deterioration"
                        ) == True
                    )
                )
            )
        ).height == 0,
        "STRONG_ATTENTION rule is correct.",
    )

    # --------------------------------------------------------------
    # 16. AGGREGATE EXCLUSION
    # --------------------------------------------------------------

    header("16. AGGREGATE EXCLUSION")

    aggregates = saved.filter(
        pl.col("is_aggregate")
    )

    check(
        aggregates.height == 4,
        "Mart contains four aggregate rows.",
    )

    check(
        aggregates.filter(
            pl.col("early_warning_eligible")
        ).height == 0,
        "Aggregate rows are not early-warning eligible.",
    )

    # --------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------

    header(
        "CET1 EARLY-WARNING MART "
        "INDEPENDENT VALIDATION SUMMARY"
    )

    total_checks = (
        checks_passed + checks_failed
    )

    print(f"Total checks : {total_checks}")
    print(f"PASS         : {checks_passed}")
    print(f"FAIL         : {checks_failed}")

    print()

    if checks_failed == 0:

        print("OVERALL STATUS : PASS")

        print()
        print(
            "ACTION: mart_cet1_early_warning can be "
            "approved for downstream CET1 management "
            "monitoring and reporting."
        )

    else:

        print("OVERALL STATUS : FAIL")

        print()
        print(
            "ACTION: Do not approve the CET1 "
            "early-warning mart."
        )

        raise SystemExit(1)

    print()
    print("VALIDATION MODE : READ ONLY")
    print("No data was modified.")


if __name__ == "__main__":
    main()
