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

OUTPUT_PATH = (
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


def header(title: str) -> None:
    print()
    print("=" * 105)
    print(title)
    print("=" * 105)


def main() -> None:

    header("BUILD CET1 EARLY-WARNING ANALYTICAL MART")

    print(f"Source : {SOURCE_PATH}")
    print(f"Output : {OUTPUT_PATH}")

    # 1
    header("1. FILE AVAILABILITY")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    print("[PASS] Source trend mart found.")

    # 2
    header("2. LOAD APPROVED CET1 TREND MART")

    df = pl.read_parquet(SOURCE_PATH)

    if df.height != 469:
        raise ValueError(
            f"Expected 469 rows, found {df.height}"
        )

    if df.width != 61:
        raise ValueError(
            f"Expected 61 columns, found {df.width}"
        )

    print(
        f"[PASS] Source opened: "
        f"{df.height} rows x {df.width} columns"
    )

    # 3
    header("3. BUILD EARLY-WARNING SIGNALS")

    mart = df.with_columns(
        [
            (~pl.col("is_aggregate"))
            .alias("early_warning_eligible"),

            pl.when(
                (~pl.col("is_aggregate"))
                & pl.col("qoq_available")
            )
            .then(
                pl.col("cet1_ratio_change_qoq") < 0
            )
            .otherwise(None)
            .alias("direct_cet1_deterioration"),

            pl.when(
                (~pl.col("is_aggregate"))
                & pl.col("eu_percentile_qoq_available")
            )
            .then(
                pl.col("eu_percentile_change_qoq") < 0
            )
            .otherwise(None)
            .alias("eu_peer_deterioration"),

            (
                (~pl.col("is_aggregate"))
                & pl.col(
                    "country_percentile_qoq_available"
                )
            )
            .alias(
                "country_peer_deterioration_available"
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
            .alias("country_peer_deterioration"),

            (
                (~pl.col("is_aggregate"))
                & pl.col(
                    "three_period_pattern_available"
                )
            )
            .alias(
                "persistent_deterioration_available"
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
            .alias("persistent_deterioration"),
        ]
    )

    # 4
    header("4. CALCULATE SUPPORTING SIGNAL COUNT")

    mart = mart.with_columns(
        pl.when(
            (~pl.col("is_aggregate"))
            & pl.col("qoq_available")
        )
        .then(
            pl.col("eu_peer_deterioration")
            .fill_null(False)
            .cast(pl.Int8)
            +
            pl.col("country_peer_deterioration")
            .fill_null(False)
            .cast(pl.Int8)
            +
            pl.col("persistent_deterioration")
            .fill_null(False)
            .cast(pl.Int8)
        )
        .otherwise(None)
        .cast(pl.Int8)
        .alias("supporting_signal_count")
    )

    # 5
    header("5. ASSIGN MONITORING STATUS")

    mart = mart.with_columns(
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
            pl.col("direct_cet1_deterioration")
            == False
        )
        .then(
            pl.lit(
                "NO_CURRENT_DETERIORATION_SIGNAL"
            )
        )

        .when(
            (
                pl.col("direct_cet1_deterioration")
                == True
            )
            &
            (
                pl.col("persistent_deterioration")
                == True
            )
            &
            (
                (
                    pl.col("eu_peer_deterioration")
                    == True
                )
                |
                (
                    pl.col(
                        "country_peer_deterioration"
                    )
                    == True
                )
            )
        )
        .then(
            pl.lit("STRONG_ATTENTION")
        )

        .when(
            (
                pl.col("direct_cet1_deterioration")
                == True
            )
            &
            (
                pl.col("supporting_signal_count")
                >= 1
            )
        )
        .then(
            pl.lit("ELEVATED_ATTENTION")
        )

        .when(
            (
                pl.col("direct_cet1_deterioration")
                == True
            )
            &
            (
                pl.col("supporting_signal_count")
                == 0
            )
        )
        .then(
            pl.lit("WATCH")
        )

        .otherwise(
            pl.lit("NOT_ASSESSABLE")
        )
        .alias("early_warning_status")
    )

    # 6
    header("6. VALIDATE STRUCTURE")

    expected_columns = 61 + len(EW_COLUMNS)

    if mart.height != 469:
        raise ValueError(
            f"Expected 469 rows, found {mart.height}"
        )

    if mart.width != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns, "
            f"found {mart.width}"
        )

    duplicates = (
        mart.group_by(["lei_code", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height > 0:
        raise ValueError(
            "Duplicate lei_code x period rows found."
        )

    print(
        f"[PASS] Rows: {mart.height}"
    )
    print(
        f"[PASS] Columns: {mart.width}"
    )
    print(
        "[PASS] Grain unique: lei_code x period"
    )

    # 7
    header("7. SIGNAL PROFILE")

    institutions = mart.filter(
        ~pl.col("is_aggregate")
    )

    print(
        "Direct CET1 deterioration:",
        institutions.filter(
            pl.col("direct_cet1_deterioration")
            == True
        ).height,
    )

    print(
        "EU peer deterioration:",
        institutions.filter(
            pl.col("eu_peer_deterioration")
            == True
        ).height,
    )

    print(
        "Country peer deterioration:",
        institutions.filter(
            pl.col("country_peer_deterioration")
            == True
        ).height,
    )

    print(
        "Persistent deterioration:",
        institutions.filter(
            pl.col("persistent_deterioration")
            == True
        ).height,
    )

    # 8
    header("8. MONITORING STATUS PROFILE")

    print(
        mart.group_by(
            "early_warning_status"
        )
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

    # 9
    header("9. BUSINESS RULE CONTROLS")

    invalid_aggregate = mart.filter(
        pl.col("is_aggregate")
        & (
            pl.col("early_warning_status")
            != "NOT_ELIGIBLE_AGGREGATE"
        )
    )

    if invalid_aggregate.height > 0:
        raise ValueError(
            "Aggregate status rule failed."
        )

    print("[PASS] Aggregate treatment correct.")

    invalid_no_qoq = mart.filter(
        (~pl.col("is_aggregate"))
        & (~pl.col("qoq_available"))
        & (
            pl.col("early_warning_status")
            != "NOT_ASSESSABLE"
        )
    )

    if invalid_no_qoq.height > 0:
        raise ValueError(
            "NOT_ASSESSABLE rule failed."
        )

    print("[PASS] Missing QoQ history treatment correct.")

    invalid_no_deterioration = mart.filter(
        (~pl.col("is_aggregate"))
        & pl.col("qoq_available")
        & (
            pl.col("direct_cet1_deterioration")
            == False
        )
        & (
            pl.col("early_warning_status")
            != "NO_CURRENT_DETERIORATION_SIGNAL"
        )
    )

    if invalid_no_deterioration.height > 0:
        raise ValueError(
            "No-deterioration rule failed."
        )

    print(
        "[PASS] No-current-deterioration rule correct."
    )

    invalid_watch = mart.filter(
        pl.col("early_warning_status")
        == "WATCH"
    ).filter(
        ~(
            (
                pl.col("direct_cet1_deterioration")
                == True
            )
            &
            (
                pl.col("supporting_signal_count")
                == 0
            )
        )
    )

    if invalid_watch.height > 0:
        raise ValueError(
            "WATCH rule failed."
        )

    print("[PASS] WATCH rule correct.")

    invalid_strong = mart.filter(
        pl.col("early_warning_status")
        == "STRONG_ATTENTION"
    ).filter(
        ~(
            (
                pl.col("direct_cet1_deterioration")
                == True
            )
            &
            (
                pl.col("persistent_deterioration")
                == True
            )
            &
            (
                (
                    pl.col("eu_peer_deterioration")
                    == True
                )
                |
                (
                    pl.col(
                        "country_peer_deterioration"
                    )
                    == True
                )
            )
        )
    )

    if invalid_strong.height > 0:
        raise ValueError(
            "STRONG_ATTENTION rule failed."
        )

    print(
        "[PASS] STRONG_ATTENTION rule correct."
    )

    # 10
    header("10. SAVE CET1 EARLY-WARNING MART")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mart.write_parquet(OUTPUT_PATH)

    print(f"[PASS] Saved: {OUTPUT_PATH}")

    # 11
    header("11. POST-SAVE VALIDATION")

    saved = pl.read_parquet(OUTPUT_PATH)

    if not saved.equals(mart):
        raise ValueError(
            "Saved mart does not match in-memory mart."
        )

    print(
        f"[PASS] Reopened: "
        f"{saved.height} rows x {saved.width} columns"
    )

    print(
        "[PASS] Saved Parquet exactly matches "
        "in-memory mart."
    )

    header("CET1 EARLY-WARNING MART BUILD COMPLETE")

    print(f"Rows    : {saved.height}")
    print(f"Columns : {saved.width}")
    print(f"Output  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
