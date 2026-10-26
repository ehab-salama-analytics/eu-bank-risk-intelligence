from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "facts"
    / "fact_other_templates.parquet"
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "marts"
    / "mart_cet1_ratio.parquet"
)


# Approved EBA items
CET1_CAPITAL_ITEM = 2520102
TOTAL_RISK_EXPOSURE_ITEM = 2520138
CET1_RATIO_ITEM = 2520140

RECONCILIATION_TOLERANCE = 1e-12


def header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


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


def main() -> None:

    header("BUILD CET1 RATIO ANALYTICAL MART")

    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Source fact      : {FACT_PATH}")
    print(f"Institution dim  : {INSTITUTION_DIM_PATH}")
    print(f"Period dim       : {PERIOD_DIM_PATH}")
    print(f"Output mart      : {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 1. FILE AVAILABILITY
    # ------------------------------------------------------------------

    header("1. FILE AVAILABILITY")

    required_files = [
        FACT_PATH,
        INSTITUTION_DIM_PATH,
        PERIOD_DIM_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        print(f"[PASS] Found: {path.name}")

    # ------------------------------------------------------------------
    # 2. LOAD DATA
    # ------------------------------------------------------------------

    header("2. LOAD APPROVED ANALYTICAL DATA")

    fact = pl.read_parquet(FACT_PATH)
    institutions = pl.read_parquet(INSTITUTION_DIM_PATH)
    periods = pl.read_parquet(PERIOD_DIM_PATH)

    print(
        f"[PASS] Fact opened       : "
        f"{fact.height:,} rows"
    )
    print(
        f"[PASS] Institution dim   : "
        f"{institutions.height:,} rows"
    )
    print(
        f"[PASS] Period dim        : "
        f"{periods.height:,} rows"
    )

    # ------------------------------------------------------------------
    # 3. EXTRACT APPROVED CET1 ITEMS
    # ------------------------------------------------------------------

    header("3. EXTRACT APPROVED CET1 ITEMS")

    cet1_capital = (
        fact
        .filter(
            pl.col("item") == CET1_CAPITAL_ITEM
        )
        .select(
            [
                "lei_code",
                "period",
                pl.col("amount").alias("cet1_capital"),
            ]
        )
    )

    trea = (
        fact
        .filter(
            pl.col("item") == TOTAL_RISK_EXPOSURE_ITEM
        )
        .select(
            [
                "lei_code",
                "period",
                pl.col("amount").alias(
                    "total_risk_exposure_amount"
                ),
            ]
        )
    )

    reported_ratio = (
        fact
        .filter(
            pl.col("item") == CET1_RATIO_ITEM
        )
        .select(
            [
                "lei_code",
                "period",
                pl.col("amount").alias("cet1_ratio"),
            ]
        )
    )

    print(
        f"CET1 Capital rows             : "
        f"{cet1_capital.height:,}"
    )
    print(
        f"Total Risk Exposure rows      : "
        f"{trea.height:,}"
    )
    print(
        f"Reported CET1 Ratio rows      : "
        f"{reported_ratio.height:,}"
    )

    # ------------------------------------------------------------------
    # 4. SOURCE GRAIN VALIDATION
    # ------------------------------------------------------------------

    header("4. SOURCE GRAIN VALIDATION")

    assert_unique_grain(
        cet1_capital,
        "CET1 Capital source",
    )
    assert_unique_grain(
        trea,
        "Total Risk Exposure source",
    )
    assert_unique_grain(
        reported_ratio,
        "Reported CET1 Ratio source",
    )

    print(
        "[PASS] All source items are unique at "
        "lei_code x period."
    )

    # ------------------------------------------------------------------
    # 5. BUILD CET1 MART BASE
    # ------------------------------------------------------------------

    header("5. BUILD CET1 MART BASE")

    mart = (
        reported_ratio
        .join(
            cet1_capital,
            on=["lei_code", "period"],
            how="left",
        )
        .join(
            trea,
            on=["lei_code", "period"],
            how="left",
        )
        .with_columns(
            [
                pl.when(
                    pl.col("total_risk_exposure_amount") != 0
                )
                .then(
                    pl.col("cet1_capital")
                    / pl.col("total_risk_exposure_amount")
                )
                .otherwise(None)
                .alias("reconstructed_cet1_ratio"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("cet1_ratio") * 100
                ).alias("cet1_ratio_pct"),

                (
                    pl.col("cet1_ratio")
                    - pl.col("reconstructed_cet1_ratio")
                ).alias("reconciliation_difference"),

                (
                    (
                        pl.col("cet1_ratio")
                        - pl.col("reconstructed_cet1_ratio")
                    ).abs()
                ).alias("reconciliation_difference_abs"),

                (
                    pl.col("total_risk_exposure_amount") == 0
                ).alias("zero_trea_flag"),
            ]
        )
    )

    print(f"[PASS] Base mart rows: {mart.height:,}")

    # ------------------------------------------------------------------
    # 6. ADD INSTITUTION METADATA
    # ------------------------------------------------------------------

    header("6. ADD INSTITUTION METADATA")

    required_institution_fields = [
        "lei_code",
        "name",
        "country",
        "desc_country",
        "entity_type",
        "is_aggregate",
    ]

    missing_institution_fields = [
        field
        for field in required_institution_fields
        if field not in institutions.columns
    ]

    if missing_institution_fields:
        raise ValueError(
            "Institution dimension is missing required fields: "
            f"{missing_institution_fields}"
        )

    institution_lookup = (
        institutions
        .select(required_institution_fields)
        .rename(
            {
                "name": "institution_name",
                "country": "institution_country_code",
                "desc_country": "institution_country",
            }
        )
    )

    mart = mart.join(
        institution_lookup,
        on="lei_code",
        how="left",
    )

    missing_institution_metadata = mart.filter(
        pl.col("institution_name").is_null()
    ).height

    if missing_institution_metadata > 0:
        raise ValueError(
            "Institution metadata missing for "
            f"{missing_institution_metadata} mart rows."
        )

    print("[PASS] Institution metadata joined using:")
    print("       name         -> institution_name")
    print("       country      -> institution_country_code")
    print("       desc_country -> institution_country")

    # ------------------------------------------------------------------
    # 7. ADD PERIOD METADATA
    # ------------------------------------------------------------------

    header("7. ADD PERIOD METADATA")

    required_period_fields = [
        "period_code",
        "period_end_date",
        "year",
        "quarter",
        "month",
        "year_quarter",
        "is_year_end",
    ]

    missing_period_fields = [
        field
        for field in required_period_fields
        if field not in periods.columns
    ]

    if missing_period_fields:
        raise ValueError(
            "Period dimension is missing required fields: "
            f"{missing_period_fields}"
        )

    period_lookup = (
        periods
        .select(required_period_fields)
        .rename(
            {
                "period_code": "period",
            }
        )
        .with_columns(
            pl.col("period").cast(pl.Int64)
        )
    )

    print(
        "[PASS] Period lookup prepared using:"
    )
    print(
        "       period_code -> period"
    )
    print(
        "       period cast -> Int64"
    )

    mart = mart.join(
        period_lookup,
        on="period",
        how="left",
    )

    missing_period_metadata = mart.filter(
        pl.col("period_end_date").is_null()
    ).height

    if missing_period_metadata > 0:
        raise ValueError(
            "Period metadata missing for "
            f"{missing_period_metadata} mart rows."
        )

    print("[PASS] Period metadata joined.")

    # ------------------------------------------------------------------
    # 8. APPLY ANALYTICAL COLUMN ORDER
    # ------------------------------------------------------------------

    header("8. APPLY ANALYTICAL COLUMN ORDER")

    final_columns = [
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

    mart = (
        mart
        .select(final_columns)
        .sort(
            [
                "lei_code",
                "period",
            ]
        )
    )

    print(f"Rows    : {mart.height:,}")
    print(f"Columns : {mart.width}")

    # ------------------------------------------------------------------
    # 9. PRE-SAVE BUSINESS VALIDATION
    # ------------------------------------------------------------------

    header("9. PRE-SAVE BUSINESS VALIDATION")

    assert_unique_grain(
        mart,
        "CET1 analytical mart",
    )

    print(
        "[PASS] Mart is unique at grain: "
        "lei_code x period."
    )

    if mart.height != reported_ratio.height:
        raise ValueError(
            "Mart row count does not reconcile to the "
            "directly reported CET1 Ratio source."
        )

    print(
        f"[PASS] Row count reconciles: "
        f"{mart.height:,}"
    )

    missing_capital = mart[
        "cet1_capital"
    ].null_count()

    missing_trea = mart[
        "total_risk_exposure_amount"
    ].null_count()

    missing_ratio = mart[
        "cet1_ratio"
    ].null_count()

    print(f"CET1 Capital NULLs        : {missing_capital}")
    print(f"TREA NULLs                : {missing_trea}")
    print(f"CET1 Ratio NULLs          : {missing_ratio}")

    if missing_capital != 0:
        raise ValueError(
            "Unexpected missing CET1 Capital values."
        )

    if missing_trea != 0:
        raise ValueError(
            "Unexpected missing Total Risk Exposure values."
        )

    if missing_ratio != 0:
        raise ValueError(
            "Unexpected missing reported CET1 Ratio values."
        )

    zero_trea_rows = mart.filter(
        pl.col("zero_trea_flag")
    ).height

    if zero_trea_rows != 0:
        raise ValueError(
            "Zero Total Risk Exposure Amount found. "
            f"Rows: {zero_trea_rows}"
        )

    print("[PASS] No zero denominators.")

    max_difference = mart.select(
        pl.col(
            "reconciliation_difference_abs"
        ).max()
    ).item()

    print(
        "Maximum direct/reconstructed difference: "
        f"{max_difference:.16g}"
    )

    if max_difference > RECONCILIATION_TOLERANCE:
        raise ValueError(
            "Reported CET1 Ratio does not reconcile within "
            f"tolerance {RECONCILIATION_TOLERANCE}."
        )

    print(
        "[PASS] Direct reported CET1 Ratio reconciles to "
        "CET1 Capital / Total Risk Exposure Amount."
    )

    # ------------------------------------------------------------------
    # 10. POPULATION PROFILE
    # ------------------------------------------------------------------

    header("10. CET1 MART POPULATION PROFILE")

    population_profile = mart.select(
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
            (~pl.col("is_aggregate"))
            .sum()
            .alias("institution_rows"),
        ]
    )

    print(population_profile)

    print()
    print("Period profile:")

    period_profile = (
        mart
        .group_by("period")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("lei_code")
                .n_unique()
                .alias("entities"),
                pl.col("cet1_ratio")
                .median()
                .alias("median_cet1_ratio"),
                pl.col("cet1_ratio")
                .min()
                .alias("minimum_cet1_ratio"),
                pl.col("cet1_ratio")
                .max()
                .alias("maximum_cet1_ratio"),
            ]
        )
        .sort("period")
    )

    print(period_profile)

    print()
    print("High-ratio observations (>100%):")

    high_ratio_profile = (
        mart
        .filter(
            pl.col("cet1_ratio") > 1.0
        )
        .select(
            [
                "lei_code",
                "institution_name",
                "institution_country",
                "period",
                "cet1_capital",
                "total_risk_exposure_amount",
                "cet1_ratio",
                "cet1_ratio_pct",
                "is_aggregate",
            ]
        )
        .sort(
            "cet1_ratio",
            descending=True,
        )
    )

    print(high_ratio_profile)

    # ------------------------------------------------------------------
    # 11. SAVE CET1 RATIO MART
    # ------------------------------------------------------------------

    header("11. SAVE CET1 RATIO MART")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mart.write_parquet(OUTPUT_PATH)

    print(f"[PASS] Saved: {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 12. POST-SAVE VALIDATION
    # ------------------------------------------------------------------

    header("12. POST-SAVE VALIDATION")

    saved = pl.read_parquet(OUTPUT_PATH)

    if saved.shape != mart.shape:
        raise ValueError(
            "Saved mart shape does not match in-memory mart."
        )

    print(
        f"[PASS] Saved mart reopened: "
        f"{saved.height:,} rows x "
        f"{saved.width} columns"
    )

    assert_unique_grain(
        saved,
        "Saved CET1 mart",
    )

    print(
        "[PASS] Saved mart remains unique at "
        "lei_code x period."
    )

    if saved.columns != final_columns:
        raise ValueError(
            "Saved mart column order does not match "
            "the approved analytical schema."
        )

    print("[PASS] Saved column order reconciles.")

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    header("CET1 RATIO MART BUILD COMPLETE")

    print(f"Output : {OUTPUT_PATH}")
    print(f"Rows   : {saved.height:,}")
    print(f"Columns: {saved.width}")

    print()
    print("Primary KPI:")
    print(
        "  Item 2520140 - COMMON EQUITY TIER 1 CAPITAL "
        "RATIO (transitional period)"
    )

    print()
    print("Validation control:")
    print(
        "  Item 2520102 CET1 Capital"
    )
    print(
        "  divided by"
    )
    print(
        "  Item 2520138 Total Risk Exposure Amount"
    )

    print()
    print(
        "No regulatory thresholds, capital classifications, "
        "or risk flags were introduced."
    )

    print()
    print(
        "High CET1 Ratio observations were retained unchanged."
    )


if __name__ == "__main__":
    main()
