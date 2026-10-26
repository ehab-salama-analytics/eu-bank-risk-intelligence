"""
Build the analytical EBA Financial Year-End dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5K — Analytical Financial Year-End Dimension

Inputs:
    data/staging/reference/dim_fin_year_end.parquet
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/dimensions/dim_fin_year_end.parquet

Purpose:
    Build a reusable analytical Financial Year-End dimension from the
    validated staged EBA lookup.

Validated source characteristics:
    - 4 staged Financial Year-End rows;
    - fin_end_year is unique;
    - no NULL fin_end_year keys;
    - 2 Financial Year-End codes are currently used by the
      other-template fact;
    - no fact Financial Year-End codes are missing from the lookup;
    - 2 valid lookup members are currently unused.

The script preserves all staged lookup members and adds current
fact-usage metadata.

This script DOES NOT:
    - modify staging data;
    - modify fact data;
    - infer financial year-end values;
    - remove unused lookup members;
    - calculate banking KPIs;
    - join the dimension to fact tables.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_FIN_YEAR_END_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_fin_year_end.parquet"
)

OTHER_FACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
    / "stg_other_templates.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dim_fin_year_end.parquet"
)


# =============================================================================
# Expected staged structure
# =============================================================================

EXPECTED_COLUMNS = [
    "fin_end_year",
    "label",
]


# =============================================================================
# Loading
# =============================================================================

def load_staged_fin_year_end() -> pl.DataFrame:
    """
    Load the validated staged Financial Year-End lookup.
    """

    if not STAGED_FIN_YEAR_END_FILE.is_file():
        raise FileNotFoundError(
            f"Staged Financial Year-End lookup not found: "
            f"{STAGED_FIN_YEAR_END_FILE}"
        )

    df = pl.read_parquet(
        STAGED_FIN_YEAR_END_FILE
    )

    if df.columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected staged Financial Year-End schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_other_fin_year_end_codes() -> set[int]:
    """
    Return distinct non-null fin_end_year codes used by
    stg_other_templates.
    """

    if not OTHER_FACT_FILE.is_file():
        raise FileNotFoundError(
            f"Other-template staging file not found: "
            f"{OTHER_FACT_FILE}"
        )

    lazy_df = pl.scan_parquet(
        OTHER_FACT_FILE
    )

    schema = lazy_df.collect_schema()

    if "fin_end_year" not in schema.names():
        raise KeyError(
            "stg_other_templates does not contain 'fin_end_year'."
        )

    values = (
        lazy_df
        .select(
            pl.col("fin_end_year")
            .drop_nulls()
            .unique()
        )
        .collect()["fin_end_year"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_fin_year_end(
    df: pl.DataFrame,
) -> None:
    """
    Validate staged Financial Year-End lookup.
    """

    print()
    print("=" * 100)
    print("STAGED FINANCIAL YEAR-END SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows : {df.height:,}"
    )

    if df.height != 4:
        raise ValueError(
            f"Expected 4 staged Financial Year-End rows, "
            f"found {df.height:,}."
        )

    print(
        "[PASS] Staged row count matches validated profile: 4."
    )

    null_keys = df[
        "fin_end_year"
    ].null_count()

    if null_keys != 0:
        raise ValueError(
            f"Financial Year-End lookup contains "
            f"{null_keys:,} NULL key value(s)."
        )

    print(
        "[PASS] fin_end_year contains 0 NULL values."
    )

    distinct_keys = df[
        "fin_end_year"
    ].n_unique()

    if distinct_keys != df.height:
        raise ValueError(
            "Financial Year-End lookup key is not unique."
        )

    print(
        f"[PASS] fin_end_year is unique: "
        f"{distinct_keys:,} distinct code(s)."
    )

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:
        raise ValueError(
            f"Financial Year-End lookup contains "
            f"{null_labels:,} NULL label(s)."
        )

    print(
        "[PASS] label contains 0 NULL values."
    )


# =============================================================================
# Build analytical dimension
# =============================================================================

def build_analytical_dimension(
    staged: pl.DataFrame,
    other_codes: set[int],
) -> pl.DataFrame:
    """
    Preserve all Financial Year-End members and add current fact-usage
    metadata.
    """

    return (
        staged
        .with_columns(
            pl.col("fin_end_year")
            .is_in(
                list(other_codes)
            )
            .alias(
                "used_in_other_templates"
            )
        )
        .with_columns(
            pl.when(
                pl.col(
                    "used_in_other_templates"
                )
            )
            .then(
                pl.lit(
                    "other_templates"
                )
            )
            .otherwise(
                pl.lit(
                    "not_used_in_current_facts"
                )
            )
            .alias(
                "fact_usage"
            )
        )
        .sort(
            "fin_end_year"
        )
    )


# =============================================================================
# Analytical validation
# =============================================================================

def validate_analytical_dimension(
    df: pl.DataFrame,
    staged: pl.DataFrame,
    other_codes: set[int],
) -> None:
    """
    Validate analytical Financial Year-End dimension.
    """

    print()
    print("=" * 100)
    print("ANALYTICAL FINANCIAL YEAR-END DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    # -------------------------------------------------------------------------

    if df.height != staged.height:
        raise ValueError(
            f"Analytical Financial Year-End row count mismatch. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} Financial Year-End lookup members preserved."
    )

    # -------------------------------------------------------------------------
    # Key integrity
    # -------------------------------------------------------------------------

    if df[
        "fin_end_year"
    ].n_unique() != df.height:
        raise ValueError(
            "Analytical Financial Year-End key is not unique."
        )

    print(
        "[PASS] fin_end_year remains unique."
    )

    if df[
        "fin_end_year"
    ].null_count() != 0:
        raise ValueError(
            "Analytical Financial Year-End contains NULL keys."
        )

    print(
        "[PASS] fin_end_year contains 0 NULL values."
    )

    if df[
        "label"
    ].null_count() != 0:
        raise ValueError(
            "Analytical Financial Year-End contains NULL labels."
        )

    print(
        "[PASS] label contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Fact coverage
    # -------------------------------------------------------------------------

    dimension_codes = {
        int(value)
        for value in df[
            "fin_end_year"
        ].to_list()
    }

    missing_other = sorted(
        other_codes
        - dimension_codes
    )

    if missing_other:
        raise ValueError(
            "Other-template Financial Year-End coverage failed. "
            f"Missing: {missing_other}"
        )

    print(
        f"[PASS] All {len(other_codes):,} other-template "
        "Financial Year-End codes are represented."
    )

    # -------------------------------------------------------------------------
    # Usage-flag reconciliation
    # -------------------------------------------------------------------------

    flagged_codes = {
        int(value)
        for value in (
            df
            .filter(
                pl.col(
                    "used_in_other_templates"
                )
            )[
                "fin_end_year"
            ]
            .to_list()
        )
    }

    if flagged_codes != other_codes:
        raise ValueError(
            "Financial Year-End usage flags do not reconcile "
            "to other-template fact codes."
        )

    print(
        "[PASS] Other-template usage flags reconcile."
    )

    # -------------------------------------------------------------------------
    # Expected usage counts
    # -------------------------------------------------------------------------

    used_count = (
        df
        .filter(
            pl.col(
                "used_in_other_templates"
            )
        )
        .height
    )

    unused_count = (
        df.height
        - used_count
    )

    if used_count != 2:
        raise ValueError(
            f"Expected 2 currently used Financial Year-End codes, "
            f"found {used_count:,}."
        )

    if unused_count != 2:
        raise ValueError(
            f"Expected 2 currently unused Financial Year-End codes, "
            f"found {unused_count:,}."
        )

    print(
        "[PASS] Current usage population reconciles: "
        "2 used / 2 unused."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save analytical Financial Year-End dimension.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
    )

    print()
    print(
        "Saved analytical Financial Year-End dimension:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


# =============================================================================
# Post-save validation
# =============================================================================

def validate_saved_dimension(
    expected_rows: int,
) -> None:
    """
    Re-read persisted analytical Financial Year-End dimension.
    """

    if not OUTPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Expected output not found: "
            f"{OUTPUT_FILE}"
        )

    saved = pl.read_parquet(
        OUTPUT_FILE
    )

    if saved.height != expected_rows:
        raise ValueError(
            f"Saved Financial Year-End row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "fin_end_year"
    ].n_unique() != saved.height:
        raise ValueError(
            "Saved analytical Financial Year-End key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows                      : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Financial Year-End codes: "
        f"{saved['fin_end_year'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "fin_year_end_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Financial Year-End usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Financial Year-End dimension:"
    )

    print(
        saved
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build analytical Financial Year-End dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL FINANCIAL YEAR-END DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Financial Year-End : "
        f"{STAGED_FIN_YEAR_END_FILE}"
    )

    print(
        f"Other fact                : "
        f"{OTHER_FACT_FILE}"
    )

    print(
        f"Output                    : "
        f"{OUTPUT_FILE}"
    )

    staged = (
        load_staged_fin_year_end()
    )

    other_codes = (
        load_other_fin_year_end_codes()
    )

    validate_staged_fin_year_end(
        staged
    )

    analytical = (
        build_analytical_dimension(
            staged=staged,
            other_codes=other_codes,
        )
    )

    validate_analytical_dimension(
        df=analytical,
        staged=staged,
        other_codes=other_codes,
    )

    save_dimension(
        analytical
    )

    validate_saved_dimension(
        expected_rows=staged.height
    )

    print()
    print("=" * 100)
    print(
        "ANALYTICAL FINANCIAL YEAR-END DIMENSION BUILD "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
