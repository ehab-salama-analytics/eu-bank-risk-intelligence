"""
Build the analytical EBA Exposure dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-5B — Analytical Exposure Dimension

Inputs:
    data/staging/reference/dim_exposure.parquet
    data/staging/facts/stg_credit_risk.parquet
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/dimensions/dim_exposure.parquet

Purpose:
    Build a reusable analytical Exposure dimension from the validated
    staged EBA lookup.

The staged lookup currently contains:
    - 52 unique Exposure codes;
    - no NULL Exposure keys;
    - full coverage of Exposure codes used by both staged fact datasets.

The analytical dimension preserves every staged lookup member and adds
fact-usage metadata.

This script DOES NOT:
    - modify staged reference data;
    - modify staged fact data;
    - remove unused lookup members;
    - calculate banking KPIs;
    - join the dimension to fact tables;
    - create warehouse surrogate keys.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_EXPOSURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_exposure.parquet"
)

CREDIT_FACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "facts"
    / "stg_credit_risk.parquet"
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
    / "dim_exposure.parquet"
)


# =============================================================================
# Expected staged structure
# =============================================================================

EXPECTED_COLUMNS = [
    "exposure",
    "label",
]


# =============================================================================
# Source loading
# =============================================================================

def load_staged_exposure() -> pl.DataFrame:
    """
    Load the validated staged Exposure lookup.
    """

    if not STAGED_EXPOSURE_FILE.is_file():

        raise FileNotFoundError(
            f"Staged Exposure lookup not found: "
            f"{STAGED_EXPOSURE_FILE}"
        )

    df = pl.read_parquet(
        STAGED_EXPOSURE_FILE
    )

    if df.columns != EXPECTED_COLUMNS:

        raise ValueError(
            "Unexpected staged Exposure schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {df.columns}"
        )

    return df


def load_fact_exposure_codes(
    file_path: Path,
    dataset_name: str,
) -> set[int]:
    """
    Return distinct non-null Exposure codes used by one staged fact table.
    """

    if not file_path.is_file():

        raise FileNotFoundError(
            f"{dataset_name} not found: {file_path}"
        )

    lazy_df = pl.scan_parquet(
        file_path
    )

    schema = lazy_df.collect_schema()

    if "exposure" not in schema.names():

        raise KeyError(
            f"{dataset_name} does not contain 'exposure'."
        )

    values = (
        lazy_df
        .select(
            pl.col("exposure")
            .drop_nulls()
            .unique()
        )
        .collect()["exposure"]
        .to_list()
    )

    return {
        int(value)
        for value in values
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_staged_exposure(
    df: pl.DataFrame,
) -> None:
    """
    Reconfirm the critical properties of the staged lookup.
    """

    print()
    print("=" * 100)
    print("STAGED EXPOSURE SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged rows          : "
        f"{df.height:,}"
    )

    # -------------------------------------------------------------------------
    # Key completeness
    # -------------------------------------------------------------------------

    null_keys = df[
        "exposure"
    ].null_count()

    if null_keys != 0:

        raise ValueError(
            f"Staged Exposure lookup contains "
            f"{null_keys:,} NULL key value(s)."
        )

    print(
        "[PASS] exposure contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Key uniqueness
    # -------------------------------------------------------------------------

    distinct_keys = df[
        "exposure"
    ].n_unique()

    if distinct_keys != df.height:

        raise ValueError(
            "Staged Exposure key is not unique."
        )

    print(
        f"[PASS] exposure is unique: "
        f"{distinct_keys:,} distinct code(s)."
    )

    # -------------------------------------------------------------------------
    # Label completeness
    # -------------------------------------------------------------------------

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:

        raise ValueError(
            f"Staged Exposure lookup contains "
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
    credit_codes: set[int],
    other_codes: set[int],
) -> pl.DataFrame:
    """
    Preserve all staged Exposure rows and append fact-usage indicators.
    """

    analytical = (
        staged
        .with_columns(
            [
                pl.col("exposure")
                .is_in(
                    list(
                        credit_codes
                    )
                )
                .alias(
                    "used_in_credit_risk"
                ),

                pl.col("exposure")
                .is_in(
                    list(
                        other_codes
                    )
                )
                .alias(
                    "used_in_other_templates"
                ),
            ]
        )
        .with_columns(
            pl.when(
                pl.col(
                    "used_in_credit_risk"
                )
                &
                pl.col(
                    "used_in_other_templates"
                )
            )
            .then(
                pl.lit(
                    "both"
                )
            )
            .when(
                pl.col(
                    "used_in_credit_risk"
                )
            )
            .then(
                pl.lit(
                    "credit_risk"
                )
            )
            .when(
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
            "exposure"
        )
    )

    return analytical


# =============================================================================
# Analytical validation
# =============================================================================

def validate_analytical_dimension(
    df: pl.DataFrame,
    staged: pl.DataFrame,
    credit_codes: set[int],
    other_codes: set[int],
) -> None:
    """
    Validate the analytical Exposure dimension before saving.
    """

    print()
    print("=" * 100)
    print("ANALYTICAL EXPOSURE DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    # -------------------------------------------------------------------------

    if df.height != staged.height:

        raise ValueError(
            f"Analytical row count does not match staged lookup. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} lookup members preserved."
    )

    # -------------------------------------------------------------------------
    # Key uniqueness
    # -------------------------------------------------------------------------

    if df[
        "exposure"
    ].n_unique() != df.height:

        raise ValueError(
            "Analytical Exposure key is not unique."
        )

    print(
        "[PASS] exposure remains unique."
    )

    # -------------------------------------------------------------------------
    # Critical completeness
    # -------------------------------------------------------------------------

    if df[
        "exposure"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Exposure contains NULL keys."
        )

    print(
        "[PASS] exposure contains 0 NULL values."
    )

    if df[
        "label"
    ].null_count() != 0:

        raise ValueError(
            "Analytical Exposure contains NULL labels."
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
            "exposure"
        ].to_list()
    }

    missing_credit = sorted(
        credit_codes
        - dimension_codes
    )

    if missing_credit:

        raise ValueError(
            f"Credit-risk Exposure coverage failed. "
            f"Missing: {missing_credit}"
        )

    print(
        f"[PASS] All {len(credit_codes):,} "
        "credit-risk Exposure codes are represented."
    )

    missing_other = sorted(
        other_codes
        - dimension_codes
    )

    if missing_other:

        raise ValueError(
            f"Other-template Exposure coverage failed. "
            f"Missing: {missing_other}"
        )

    print(
        f"[PASS] All {len(other_codes):,} "
        "other-template Exposure codes are represented."
    )

    # -------------------------------------------------------------------------
    # Usage-flag reconciliation
    # -------------------------------------------------------------------------

    flagged_credit = set(
        df
        .filter(
            pl.col(
                "used_in_credit_risk"
            )
        )[
            "exposure"
        ]
        .to_list()
    )

    if flagged_credit != credit_codes:

        raise ValueError(
            "Credit-risk usage flags do not reconcile "
            "to fact Exposure codes."
        )

    print(
        "[PASS] Credit-risk usage flags reconcile."
    )

    flagged_other = set(
        df
        .filter(
            pl.col(
                "used_in_other_templates"
            )
        )[
            "exposure"
        ]
        .to_list()
    )

    if flagged_other != other_codes:

        raise ValueError(
            "Other-template usage flags do not reconcile "
            "to fact Exposure codes."
        )

    print(
        "[PASS] Other-template usage flags reconcile."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save analytical Exposure dimension.
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
        "Saved analytical Exposure dimension:"
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
    Re-read the saved Parquet file.
    """

    if not OUTPUT_FILE.is_file():

        raise FileNotFoundError(
            f"Expected output not found: {OUTPUT_FILE}"
        )

    saved = pl.read_parquet(
        OUTPUT_FILE
    )

    if saved.height != expected_rows:

        raise ValueError(
            f"Saved row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "exposure"
    ].n_unique() != saved.height:

        raise ValueError(
            "Saved analytical Exposure key is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows              : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Exposure codes : "
        f"{saved['exposure'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "exposure_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Exposure usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Exposure sample:"
    )

    print(
        saved
        .head(
            15
        )
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build analytical Exposure dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL EXPOSURE DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Exposure : {STAGED_EXPOSURE_FILE}"
    )

    print(
        f"Credit fact     : {CREDIT_FACT_FILE}"
    )

    print(
        f"Other fact      : {OTHER_FACT_FILE}"
    )

    print(
        f"Output          : {OUTPUT_FILE}"
    )

    # -------------------------------------------------------------------------
    # Load sources
    # -------------------------------------------------------------------------

    staged = (
        load_staged_exposure()
    )

    credit_codes = (
        load_fact_exposure_codes(
            CREDIT_FACT_FILE,
            "stg_credit_risk",
        )
    )

    other_codes = (
        load_fact_exposure_codes(
            OTHER_FACT_FILE,
            "stg_other_templates",
        )
    )

    # -------------------------------------------------------------------------
    # Validate source
    # -------------------------------------------------------------------------

    validate_staged_exposure(
        staged
    )

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    analytical = (
        build_analytical_dimension(
            staged=staged,
            credit_codes=credit_codes,
            other_codes=other_codes,
        )
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_analytical_dimension(
        df=analytical,
        staged=staged,
        credit_codes=credit_codes,
        other_codes=other_codes,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_dimension(
        analytical
    )

    # -------------------------------------------------------------------------
    # Post-save validation
    # -------------------------------------------------------------------------

    validate_saved_dimension(
        expected_rows=staged.height
    )

    print()
    print("=" * 100)
    print(
        "ANALYTICAL EXPOSURE DIMENSION BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
