"""
Build the analytical EBA institution dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-4 — Analytical Institution Dimension

Inputs:
    data/staging/reference/dim_institution.parquet
    data/staging/facts/stg_credit_risk.parquet
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/dimensions/dim_institution.parquet

Purpose:
    Build a one-row-per-reporting-entity analytical institution dimension.

Important source finding:
    The staged EBA institution metadata already contains the aggregate:

        xxxxxxxxxxxxxxxxxxxx = All other banks

    Therefore no synthetic aggregate row is added.

    The analytical layer standardises LEI casing to uppercase so that the
    aggregate identifier aligns with the staged fact representation:

        XXXXXXXXXXXXXXXXXXXX

This script DOES NOT:
    - modify raw data;
    - modify staged reference data;
    - modify staged fact data;
    - invent institution metadata;
    - calculate banking KPIs.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_INSTITUTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_institution.parquet"
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
    / "dim_institution.parquet"
)


# =============================================================================
# Known EBA aggregate
# =============================================================================

ALL_OTHER_BANKS_LEI = "XXXXXXXXXXXXXXXXXXXX"
ALL_OTHER_BANKS_NAME = "All other banks"


# =============================================================================
# Required source structure
# =============================================================================

REQUIRED_SOURCE_COLUMNS = [
    "lei_code",
    "name",
    "country",
]

OPTIONAL_SOURCE_COLUMNS = [
    "desc_country",
    "finrep",
    "fin_year_end",
    "footnotes",
]


# =============================================================================
# Normalisation
# =============================================================================

def normalise_lei(
    value: object,
) -> str:
    """
    Normalise LEI/reporting identifiers for analytical comparison.

    The staged source itself is not modified.
    """

    return (
        str(value)
        .strip()
        .upper()
    )


# =============================================================================
# Loading
# =============================================================================

def load_staged_institutions() -> pl.DataFrame:
    """
    Load the validated staged institution reference table.
    """

    if not STAGED_INSTITUTION_FILE.is_file():

        raise FileNotFoundError(
            "Staged institution table not found: "
            f"{STAGED_INSTITUTION_FILE}"
        )

    df = pl.read_parquet(
        STAGED_INSTITUTION_FILE
    )

    missing_required = [
        column
        for column in REQUIRED_SOURCE_COLUMNS
        if column not in df.columns
    ]

    if missing_required:

        raise KeyError(
            "Required institution column(s) missing: "
            + ", ".join(missing_required)
        )

    return df


def load_fact_lei_codes(
    file_path: Path,
    dataset_name: str,
) -> set[str]:
    """
    Return distinct normalised reporting identifiers
    from one staged fact table.
    """

    if not file_path.is_file():

        raise FileNotFoundError(
            f"{dataset_name} not found: {file_path}"
        )

    lazy_df = pl.scan_parquet(
        file_path
    )

    schema = lazy_df.collect_schema()

    if "lei_code" not in schema.names():

        raise KeyError(
            f"{dataset_name} does not contain 'lei_code'."
        )

    values = (
        lazy_df
        .select(
            pl.col("lei_code")
            .drop_nulls()
            .unique()
        )
        .collect()["lei_code"]
        .to_list()
    )

    return {
        normalise_lei(value)
        for value in values
    }


# =============================================================================
# Staged source validation
# =============================================================================

def validate_staged_institutions(
    df: pl.DataFrame,
) -> None:
    """
    Validate the institution staging population before analytical modelling.
    """

    print()
    print("=" * 100)
    print("STAGED INSTITUTION SOURCE VALIDATION")
    print("=" * 100)

    print(
        f"[INFO] Staged institution rows : "
        f"{df.height:,}"
    )

    # -------------------------------------------------------------------------
    # LEI completeness
    # -------------------------------------------------------------------------

    null_lei = df[
        "lei_code"
    ].null_count()

    if null_lei != 0:

        raise ValueError(
            f"Staged institution table contains "
            f"{null_lei:,} NULL lei_code value(s)."
        )

    print(
        "[PASS] lei_code contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Raw source uniqueness
    # -------------------------------------------------------------------------

    if df[
        "lei_code"
    ].n_unique() != df.height:

        raise ValueError(
            "Staged institution lei_code is not unique."
        )

    print(
        f"[PASS] Source lei_code is unique: "
        f"{df.height:,} distinct code(s)."
    )

    # -------------------------------------------------------------------------
    # Normalised uniqueness
    #
    # This protects against identifiers differing only by case.
    # -------------------------------------------------------------------------

    normalised = (
        df
        .with_columns(
            pl.col("lei_code")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_uppercase()
            .alias("_normalised_lei")
        )
    )

    normalised_distinct = normalised[
        "_normalised_lei"
    ].n_unique()

    if normalised_distinct != df.height:

        duplicates = (
            normalised
            .group_by(
                "_normalised_lei"
            )
            .agg(
                pl.len().alias(
                    "row_count"
                )
            )
            .filter(
                pl.col("row_count") > 1
            )
        )

        print(
            duplicates
        )

        raise ValueError(
            "Institution identifiers are not unique after "
            "case normalisation."
        )

    print(
        "[PASS] lei_code remains unique after case normalisation."
    )

    # -------------------------------------------------------------------------
    # Name completeness
    # -------------------------------------------------------------------------

    null_names = df[
        "name"
    ].null_count()

    if null_names != 0:

        raise ValueError(
            f"Staged institution table contains "
            f"{null_names:,} NULL name value(s)."
        )

    print(
        "[PASS] name contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Aggregate detection — case insensitive
    # -------------------------------------------------------------------------

    aggregate_rows = (
        normalised
        .filter(
            pl.col("_normalised_lei")
            == ALL_OTHER_BANKS_LEI
        )
    )

    if aggregate_rows.height != 1:

        raise ValueError(
            "Expected exactly one EBA 'All other banks' row "
            "in staged institution metadata after LEI normalisation. "
            f"Found {aggregate_rows.height:,}."
        )

    aggregate_name = aggregate_rows[
        "name"
    ][0]

    if (
        str(aggregate_name)
        .strip()
        .lower()
        != ALL_OTHER_BANKS_NAME.lower()
    ):

        raise ValueError(
            "The recognised aggregate LEI exists but its name "
            f"is unexpected: {aggregate_name!r}"
        )

    print(
        "[PASS] Existing EBA 'All other banks' aggregate "
        "identified in staged metadata."
    )

    print(
        "[PASS] No synthetic aggregate row is required."
    )


# =============================================================================
# Analytical construction
# =============================================================================

def build_analytical_dimension(
    staged: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build analytical institution dimension.

    All 120 staged EBA reporting entities are preserved.

    LEI codes are standardised to uppercase in the analytical layer.

    The existing EBA 'All other banks' row is classified as aggregate.
    """

    source_columns = [
        column
        for column in (
            REQUIRED_SOURCE_COLUMNS
            + OPTIONAL_SOURCE_COLUMNS
        )
        if column in staged.columns
    ]

    analytical = (
        staged
        .select(
            source_columns
        )
        .with_columns(
            pl.col("lei_code")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_uppercase()
            .alias("lei_code")
        )
        .with_columns(
            [
                pl.when(
                    pl.col("lei_code")
                    == ALL_OTHER_BANKS_LEI
                )
                .then(
                    pl.lit("aggregate")
                )
                .otherwise(
                    pl.lit("institution")
                )
                .alias(
                    "entity_type"
                ),

                (
                    pl.col("lei_code")
                    == ALL_OTHER_BANKS_LEI
                )
                .alias(
                    "is_aggregate"
                ),

                pl.lit(
                    "eba_institution_metadata"
                )
                .alias(
                    "record_source"
                ),
            ]
        )
        .sort(
            [
                "is_aggregate",
                "name",
            ]
        )
    )

    return analytical


# =============================================================================
# Analytical validation
# =============================================================================

def validate_analytical_dimension(
    df: pl.DataFrame,
    staged: pl.DataFrame,
    credit_lei_codes: set[str],
    other_lei_codes: set[str],
) -> None:
    """
    Validate analytical institution dimension.
    """

    print()
    print("=" * 100)
    print("ANALYTICAL INSTITUTION DIMENSION VALIDATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Row reconciliation
    #
    # No new entity should be created because the aggregate already exists
    # in the staged EBA metadata.
    # -------------------------------------------------------------------------

    if df.height != staged.height:

        raise ValueError(
            f"Analytical row count must equal staged row count. "
            f"Staged={staged.height:,}, "
            f"analytical={df.height:,}"
        )

    print(
        f"[PASS] Row reconciliation: "
        f"{df.height:,} reporting entities preserved."
    )

    # -------------------------------------------------------------------------
    # LEI uniqueness
    # -------------------------------------------------------------------------

    distinct_lei = df[
        "lei_code"
    ].n_unique()

    if distinct_lei != df.height:

        raise ValueError(
            "Analytical lei_code is not unique."
        )

    print(
        f"[PASS] lei_code is unique: "
        f"{distinct_lei:,} distinct code(s)."
    )

    # -------------------------------------------------------------------------
    # Completeness
    # -------------------------------------------------------------------------

    if df[
        "lei_code"
    ].null_count() != 0:

        raise ValueError(
            "Analytical dimension contains NULL lei_code."
        )

    print(
        "[PASS] lei_code contains 0 NULL values."
    )

    if df[
        "name"
    ].null_count() != 0:

        raise ValueError(
            "Analytical dimension contains NULL name."
        )

    print(
        "[PASS] name contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Aggregate classification
    # -------------------------------------------------------------------------

    aggregate = df.filter(
        pl.col("lei_code")
        == ALL_OTHER_BANKS_LEI
    )

    if aggregate.height != 1:

        raise ValueError(
            "Expected exactly one analytical All-other-banks entity."
        )

    if not bool(
        aggregate[
            "is_aggregate"
        ][0]
    ):

        raise ValueError(
            "All-other-banks row is not classified as aggregate."
        )

    if aggregate[
        "entity_type"
    ][0] != "aggregate":

        raise ValueError(
            "All-other-banks entity_type is incorrect."
        )

    print(
        "[PASS] Exactly one EBA 'All other banks' "
        "aggregate member exists."
    )

    # -------------------------------------------------------------------------
    # Entity counts
    # -------------------------------------------------------------------------

    regular_count = (
        df
        .filter(
            ~pl.col("is_aggregate")
        )
        .height
    )

    aggregate_count = (
        df
        .filter(
            pl.col("is_aggregate")
        )
        .height
    )

    if regular_count != 119:

        raise ValueError(
            f"Expected 119 regular institutions, "
            f"found {regular_count:,}."
        )

    if aggregate_count != 1:

        raise ValueError(
            f"Expected 1 aggregate entity, "
            f"found {aggregate_count:,}."
        )

    print(
        f"[PASS] Regular institutions : "
        f"{regular_count:,}"
    )

    print(
        f"[PASS] Aggregate entities    : "
        f"{aggregate_count:,}"
    )

    # -------------------------------------------------------------------------
    # Fact coverage
    # -------------------------------------------------------------------------

    dimension_codes = {
        normalise_lei(value)
        for value in df[
            "lei_code"
        ].to_list()
    }

    missing_credit = sorted(
        credit_lei_codes
        - dimension_codes
    )

    if missing_credit:

        raise ValueError(
            "Credit-risk institution coverage failed. "
            f"Missing: {missing_credit}"
        )

    print(
        f"[PASS] All {len(credit_lei_codes):,} "
        "credit-risk reporting codes are represented."
    )

    missing_other = sorted(
        other_lei_codes
        - dimension_codes
    )

    if missing_other:

        raise ValueError(
            "Other-template institution coverage failed. "
            f"Missing: {missing_other}"
        )

    print(
        f"[PASS] All {len(other_lei_codes):,} "
        "other-template reporting codes are represented."
    )

    fact_union = (
        credit_lei_codes
        | other_lei_codes
    )

    if dimension_codes != fact_union:

        dimension_not_used = sorted(
            dimension_codes
            - fact_union
        )

        fact_not_in_dimension = sorted(
            fact_union
            - dimension_codes
        )

        raise ValueError(
            "Analytical institution population does not reconcile "
            "exactly to the current fact reporting population.\n"
            f"Dimension-only codes: {dimension_not_used}\n"
            f"Fact-only codes: {fact_not_in_dimension}"
        )

    print(
        "[PASS] Analytical institution population exactly "
        "matches the combined fact reporting population."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """
    Save corrected analytical institution dimension.

    This overwrites the previous unapproved derived analytical output.
    Raw and staged source files remain unchanged.
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
        "Saved analytical institution dimension:"
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
    Re-read persisted analytical institution dimension.
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
        "lei_code"
    ].n_unique() != saved.height:

        raise ValueError(
            "Persisted analytical institution dimension "
            "does not have unique lei_code."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows           : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct LEI codes   : "
        f"{saved['lei_code'].n_unique():,}"
    )

    classification = (
        saved
        .group_by(
            [
                "entity_type",
                "record_source",
            ]
        )
        .agg(
            pl.len().alias(
                "entity_count"
            )
        )
        .sort(
            "entity_type"
        )
    )

    print()
    print(
        "Entity classification:"
    )

    print(
        classification
    )

    print()
    print(
        "Aggregate member:"
    )

    print(
        saved
        .filter(
            pl.col("is_aggregate")
        )
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build corrected analytical institution dimension.
    """

    print("=" * 100)
    print("BUILDING ANALYTICAL INSTITUTION DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged institution : {STAGED_INSTITUTION_FILE}"
    )

    print(
        f"Credit fact        : {CREDIT_FACT_FILE}"
    )

    print(
        f"Other fact         : {OTHER_FACT_FILE}"
    )

    print(
        f"Output             : {OUTPUT_FILE}"
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    staged = (
        load_staged_institutions()
    )

    credit_lei_codes = (
        load_fact_lei_codes(
            CREDIT_FACT_FILE,
            "stg_credit_risk",
        )
    )

    other_lei_codes = (
        load_fact_lei_codes(
            OTHER_FACT_FILE,
            "stg_other_templates",
        )
    )

    # -------------------------------------------------------------------------
    # Validate source
    # -------------------------------------------------------------------------

    validate_staged_institutions(
        staged
    )

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    analytical = (
        build_analytical_dimension(
            staged
        )
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_analytical_dimension(
        df=analytical,
        staged=staged,
        credit_lei_codes=credit_lei_codes,
        other_lei_codes=other_lei_codes,
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
        expected_rows=analytical.height
    )

    print()
    print("=" * 100)
    print(
        "ANALYTICAL INSTITUTION DIMENSION BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "No raw, staging, or fact data was modified."
    )


if __name__ == "__main__":
    main()
