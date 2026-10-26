"""
Build the analytical EBA Item dimension.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-3 — Analytical Item Dimension

Inputs:
    data/staging/reference/dim_item.parquet
    data/staging/facts/stg_credit_risk.parquet
    data/staging/facts/stg_other_templates.parquet

Output:
    data/analytical/dimensions/dim_item_analytic.parquet

Purpose:
    Convert the staged EBA Single Data Dictionary from source metadata
    grain (Item + Template) into a validated one-row-per-Item analytical
    dimension.

Important:
    The staged SDD is NOT modified.

    Canonicalisation is permitted only when repeated Item rows have
    consistent analytical definitions.

This script DOES NOT:
    - modify staged reference data;
    - modify staged fact data;
    - delete source SDD records;
    - join the dimension to fact tables;
    - calculate banking KPIs.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGED_ITEM_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
    / "dim_item.parquet"
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
    / "dim_item_analytic.parquet"
)


# =============================================================================
# Canonical analytical fields
# =============================================================================

CANONICAL_FIELDS = [
    "item",
    "label",
    "category",
    "csv",
    "collection",
]

CONSISTENCY_FIELDS = [
    "label",
    "category",
    "csv",
    "collection",
]


# =============================================================================
# Source loading
# =============================================================================

def load_staged_item() -> pl.DataFrame:
    """Load the staged EBA SDD reference table."""

    if not STAGED_ITEM_FILE.is_file():

        raise FileNotFoundError(
            f"Staged Item table not found: {STAGED_ITEM_FILE}"
        )

    df = pl.read_parquet(
        STAGED_ITEM_FILE
    )

    missing_columns = [
        column
        for column in CANONICAL_FIELDS + ["template"]
        if column not in df.columns
    ]

    if missing_columns:

        raise KeyError(
            "Required staged SDD column(s) missing: "
            + ", ".join(missing_columns)
        )

    return df


def load_fact_items(
    file_path: Path,
    dataset_name: str,
) -> set[int]:
    """Return distinct non-null Item codes used by one staged fact table."""

    if not file_path.is_file():

        raise FileNotFoundError(
            f"{dataset_name} file not found: {file_path}"
        )

    lazy_df = pl.scan_parquet(
        file_path
    )

    schema = lazy_df.collect_schema()

    if "item" not in schema.names():

        raise KeyError(
            f"{dataset_name} does not contain 'item'."
        )

    items = (
        lazy_df
        .select(
            pl.col("item")
            .drop_nulls()
            .unique()
        )
        .collect()["item"]
        .to_list()
    )

    return {
        int(item)
        for item in items
    }


# =============================================================================
# Source validation
# =============================================================================

def validate_source_item_table(
    df: pl.DataFrame,
) -> None:
    """
    Validate that staged SDD rows can safely be canonicalised by Item.

    For every Item, label/category/csv/collection must contain no more than
    one distinct non-null value.
    """

    print()
    print("=" * 100)
    print("SOURCE ITEM CANONICALISATION VALIDATION")
    print("=" * 100)

    row_count = df.height

    distinct_items = df[
        "item"
    ].n_unique()

    print(
        f"[INFO] Staged SDD rows      : "
        f"{row_count:,}"
    )

    print(
        f"[INFO] Distinct Item codes  : "
        f"{distinct_items:,}"
    )

    # -------------------------------------------------------------------------
    # Item completeness
    # -------------------------------------------------------------------------

    null_items = df[
        "item"
    ].null_count()

    if null_items != 0:

        raise ValueError(
            f"Staged dim_item contains {null_items:,} NULL Item value(s)."
        )

    print(
        "[PASS] item contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Definition consistency
    # -------------------------------------------------------------------------

    grouped = (
        df
        .group_by(
            "item"
        )
        .agg(
            [
                pl.col(column)
                .drop_nulls()
                .n_unique()
                .alias(
                    f"distinct_{column}"
                )
                for column in CONSISTENCY_FIELDS
            ]
        )
    )

    conflict_expressions = [
        pl.col(
            f"distinct_{column}"
        )
        > 1
        for column in CONSISTENCY_FIELDS
    ]

    conflict_condition = conflict_expressions[0]

    for expression in conflict_expressions[1:]:

        conflict_condition = (
            conflict_condition
            | expression
        )

    conflicts = grouped.filter(
        conflict_condition
    )

    if conflicts.height > 0:

        print()
        print(
            "Conflicting Item definitions:"
        )

        print(
            conflicts
        )

        raise ValueError(
            f"{conflicts.height:,} Item code(s) have conflicting "
            "analytical definitions. Canonicalisation stopped."
        )

    print(
        "[PASS] Repeated Items retain consistent "
        "label/category/csv/collection definitions."
    )

    # -------------------------------------------------------------------------
    # Source grain
    # -------------------------------------------------------------------------

    duplicate_item_template = (
        df
        .group_by(
            [
                "item",
                "template",
            ]
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

    if duplicate_item_template.height != 0:

        raise ValueError(
            "(item, template) is unexpectedly non-unique "
            "in the staged SDD."
        )

    print(
        "[PASS] Staged source grain (item + template) remains unique."
    )


# =============================================================================
# Build analytical dimension
# =============================================================================

def build_analytical_item_dimension(
    staged_item: pl.DataFrame,
    credit_items: set[int],
    other_items: set[int],
) -> pl.DataFrame:
    """
    Build one row per EBA Item.

    The canonical descriptive fields are safe to collapse because their
    consistency has already been validated.

    Additional usage fields describe how the Item appears in the two
    validated fact datasets.
    """

    # -------------------------------------------------------------------------
    # Canonical Item definition
    # -------------------------------------------------------------------------

    canonical = (
        staged_item
        .select(
            CANONICAL_FIELDS
        )
        .unique(
            maintain_order=True
        )
    )

    # -------------------------------------------------------------------------
    # Template count
    # -------------------------------------------------------------------------

    template_counts = (
        staged_item
        .group_by(
            "item"
        )
        .agg(
            pl.col("template")
            .drop_nulls()
            .n_unique()
            .alias(
                "template_count"
            )
        )
    )

    canonical = canonical.join(
        template_counts,
        on="item",
        how="left",
        validate="1:1",
    )

    # -------------------------------------------------------------------------
    # Fact usage flags
    # -------------------------------------------------------------------------

    canonical = (
        canonical
        .with_columns(
            [
                pl.col("item")
                .is_in(
                    list(
                        credit_items
                    )
                )
                .alias(
                    "used_in_credit_risk"
                ),

                pl.col("item")
                .is_in(
                    list(
                        other_items
                    )
                )
                .alias(
                    "used_in_other_templates"
                ),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Overall usage classification
    # -------------------------------------------------------------------------

    canonical = (
        canonical
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
    )

    return (
        canonical
        .sort(
            "item"
        )
    )


# =============================================================================
# Analytical dimension validation
# =============================================================================

def validate_analytical_dimension(
    df: pl.DataFrame,
    staged_item: pl.DataFrame,
    credit_items: set[int],
    other_items: set[int],
) -> None:
    """Validate analytical Item dimension before saving."""

    print()
    print("=" * 100)
    print("ANALYTICAL ITEM DIMENSION VALIDATION")
    print("=" * 100)

    expected_item_count = staged_item[
        "item"
    ].n_unique()

    # -------------------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------------------

    if df.height != expected_item_count:

        raise ValueError(
            "Analytical Item row count does not equal "
            "the number of distinct staged Item codes."
        )

    print(
        f"[PASS] Row count: "
        f"{df.height:,} one-row-per-Item records."
    )

    # -------------------------------------------------------------------------
    # Item uniqueness
    # -------------------------------------------------------------------------

    duplicate_items = (
        df["item"]
        .is_duplicated()
        .sum()
    )

    if duplicate_items != 0:

        raise ValueError(
            "Analytical Item dimension contains duplicate Item codes."
        )

    print(
        "[PASS] item is unique."
    )

    # -------------------------------------------------------------------------
    # Item completeness
    # -------------------------------------------------------------------------

    null_items = df[
        "item"
    ].null_count()

    if null_items != 0:

        raise ValueError(
            "Analytical Item dimension contains NULL Item codes."
        )

    print(
        "[PASS] item contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Label completeness
    # -------------------------------------------------------------------------

    null_labels = df[
        "label"
    ].null_count()

    if null_labels != 0:

        raise ValueError(
            f"Analytical Item dimension contains "
            f"{null_labels:,} NULL label(s)."
        )

    print(
        "[PASS] label contains 0 NULL values."
    )

    # -------------------------------------------------------------------------
    # Credit-risk fact coverage
    # -------------------------------------------------------------------------

    dimension_items = {
        int(item)
        for item in df[
            "item"
        ].to_list()
    }

    missing_credit_items = sorted(
        credit_items
        - dimension_items
    )

    if missing_credit_items:

        raise ValueError(
            "Credit-risk Item coverage failed. "
            f"Missing Item(s): {missing_credit_items}"
        )

    print(
        f"[PASS] All {len(credit_items):,} credit-risk "
        "Item codes are represented."
    )

    # -------------------------------------------------------------------------
    # Other-template fact coverage
    # -------------------------------------------------------------------------

    missing_other_items = sorted(
        other_items
        - dimension_items
    )

    if missing_other_items:

        raise ValueError(
            "Other-template Item coverage failed. "
            f"Missing Item(s): {missing_other_items}"
        )

    print(
        f"[PASS] All {len(other_items):,} other-template "
        "Item codes are represented."
    )

    # -------------------------------------------------------------------------
    # Union coverage
    # -------------------------------------------------------------------------

    fact_item_union = (
        credit_items
        | other_items
    )

    missing_union = sorted(
        fact_item_union
        - dimension_items
    )

    if missing_union:

        raise ValueError(
            f"Combined fact Item coverage failed: {missing_union}"
        )

    print(
        f"[PASS] All {len(fact_item_union):,} distinct Item codes "
        "used by the staged facts are represented."
    )


# =============================================================================
# Save
# =============================================================================

def save_dimension(
    df: pl.DataFrame,
) -> None:
    """Save analytical Item dimension."""

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
        "Saved analytical Item dimension:"
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
    """Re-read persisted analytical Item dimension."""

    if not OUTPUT_FILE.is_file():

        raise FileNotFoundError(
            f"Expected output file not found: {OUTPUT_FILE}"
        )

    saved = pl.read_parquet(
        OUTPUT_FILE
    )

    if saved.height != expected_rows:

        raise ValueError(
            f"Saved Item dimension row count mismatch. "
            f"Expected={expected_rows:,}, "
            f"saved={saved.height:,}"
        )

    if saved[
        "item"
    ].n_unique() != saved.height:

        raise ValueError(
            "Saved Item dimension Item code is not unique."
        )

    print()
    print("=" * 100)
    print("POST-SAVE VALIDATION")
    print("=" * 100)

    print(
        f"[PASS] Saved rows          : "
        f"{saved.height:,}"
    )

    print(
        f"[PASS] Distinct Item codes : "
        f"{saved['item'].n_unique():,}"
    )

    usage_summary = (
        saved
        .group_by(
            "fact_usage"
        )
        .agg(
            pl.len().alias(
                "item_count"
            )
        )
        .sort(
            "fact_usage"
        )
    )

    print()
    print(
        "Item usage summary:"
    )

    print(
        usage_summary
    )

    print()
    print(
        "Analytical Item sample:"
    )

    print(
        saved
        .select(
            [
                "item",
                "label",
                "category",
                "template_count",
                "used_in_credit_risk",
                "used_in_other_templates",
                "fact_usage",
            ]
        )
        .head(
            15
        )
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Build analytical Item dimension."""

    print("=" * 100)
    print("BUILDING ANALYTICAL ITEM DIMENSION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 100)

    print(
        f"Staged Item source : {STAGED_ITEM_FILE}"
    )

    print(
        f"Credit fact source : {CREDIT_FACT_FILE}"
    )

    print(
        f"Other fact source  : {OTHER_FACT_FILE}"
    )

    print(
        f"Output             : {OUTPUT_FILE}"
    )

    # -------------------------------------------------------------------------
    # Load validated sources
    # -------------------------------------------------------------------------

    staged_item = load_staged_item()

    credit_items = load_fact_items(
        CREDIT_FACT_FILE,
        "stg_credit_risk",
    )

    other_items = load_fact_items(
        OTHER_FACT_FILE,
        "stg_other_templates",
    )

    # -------------------------------------------------------------------------
    # Validate staged SDD canonicalisation
    # -------------------------------------------------------------------------

    validate_source_item_table(
        staged_item
    )

    # -------------------------------------------------------------------------
    # Build analytical Item dimension
    # -------------------------------------------------------------------------

    analytical_item = (
        build_analytical_item_dimension(
            staged_item=staged_item,
            credit_items=credit_items,
            other_items=other_items,
        )
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_analytical_dimension(
        df=analytical_item,
        staged_item=staged_item,
        credit_items=credit_items,
        other_items=other_items,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_dimension(
        analytical_item
    )

    # -------------------------------------------------------------------------
    # Validate persisted result
    # -------------------------------------------------------------------------

    validate_saved_dimension(
        expected_rows=analytical_item.height
    )

    print()
    print("=" * 100)
    print(
        "ANALYTICAL ITEM DIMENSION BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 100)

    print(
        "The staged SDD and staged fact tables were not modified."
    )


if __name__ == "__main__":
    main()
