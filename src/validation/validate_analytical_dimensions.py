"""
Integrated validation of the analytical dimension layer.

Project:
    European Bank Risk Intelligence & Automated Reporting Platform

Phase:
    Phase 3A-6 — Integrated Analytical Dimension Validation

Purpose:
    Validate the analytical dimensions as a complete reusable semantic layer
    before analytical fact-table construction and dimensional joins.

Validation principles:
    - READ ONLY
    - no data modification
    - no deduplication
    - no replacement of missing values
    - no automatic correction
    - failures must be investigated rather than silently repaired

Inputs:
    data/analytical/dimensions/*.parquet

Outputs:
    Terminal validation report only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DIMENSION_DIR = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "dimensions"
)


# =============================================================================
# Validation configuration
# =============================================================================

@dataclass(frozen=True)
class DimensionSpec:
    name: str
    filename: str
    key_column: str
    expected_rows: int
    required_columns: tuple[str, ...]


DIMENSIONS: tuple[DimensionSpec, ...] = (

    DimensionSpec(
        name="dim_period",
        filename="dim_period.parquet",
        key_column="period_key",
        expected_rows=4,
        required_columns=(
            "period_key",
            "period_code",
            "period_end_date",
            "year",
            "quarter",
            "month",
            "year_quarter",
            "is_year_end",
        ),
    ),

    DimensionSpec(
        name="dim_item_analytic",
        filename="dim_item_analytic.parquet",
        key_column="item",
        expected_rows=285,
        required_columns=(
            "item",
            "label",
            "category",
            "template_count",
            "used_in_credit_risk",
            "used_in_other_templates",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_institution",
        filename="dim_institution.parquet",
        key_column="lei_code",
        expected_rows=120,
        required_columns=(
            "lei_code",
            "name",
            "entity_type",
            "is_aggregate",
            "record_source",
        ),
    ),

    DimensionSpec(
        name="dim_exposure",
        filename="dim_exposure.parquet",
        key_column="exposure",
        expected_rows=52,
        required_columns=(
            "exposure",
            "label",
            "used_in_credit_risk",
            "used_in_other_templates",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_country",
        filename="dim_country.parquet",
        key_column="country",
        expected_rows=359,
        required_columns=(
            "country",
            "label",
            "iso_code",
            "used_in_credit_risk",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_portfolio",
        filename="dim_portfolio.parquet",
        key_column="portfolio",
        expected_rows=8,
        required_columns=(
            "portfolio",
            "label",
            "used_in_credit_risk",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_status",
        filename="dim_status.parquet",
        key_column="status",
        expected_rows=5,
        required_columns=(
            "status",
            "label",
            "used_in_credit_risk",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_perf_status",
        filename="dim_perf_status.parquet",
        key_column="perf_status",
        expected_rows=12,
        required_columns=(
            "perf_status",
            "label",
            "used_in_credit_risk",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_nace",
        filename="dim_nace.parquet",
        key_column="nace_codes",
        expected_rows=20,
        required_columns=(
            "nace_codes",
            "label",
            "used_in_credit_risk",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_asset_stage",
        filename="dim_asset_stage.parquet",
        key_column="assets_stages",
        expected_rows=4,
        required_columns=(
            "assets_stages",
            "label",
            "used_in_other_templates",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_asset_fv",
        filename="dim_asset_fv.parquet",
        key_column="assets_fv",
        expected_rows=4,
        required_columns=(
            "assets_fv",
            "label",
            "used_in_other_templates",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_financial_instrument",
        filename="dim_financial_instrument.parquet",
        key_column="financial_instruments",
        expected_rows=9,
        required_columns=(
            "financial_instruments",
            "label",
            "used_in_other_templates",
            "fact_usage",
        ),
    ),

    DimensionSpec(
        name="dim_fin_year_end",
        filename="dim_fin_year_end.parquet",
        key_column="fin_end_year",
        expected_rows=4,
        required_columns=(
            "fin_end_year",
            "label",
            "used_in_other_templates",
            "fact_usage",
        ),
    ),
)


# =============================================================================
# Result tracking
# =============================================================================

RESULTS: list[dict[str, str]] = []


def record(
    status: str,
    dimension: str,
    check: str,
    message: str,
) -> None:
    """Record and print one validation result."""

    RESULTS.append(
        {
            "status": status,
            "dimension": dimension,
            "check": check,
            "message": message,
        }
    )

    print(
        f"[{status:<7}] "
        f"{dimension:<26} | "
        f"{check:<24} | "
        f"{message}"
    )


# =============================================================================
# Generic validation helpers
# =============================================================================

def validate_dimension(
    spec: DimensionSpec,
) -> None:
    """
    Validate one analytical dimension.
    """

    file_path = (
        DIMENSION_DIR
        / spec.filename
    )

    print()
    print("=" * 110)
    print(
        f"DIMENSION: {spec.name}"
    )
    print(
        f"FILE     : {file_path}"
    )
    print("=" * 110)

    # -------------------------------------------------------------------------
    # File existence
    # -------------------------------------------------------------------------

    if not file_path.is_file():
        record(
            "FAIL",
            spec.name,
            "file_exists",
            f"Missing expected file: {spec.filename}",
        )
        return

    record(
        "PASS",
        spec.name,
        "file_exists",
        f"Found {spec.filename}",
    )

    # -------------------------------------------------------------------------
    # Parquet read
    # -------------------------------------------------------------------------

    try:
        df = pl.read_parquet(
            file_path
        )

    except Exception as exc:
        record(
            "FAIL",
            spec.name,
            "parquet_read",
            f"Unable to read Parquet: {exc}",
        )
        return

    record(
        "PASS",
        spec.name,
        "parquet_read",
        "Parquet file opened successfully",
    )

    # -------------------------------------------------------------------------
    # Dimensions
    # -------------------------------------------------------------------------

    record(
        "PASS",
        spec.name,
        "table_dimensions",
        f"rows={df.height:,}, columns={df.width:,}",
    )

    # -------------------------------------------------------------------------
    # Non-empty
    # -------------------------------------------------------------------------

    if df.height == 0:
        record(
            "FAIL",
            spec.name,
            "non_empty_table",
            "Dimension contains 0 rows",
        )
        return

    record(
        "PASS",
        spec.name,
        "non_empty_table",
        f"Dimension contains {df.height:,} row(s)",
    )

    # -------------------------------------------------------------------------
    # Expected row count
    # -------------------------------------------------------------------------

    if df.height == spec.expected_rows:
        record(
            "PASS",
            spec.name,
            "expected_row_count",
            f"Expected {spec.expected_rows:,} row(s)",
        )

    else:
        record(
            "FAIL",
            spec.name,
            "expected_row_count",
            (
                f"Expected {spec.expected_rows:,} row(s), "
                f"found {df.height:,}"
            ),
        )

    # -------------------------------------------------------------------------
    # Duplicate column names
    # -------------------------------------------------------------------------

    if len(df.columns) == len(set(df.columns)):
        record(
            "PASS",
            spec.name,
            "duplicate_columns",
            "Column names are unique",
        )

    else:
        record(
            "FAIL",
            spec.name,
            "duplicate_columns",
            "Duplicate column names detected",
        )

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    missing_columns = [
        column
        for column in spec.required_columns
        if column not in df.columns
    ]

    if not missing_columns:
        record(
            "PASS",
            spec.name,
            "required_columns",
            (
                f"All {len(spec.required_columns):,} "
                "required column(s) present"
            ),
        )

    else:
        record(
            "FAIL",
            spec.name,
            "required_columns",
            f"Missing columns: {missing_columns}",
        )

    # -------------------------------------------------------------------------
    # Key exists
    # -------------------------------------------------------------------------

    if spec.key_column not in df.columns:
        record(
            "FAIL",
            spec.name,
            "key_column_exists",
            (
                f"Key column '{spec.key_column}' "
                "not found"
            ),
        )
        return

    record(
        "PASS",
        spec.name,
        "key_column_exists",
        f"Key column '{spec.key_column}' found",
    )

    # -------------------------------------------------------------------------
    # NULL keys
    # -------------------------------------------------------------------------

    null_keys = (
        df[
            spec.key_column
        ]
        .null_count()
    )

    if null_keys == 0:
        record(
            "PASS",
            spec.name,
            "key_null_values",
            (
                f"No NULL values in "
                f"'{spec.key_column}'"
            ),
        )

    else:
        record(
            "FAIL",
            spec.name,
            "key_null_values",
            (
                f"{null_keys:,} NULL value(s) in "
                f"'{spec.key_column}'"
            ),
        )

    # -------------------------------------------------------------------------
    # Blank string keys
    # -------------------------------------------------------------------------

    key_dtype = df.schema[
        spec.key_column
    ]

    if key_dtype == pl.String:

        blank_keys = (
            df
            .filter(
                pl.col(
                    spec.key_column
                )
                .fill_null("")
                .str.strip_chars()
                .eq("")
            )
            .height
        )

        if blank_keys == 0:
            record(
                "PASS",
                spec.name,
                "key_blank_values",
                (
                    f"No blank values in "
                    f"'{spec.key_column}'"
                ),
            )

        else:
            record(
                "FAIL",
                spec.name,
                "key_blank_values",
                (
                    f"{blank_keys:,} blank value(s) in "
                    f"'{spec.key_column}'"
                ),
            )

    else:
        record(
            "PASS",
            spec.name,
            "key_blank_values",
            "Not applicable to non-string key",
        )

    # -------------------------------------------------------------------------
    # Key uniqueness
    # -------------------------------------------------------------------------

    distinct_keys = (
        df[
            spec.key_column
        ]
        .n_unique()
    )

    if distinct_keys == df.height:
        record(
            "PASS",
            spec.name,
            "key_uniqueness",
            (
                f"{distinct_keys:,} distinct key(s) "
                f"across {df.height:,} row(s)"
            ),
        )

    else:

        duplicate_groups = (
            df
            .group_by(
                spec.key_column
            )
            .len()
            .filter(
                pl.col("len") > 1
            )
            .height
        )

        record(
            "FAIL",
            spec.name,
            "key_uniqueness",
            (
                f"{duplicate_groups:,} duplicate key group(s); "
                f"{distinct_keys:,} distinct key(s) "
                f"across {df.height:,} row(s)"
            ),
        )

    # -------------------------------------------------------------------------
    # Label quality
    # -------------------------------------------------------------------------

    if "label" in df.columns:

        null_labels = (
            df[
                "label"
            ]
            .null_count()
        )

        blank_labels = (
            df
            .filter(
                pl.col("label")
                .fill_null("")
                .str.strip_chars()
                .eq("")
            )
            .height
        )

        if (
            null_labels == 0
            and blank_labels == 0
        ):

            record(
                "PASS",
                spec.name,
                "label_completeness",
                "No NULL or blank analytical labels",
            )

        else:

            record(
                "WARNING",
                spec.name,
                "label_completeness",
                (
                    f"NULL labels={null_labels:,}; "
                    f"blank labels={blank_labels:,}"
                ),
            )

    else:
        record(
            "PASS",
            spec.name,
            "label_completeness",
            "Not applicable to this dimension",
        )

    # -------------------------------------------------------------------------
    # Usage metadata consistency
    # -------------------------------------------------------------------------

    if "fact_usage" in df.columns:

        null_usage = (
            df[
                "fact_usage"
            ]
            .null_count()
        )

        if null_usage == 0:
            record(
                "PASS",
                spec.name,
                "fact_usage_completeness",
                "fact_usage contains no NULL values",
            )

        else:
            record(
                "WARNING",
                spec.name,
                "fact_usage_completeness",
                (
                    f"fact_usage contains "
                    f"{null_usage:,} NULL value(s)"
                ),
            )

    # -------------------------------------------------------------------------
    # Informational schema output
    # -------------------------------------------------------------------------

    print()
    print("Schema:")

    for column_name, dtype in df.schema.items():
        print(
            f"  {column_name:<30} {dtype}"
        )


# =============================================================================
# Cross-dimension controls
# =============================================================================

def validate_cross_dimension_controls() -> None:
    """
    Perform several model-level controls after individual dimension checks.
    """

    print()
    print("=" * 110)
    print("CROSS-DIMENSION CONTROLS")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # No duplicate configured filenames
    # -------------------------------------------------------------------------

    filenames = [
        spec.filename
        for spec in DIMENSIONS
    ]

    if len(filenames) == len(set(filenames)):
        record(
            "PASS",
            "MODEL",
            "unique_filenames",
            "All configured analytical dimension filenames are unique",
        )

    else:
        record(
            "FAIL",
            "MODEL",
            "unique_filenames",
            "Duplicate configured analytical dimension filenames found",
        )

    # -------------------------------------------------------------------------
    # No duplicate configured business keys
    #
    # Repeated business key names can be legitimate across separate dimensions,
    # so here we check only duplicate dimension-name definitions.
    # -------------------------------------------------------------------------

    dimension_names = [
        spec.name
        for spec in DIMENSIONS
    ]

    if len(dimension_names) == len(set(dimension_names)):
        record(
            "PASS",
            "MODEL",
            "unique_dimension_names",
            "All analytical dimension names are unique",
        )

    else:
        record(
            "FAIL",
            "MODEL",
            "unique_dimension_names",
            "Duplicate analytical dimension definitions found",
        )

    # -------------------------------------------------------------------------
    # Expected file count
    # -------------------------------------------------------------------------

    existing_expected_files = sum(
        1
        for spec in DIMENSIONS
        if (
            DIMENSION_DIR
            / spec.filename
        ).is_file()
    )

    if existing_expected_files == len(DIMENSIONS):

        record(
            "PASS",
            "MODEL",
            "expected_dimensions",
            (
                f"All {len(DIMENSIONS):,} expected "
                "analytical dimensions exist"
            ),
        )

    else:

        record(
            "FAIL",
            "MODEL",
            "expected_dimensions",
            (
                f"{existing_expected_files:,} of "
                f"{len(DIMENSIONS):,} expected dimensions exist"
            ),
        )


# =============================================================================
# Summary
# =============================================================================

def print_summary() -> None:
    """
    Print consolidated validation summary.
    """

    pass_count = sum(
        result["status"] == "PASS"
        for result in RESULTS
    )

    warning_count = sum(
        result["status"] == "WARNING"
        for result in RESULTS
    )

    fail_count = sum(
        result["status"] == "FAIL"
        for result in RESULTS
    )

    total_count = len(
        RESULTS
    )

    failed_dimensions = sorted(
        {
            result["dimension"]
            for result in RESULTS
            if (
                result["status"] == "FAIL"
                and result["dimension"] != "MODEL"
            )
        }
    )

    warning_dimensions = sorted(
        {
            result["dimension"]
            for result in RESULTS
            if (
                result["status"] == "WARNING"
                and result["dimension"] != "MODEL"
            )
        }
    )

    print()
    print("=" * 110)
    print("ANALYTICAL DIMENSION VALIDATION SUMMARY")
    print("=" * 110)

    print(
        f"Dimension directory : {DIMENSION_DIR}"
    )

    print(
        f"Expected dimensions : {len(DIMENSIONS):,}"
    )

    print(
        f"Total checks        : {total_count:,}"
    )

    print(
        f"PASS                : {pass_count:,}"
    )

    print(
        f"WARNING             : {warning_count:,}"
    )

    print(
        f"FAIL                : {fail_count:,}"
    )

    print()

    print(
        "Dimensions with FAIL    : "
        + (
            ", ".join(
                failed_dimensions
            )
            if failed_dimensions
            else "None"
        )
    )

    print(
        "Dimensions with WARNING : "
        + (
            ", ".join(
                warning_dimensions
            )
            if warning_dimensions
            else "None"
        )
    )

    print("-" * 110)

    if fail_count > 0:

        print(
            "OVERALL STATUS : FAIL"
        )

        print(
            "INTERPRETATION  : One or more analytical dimension "
            "integrity controls failed."
        )

        print(
            "ACTION          : Investigate before building "
            "analytical fact tables."
        )

    elif warning_count > 0:

        print(
            "OVERALL STATUS : WARNING"
        )

        print(
            "INTERPRETATION  : No blocking dimension failures, "
            "but documented observations require review."
        )

        print(
            "ACTION          : Review warnings before approving "
            "the analytical dimension layer."
        )

    else:

        print(
            "OVERALL STATUS : PASS"
        )

        print(
            "INTERPRETATION  : The analytical dimension layer "
            "passed the configured structural and key-integrity controls."
        )

        print(
            "ACTION          : Dimension layer can be considered "
            "for approval before analytical fact construction."
        )

    print()
    print(
        "VALIDATION MODE : READ ONLY"
    )

    print(
        "No analytical, staging, or raw data was modified."
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Run integrated analytical dimension validation.
    """

    print("=" * 110)
    print("EBA ANALYTICAL DIMENSION LAYER VALIDATION")
    print(
        "European Bank Risk Intelligence "
        "& Automated Reporting Platform"
    )
    print("=" * 110)

    print(
        f"Project root        : {PROJECT_ROOT}"
    )

    print(
        f"Dimension directory : {DIMENSION_DIR}"
    )

    print()

    print(
        "VALIDATION MODE: READ ONLY"
    )

    print(
        "No analytical, staging, or raw data will be modified."
    )

    for spec in DIMENSIONS:
        validate_dimension(
            spec
        )

    validate_cross_dimension_controls()

    print_summary()


if __name__ == "__main__":
    main()
