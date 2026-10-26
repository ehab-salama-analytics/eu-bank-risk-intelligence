from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "eba_2025"
    / "reference"
)

STAGING_REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "reference"
)

SDD_FILE = RAW_REFERENCE_DIR / "SDD.xlsx"
METADATA_FILE = RAW_REFERENCE_DIR / "TR_Metadata.xlsx"


def to_snake_case(column_name: str) -> str:
    """
    Convert a column name to lowercase snake_case.

    Examples:
        LEI_Code -> lei_code
        Desc_country -> desc_country
        Financial_instruments -> financial_instruments
    """
    column_name = str(column_name).strip()

    column_name = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        column_name,
    )

    column_name = re.sub(
        r"_+",
        "_",
        column_name,
    )

    return column_name.strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe column names.
    """
    df = df.copy()

    df.columns = [
        to_snake_case(column)
        for column in df.columns
    ]

    return df


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic structural cleaning without changing business meaning.
    """
    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all",
    )

    # Standardize column names
    df = standardize_columns(df)

    return df


def save_parquet(
    df: pd.DataFrame,
    output_name: str,
) -> None:
    """
    Save a dataframe to the staging reference directory.
    """
    output_path = (
        STAGING_REFERENCE_DIR
        / f"{output_name}.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"{output_name:<30} "
        f"rows={len(df):>6,} "
        f"columns={len(df.columns):>3} "
        f"-> {output_path.name}"
    )


def load_sdd() -> pd.DataFrame:
    """
    Load the Single Data Dictionary.

    The real header is Excel row index 1.
    """
    df = pd.read_excel(
        SDD_FILE,
        sheet_name="SDD",
        header=1,
        engine="openpyxl",
    )

    return clean_table(df)


def load_institutions() -> pd.DataFrame:
    """
    Load institution metadata.

    The real header is Excel row index 1.
    """
    df = pd.read_excel(
        METADATA_FILE,
        sheet_name="List of Institutions",
        header=1,
        engine="openpyxl",
    )

    return clean_table(df)


def load_lookup_sheet(
    sheet_name: str,
) -> pd.DataFrame:
    """
    Load a standard two- or three-column metadata lookup sheet.
    """
    df = pd.read_excel(
        METADATA_FILE,
        sheet_name=sheet_name,
        header=0,
        engine="openpyxl",
    )

    return clean_table(df)


def main() -> None:

    print("=" * 80)
    print("BUILDING EBA REFERENCE TABLES")
    print("=" * 80)

    STAGING_REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Main reference tables
    sdd = load_sdd()
    institutions = load_institutions()

    save_parquet(
        sdd,
        "dim_item",
    )

    save_parquet(
        institutions,
        "dim_institution",
    )

    # Dimension lookup tables
    lookup_tables = {
        "Portfolio": "dim_portfolio",
        "Country": "dim_country",
        "Financial_instruments": "dim_financial_instrument",
        "Exposure": "dim_exposure",
        "Status": "dim_status",
        "Perf_status": "dim_perf_status",
        "ASSETS_Stages": "dim_asset_stage",
        "ASSETS_FV": "dim_asset_fv",
        "NACE_codes": "dim_nace",
        "Fin_end_year": "dim_fin_year_end",
    }

    for sheet_name, output_name in lookup_tables.items():

        df = load_lookup_sheet(
            sheet_name
        )

        save_parquet(
            df,
            output_name,
        )

    print("=" * 80)
    print("Reference staging build completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
