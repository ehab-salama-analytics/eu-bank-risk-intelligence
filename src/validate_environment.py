import sys
import platform

import pandas as pd
import polars as pl
import duckdb
import pyarrow as pa
import sqlalchemy
import numpy as np
import scipy
import statsmodels
import sklearn
import matplotlib
import seaborn
import plotly
import pandera
import pytest


def main():
    print("=" * 60)
    print("EU BANK RISK INTELLIGENCE - ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"Python version      : {sys.version.split()[0]}")
    print(f"Platform            : {platform.platform()}")
    print()

    packages = {
        "pandas": pd.__version__,
        "polars": pl.__version__,
        "duckdb": duckdb.__version__,
        "pyarrow": pa.__version__,
        "sqlalchemy": sqlalchemy.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "plotly": plotly.__version__,
        "pandera": pandera.__version__,
        "pytest": pytest.__version__,
    }

    for name, version in packages.items():
        print(f"{name:<18}: {version}")

    print()
    print("=" * 60)
    print("Environment validation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    