# Reproduce the analytical pipeline

## Data provenance and reuse

Source: **European Banking Authority, 2025 EU-wide Transparency Exercise**, observation dates 30 September 2024, 31 December 2024, 31 March 2025 and 30 June 2025.

Open the [official EBA exercise page](https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/eu-wide-transparency-exercise), select **2025 → Full database**, and obtain **Credit risk**, **Other templates**, **Data dictionary** and **Metadata**. Extract archives if supplied. The required local filenames are below. Market-risk and sovereign files are not required for these modules. Download from the official source; do not substitute another exercise year.

The [EBA legal notice](https://www.eba.europa.eu/find-out-about-us/legal-notice) permits reproduction of EBA material with source acknowledgement unless otherwise stated, with separate restrictions for third-party material and the EBA logo. Check any notices attached to the downloaded files. This repository distributes project code, explanations and attributed report images; it does not redistribute raw files, generated datasets or EBA logos. EBA data and third-party dependencies retain their own terms. No additional open-source licence grant is declared for the project in this release.

## Setup

Tested on Windows with **Python 3.13.15**, Polars 1.44.2, Pandas 3.0.6, PyArrow 25.0.1 and OpenPyXL 3.1.5. The full tested dependency set is in `requirements-lock.txt`; `requirements.txt` lists only direct pipeline dependencies.

```sh
git clone https://github.com/ehab-salama-analytics/eu-bank-risk-intelligence.git
cd eu-bank-risk-intelligence
python -m venv .venv
```

In Windows PowerShell, no activation is required:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -X utf8 src/validate_environment.py
```

On another platform, use `.venv/bin/python` and `requirements.txt` so pip resolves platform dependencies. Other platforms have not been tested. The publication rebuild used the existing tested environment; installation in a completely fresh environment has not been verified.

## Required input layout

```text
data/raw/eba_2025/
├── tr_cre.csv
├── tr_oth.csv
└── reference/
    ├── SDD.xlsx
    └── TR_Metadata.xlsx
```

Keep source contents unchanged. The reference loader reads the `SDD` and `List of Institutions` worksheets with header row index 1, and the metadata lookup sheets with header index 0. File and worksheet names are significant. Inputs are not included in Git.

## Execute

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/run_pipeline.py all
```

Or run the build and validation separately:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/run_pipeline.py build
.\.venv\Scripts\python.exe -X utf8 src/run_pipeline.py validate
```

`build` overwrites generated staging and analytical outputs, leaving raw inputs unchanged. `validate` reads existing files without modifying the data. `all` performs both in sequence. The runner stops at the first nonzero exit code; review warnings even when the command succeeds. UTF-8 mode prevents redirected Windows consoles from failing on printed table characters.

The [runner](../src/run_pipeline.py) executes:

1. Reference staging, then credit-risk and other-template staging.
2. Credit-risk and other-template analytical facts.
3. Institution, period, item and ten supporting analytical dimensions.
4. NPE ratio → benchmark → trend → monitoring.
5. CET1 ratio → benchmark → trend → monitoring.
6. Fifteen reference, staging, fact, dimension and mart validators.

To inspect a particular control independently:

```sh
python -X utf8 src/validation/validate_npe_early_warning_mart.py
python -X utf8 src/validation/validate_cet1_early_warning_mart.py
```

## Expected outputs

| Location | Expected result |
|---|---|
| `data/staging/reference/` | 12 reference Parquet tables |
| `data/staging/facts/` | Two staged facts |
| `data/analytical/facts/` | Credit-risk: 648,951 rows; other-template: 91,277 rows |
| `data/analytical/dimensions/` | 13 analytical dimension tables |
| `data/analytical/marts/` | Four `mart_npe_*` tables, 426 rows each; four `mart_cet1_*` tables, 469 rows each |

These counts are specific to the reviewed source snapshot. Several build and validation controls deliberately assert them. If EBA revises a file, investigate the difference before updating expectations. Do not simply bypass a failing control or fabricate missing observations.

The publication test rebuilt the pipeline in an isolated folder from the four local EBA source files, ran all 15 validators and compared all 23 analytical outputs with the saved project outputs. See [validation evidence](validation.md).

## Power BI boundary

The public release does not include a PBIX/PBIT/PBIP report or its embedded data and machine-local connections. It therefore does **not** recreate the authored report automatically. The two early-warning marts can be loaded into a separately created report; [the dashboard guide](dashboard.md) describes the existing views, filters and interpretation. A Python build does not refresh an already open Power BI file.
