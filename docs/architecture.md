# Architecture and source-to-KPI mapping

## Implemented flow

| Layer | Inputs and outputs | Purpose |
|---|---|---|
| Official source | `tr_cre.csv`, `tr_oth.csv`, `SDD.xlsx`, `TR_Metadata.xlsx` | EBA 2025 disclosures, definitions and reference codes |
| Staging | `data/staging/facts/` and `data/staging/reference/` | Standardise names and types, retaining rows and missing amounts |
| Facts | `data/analytical/facts/` | Two separate subject areas with source coordinates retained |
| Dimensions | `data/analytical/dimensions/` | 13 analytical dimensions, including institution, period and item |
| Marts | `data/analytical/marts/` | Four sequential analytical layers per risk module |
| Validation | `src/validation/` | Read-only comparison of saved results with source or independently reconstructed expectations |
| Reporting | Local Power BI report | Executive, asset-quality and capital pages |

The build order is explicit in [run_pipeline.py](../src/run_pipeline.py). A failed script stops the runner. Validators must still be read for non-blocking source warnings.

## Facts, grain and coverage

The credit-risk fact contains **648,951 rows**; the other-template fact **91,277 rows**. Template item and applicable dimensional coordinates define each observation. Source row, column and sheet identifiers support tracing values back to the disclosures. Amount units depend on the EBA item: not every amount column represents money.

Each KPI mart has one row per `lei_code × period`. The NPE marts contain 426 rows each; the CET1 marts 469 each. These totals include aggregate observations where present. Institution-level benchmarking and monitoring exclude the EBA “All other banks” aggregate (`XXXXXXXXXXXXXXXXXXXX`).

Cross-fact coverage is intentionally not forced to match. There are 472 credit-risk and 474 other-template entity-period combinations. Two Rabobank combinations, 2024 Q3 and 2025 Q1, are absent from the raw credit-risk file as well as its staging and analytical fact. The model preserves this source difference; it creates neither synthetic observations nor zero amounts.

The staged data dictionary repeats some item codes across templates. The analytical item dimension checks metadata consistency and consolidates item definitions before item-level joins; the staged dictionary is not used as an unchecked item-only many-to-one join.

## Source-to-KPI mapping

| Output | Source | Coordinates / interpretation |
|---|---|---|
| NPE denominator | Credit-risk item `2520603` | Total gross carrying amount: portfolio, country, exposure, status and performance status all `0` |
| NPE numerator | Credit-risk item `2520603` | Same total coordinates; performance status `2` for non-performing exposures |
| NPE ratio | Numerator / denominator | Selected loan-and-advance scope; zero denominator yields NULL |
| CET1 ratio | Other-template item `2520140` | Direct reported transitional CET1 capital ratio, stored as a fraction |
| CET1 reconciliation | Items `2520102` / `2520138` | CET1 capital divided by total risk exposure amount |
| Institution labels | `TR_Metadata.xlsx` | Name, country and aggregate classification |
| Period labels | Source period codes | `202409`, `202412`, `202503`, `202506` mapped to quarter-end dates |

For both modules, the ratio mart feeds the benchmark mart, which feeds the trend mart, which feeds the early-warning mart. Formulas and eligibility rules are in [methodology](methodology.md).

## Report model boundary

The local report imports the two early-warning marts separately. Each has its own automatic date-table relationship; there is no shared cross-risk institution/period model. The existing independent pages work within that scope. A combined NPE/CET1 dashboard would require an explicit shared-dimension design and renewed acceptance testing.

SQL, cloud deployment, scheduling and automatic report distribution are not operational components of this release.
