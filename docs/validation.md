# Validation evidence and limitations

**Verification date: 27 October 2026.** These are internal project controls, not third-party certification or a test of predictive performance.

## Executed checks

All 15 selected read-only validators completed with exit code 0 against the saved project data. Twelve report PASS; three report documented WARNING states, with no failed checks.

| Validator (`src/validation/validate_…py`) | Result |
|---|---|
| `reference_tables` | 178 pass, 1 warning, 0 fail |
| `credit_risk_staging` | 15 pass, 1 warning, 0 fail |
| `other_templates_staging` | 21 pass, 1 warning, 0 fail |
| `credit_risk_fact` | 82/82 pass |
| `other_templates_fact` | 90/90 pass |
| `analytical_dimensions` | 170/170 pass |
| `analytical_fact_layer` | 18/18 pass |
| `npe_ratio_mart` | PASS |
| `npe_benchmark_mart` | PASS |
| `npe_trend_mart` | PASS |
| `npe_early_warning_mart` | 49/49 pass |
| `cet1_ratio_mart` | 82/82 pass |
| `cet1_benchmark_mart` | 89/89 pass |
| `cet1_trend_mart` | 84/84 pass |
| `cet1_early_warning_mart` | 53/53 pass |

Counts follow each validator's final control summary. The three earlier NPE validators report PASS without a comparable numbered summary, so no synthetic total is assigned to them. Printed diagnostic messages are not automatically separate checks.

## Source warnings retained

- **Reference dictionary:** 34 item codes repeat across 84 staged dictionary rows because items can occur in multiple templates. The analytical item dimension handles the item-level consistency/consolidation; an unchecked join to the staged dictionary would multiply rows.
- **Credit-risk staging:** 1,637 NULL amounts preserve blank source values.
- **Other-template staging:** 3,964 NULL amounts preserve blank source values.

These warnings identify known source characteristics. They are not silently converted into complete data. Cross-fact entity-period differences are preserved and explained in [architecture](architecture.md).

## Reproduction and numerical reconciliation

The 26 transformation scripts were executed in dependency order in a separate local folder, starting from copies of the four official-source inputs. All 15 validators then ran successfully against the rebuilt files. All **23 analytical outputs**—two facts, 13 dimensions and eight marts—matched the existing outputs in schema and values, allowing for row-order differences. The original analytical data were not overwritten by this test.

This establishes a successful local rebuild using the existing environment and source snapshot. It does not establish installation on every platform, successful future download of revised EBA files, or recreation of the Power BI report.

The 2025 Q2 figures in the README were independently aggregated from the saved monitoring marts after excluding aggregate entities. Reconciliation confirmed NPE: 107 institutions, 104 assessable, 40 deteriorating, 12 strong attention; CET1: 119, 111, 55 and 21 respectively. Unweighted medians round to 2.08% and 18.00%.

## Report acceptance

All three pages were visually reviewed in their 2025 Q2 default view; the published screenshots were captured directly from Power BI Desktop on 27 October 2026. Focused prior live tests confirmed Capital Adequacy all-institution, Austria, BAWAG and first-quarter unavailable-comparison behaviour. Those checks do not cover every filter combination, cross-highlight or export layout.

Remaining limitations:

- Historical public data, with different institution populations across risk modules and periods.
- Small-country descriptive charts need sample-size context; formal country peer ranking requires at least five eligible institutions.
- Sign-based deterioration has no materiality threshold and has not been calibrated as a predictive risk model.
- Full interaction testing, empty-filter behaviour and report-export legibility remain open.
- The Power BI model supports separate risk pages; shared cross-risk filters are not implemented.
- Production scheduling, access management, automated distribution and customer deployment are not established.
- No paying-client, pilot-success or demand claim follows from these technical tests.

Run the controls yourself using [the reproduction guide](reproduction.md). Detailed local logs and working notes are intentionally excluded from this audience-facing repository.
