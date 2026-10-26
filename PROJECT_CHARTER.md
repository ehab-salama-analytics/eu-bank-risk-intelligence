# Project scope

## Purpose and intended users

European Bank Risk Intelligence & Automated Reporting Platform demonstrates how banking-risk and reporting teams can turn official disclosures into traceable indicators, fair peer comparisons and explainable monitoring views. It is an independent portfolio case study using historical EBA data, not a commissioned client engagement.

## Delivered scope

- EBA 2025 Transparency Exercise data for 2024 Q3 through 2025 Q2.
- Python ingestion and transformation of credit-risk and other-template disclosures, with reference metadata and analytical dimensions.
- NPE and CET1 ratio, benchmark, trend and monitoring layers.
- Separate read-only validation scripts and a tested local reproduction workflow.
- Three local Power BI report pages with genuine public screenshots.

## Acceptance evidence

On 27 October 2026, a separate rebuild reproduced all 23 analytical tables and completed 15 validators without blocking failures. Three documented source-data warnings remain visible. The public findings were reconciled to the saved marts. Power BI default views and focused capital interactions have been reviewed; full interaction and export acceptance remain open.

## Boundaries

No live feed, production scheduling, automatic report distribution, predictive default model or regulatory breach assessment is claimed. Raw data, generated datasets and embedded Power BI files are outside this initial public release. Source snapshots and fixed population controls must be reviewed before extending to a new exercise.

## Commercial relevance

The project supports QuantRisk Analytics' proposed service-led model: **ANALYSE → AUTOMATE → TRAIN**. It demonstrates the ability to define financial KPIs, reconcile source data, automate repeatable processing and explain results. Prospective demand, paid pilots and customer outcomes remain to be established.

See the [README](README.md) for findings, [architecture](docs/architecture.md) for implementation and [validation evidence](docs/validation.md) for detailed boundaries.
