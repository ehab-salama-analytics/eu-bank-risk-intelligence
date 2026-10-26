# Analytical methodology

This guide consolidates the implemented NPE and CET1 rules. The linked scripts are the executable definitions. Coverage is the historical EBA 2025 exercise, 2024 Q3–2025 Q2.

## 1. Ratio definitions

**NPE ratio** = non-performing gross carrying amount / total gross carrying amount for the selected loans-and-advances scope. [Build script](../src/transformation/build_npe_ratio_mart.py). Item `2520603` uses total portfolio, country, exposure and status coordinates; numerator performance status is `2`, denominator performance status is `0`. Components must have matching institution-period coverage and unique coordinates. A zero denominator produces NULL. The numerator cannot materially exceed its denominator; `1e-9` is a numerical check tolerance, not a risk threshold.

**CET1 ratio** uses directly reported EBA item `2520140`, the transitional common equity tier 1 capital ratio. [Build script](../src/transformation/build_cet1_ratio_mart.py). It is independently reconciled against CET1 capital (`2520102`) divided by total risk exposure (`2520138`), with absolute tolerance `1e-12` in ratio units. Source-consistent high values are retained. No supervisory capital requirement is inferred.

Ratios are stored as fractions and displayed as percentages. A change of `0.01` in the fraction is **1 percentage point**, not a 1% relative change.

## 2. Benchmarking

Eligible peers are individual institutions with valid ratios in the same reporting period. Aggregate entities are excluded. European benchmarks use the available exercise population, not every European bank. Country grouping uses the institution's country, not the exposure-country coordinate.

Ranks are ascending, with ties assigned their average occupied rank. Percentile = `(average rank − 1) / (peer count − 1)`. Thus a higher percentile means a higher ratio: comparatively worse for NPE, comparatively higher capital for CET1. A percentile is not a probability of distress.

Country groups of **five or more** eligible institutions receive formal ranks and percentiles. Groups of **two to four** support descriptive comparisons only; singleton groups do not support peer ranking. Formal country-percentile movement requires eligible values in both adjacent quarters. Mean/median summaries are unweighted. Small descriptive groups should not be interpreted as robust country-wide estimates.

Scripts: [NPE benchmark](../src/transformation/build_npe_benchmark_mart.py), [CET1 benchmark](../src/transformation/build_cet1_benchmark_mart.py).

## 3. Trends and missing history

Quarter-on-quarter movement requires the same institution in genuinely adjacent calendar quarters. The next available row after a gap is not automatically a valid quarterly comparison. Three-period patterns require three consecutive quarterly observations, representing two successive changes.

Direct deterioration is a strictly positive NPE change or a strictly negative CET1 change. Zero movement does not trigger deterioration. Persistent deterioration means two successive NPE increases or CET1 decreases. The implemented sign rule has no materiality threshold, so small changes can trigger attention; interpretation remains an analyst's task.

Unavailable comparisons and signals remain NULL with explicit availability flags. They are not filled with zeros. A positive-support count counts observed true signals; an absent supporting signal does not establish the opposite economic condition. In particular, a WATCH label does not prove that all supporting evidence was available.

Quarterly median charts use each quarter's available population. They are not a matched-cohort estimate; population changes can affect the line. Peer-percentile movement also depends on the changing peer population.

Scripts: [NPE trend](../src/transformation/build_npe_trend_mart.py), [CET1 trend](../src/transformation/build_cet1_trend_mart.py).

## 4. Monitoring classification

The rules are evaluated in this order:

| Category | Rule |
|---|---|
| `NOT_ELIGIBLE_AGGREGATE` | Aggregate entity; no institution-level classification |
| `NOT_ASSESSABLE` | No genuine consecutive-quarter direct comparison |
| `NO_CURRENT_DETERIORATION_SIGNAL` | Valid direct comparison, without direct deterioration |
| `STRONG_ATTENTION` | Direct deterioration + persistent deterioration + at least one worsening eligible European or country peer position |
| `ELEVATED_ATTENTION` | Direct deterioration + at least one positive supporting signal, without meeting the strong rule |
| `WATCH` | Direct deterioration with no positive observed supporting signal |

Peer deterioration is a percentile increase for NPE and a percentile decrease for CET1. A worsening peer position alone cannot create an attention classification when the institution's own ratio does not deteriorate.

Scripts: [NPE monitoring](../src/transformation/build_npe_early_warning_mart.py), [CET1 monitoring](../src/transformation/build_cet1_early_warning_mart.py).

## 5. Interpreting the published snapshot

For 2025 Q2, NPE has 107 institutions but only 104 valid comparisons: 40 deteriorate, including 12 strong-attention cases. CET1 has 119 institutions and 111 valid comparisons: 55 deteriorate, including 21 strong-attention cases. Shares therefore use **40/104** and **55/111**, not the full current-quarter populations. Strong attention is a subset, not an extra count to add.

These are transparent management-review signals. They are not credit ratings, default probabilities, regulatory classifications or investment recommendations. Internal validation checks implementation consistency; it does not demonstrate predictive performance or customer demand.
