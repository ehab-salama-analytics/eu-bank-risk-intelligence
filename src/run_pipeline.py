"""Run the EBA 2025 build or read-only validation in dependency order."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD = [
    'reference_tables', 'credit_risk_staging', 'other_templates_staging',
    'credit_risk_fact', 'other_templates_fact',
    'institution_dimension', 'period_dimension', 'item_dimension',
    'country_dimension', 'portfolio_dimension', 'exposure_dimension',
    'status_dimension', 'perf_status_dimension',
    'financial_instrument_dimension', 'fin_year_end_dimension',
    'asset_stage_dimension', 'asset_fv_dimension', 'nace_dimension',
    'npe_ratio_mart', 'npe_benchmark_mart', 'npe_trend_mart',
    'npe_early_warning_mart', 'cet1_ratio_mart', 'cet1_benchmark_mart',
    'cet1_trend_mart', 'cet1_early_warning_mart',
]
VALIDATE = [
    'reference_tables', 'credit_risk_staging', 'other_templates_staging',
    'credit_risk_fact', 'other_templates_fact', 'analytical_dimensions',
    'analytical_fact_layer', 'npe_ratio_mart', 'npe_benchmark_mart',
    'npe_trend_mart', 'npe_early_warning_mart', 'cet1_ratio_mart',
    'cet1_benchmark_mart', 'cet1_trend_mart', 'cet1_early_warning_mart',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'validate', 'all'],
                        help='build overwrites generated outputs; validate is read-only')
    args = parser.parse_args()
    if args.action in ('build', 'all'):
        raw = ROOT / 'data/raw/eba_2025'
        for name in ['tr_cre.csv', 'tr_oth.csv', 'reference/SDD.xlsx',
                     'reference/TR_Metadata.xlsx']:
            if not (raw / name).is_file():
                parser.error(f'Missing input: data/raw/eba_2025/{name}. See docs/reproduction.md.')
    phases = []
    if args.action in ('build', 'all'):
        phases.append(('transformation', 'build', BUILD))
    if args.action in ('validate', 'all'):
        phases.append(('validation', 'validate', VALIDATE))
    for folder, prefix, names in phases:
        for name in names:
            script = ROOT / 'src' / folder / f'{prefix}_{name}.py'
            print(f'\n>>> {script.relative_to(ROOT).as_posix()}', flush=True)
            # UTF-8 prevents Windows redirected-console failures on table glyphs.
            subprocess.run([sys.executable, '-X', 'utf8', str(script)],
                           cwd=ROOT, check=True)
    print('\nCompleted. Review validator warnings as well as exit codes.')


if __name__ == '__main__':
    main()
