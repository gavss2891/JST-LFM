"""
summarise_results.py

Reads all results/<dataset>/final_results.csv files and writes comparison
tables to results/summary/:
  - mse_overall.csv   — main models, rows=dataset, columns=model
  - mse_asym.csv      — JST-LFM vs JST-LFM-asym comparison
"""

import csv
import os
from pathlib import Path

RESULTS_DIR = Path('results')
OUT_DIR = RESULTS_DIR / 'summary'

MODELS_MAIN = ['Offset', 'LFM', 'LDAFirst', 'LDA-LFM', 'JSTFirst', 'JST-LFM']
COMPARISONS_MAIN = [
    ('LDA-LFM',  'JST-LFM',  'imp_LDA-LFM→JST-LFM'),
    ('LDAFirst', 'JSTFirst', 'imp_LDAFirst→JSTFirst'),
    ('JSTFirst', 'JST-LFM',  'imp_JSTFirst→JST-LFM'),
]

MODELS_ASYM = ['JST-LFM', 'JST-LFM-asym']
COMPARISONS_ASYM = [
    ('JST-LFM', 'JST-LFM-asym', 'imp_JST-LFM→JST-LFM-asym'),
]

STATS_COLS = ['n_reviews', 'n_users', 'n_items', 'density', 'total_words', 'rating_distribution']


def read_final_results(path):
    rows = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows[row['model']] = row
    return rows


def read_dataset_stats(path):
    if not path.exists():
        return {}
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def pct_improvement(baseline, improved):
    if baseline == '' or improved == '' or baseline == 0:
        return ''
    return round((float(baseline) - float(improved)) / float(baseline) * 100, 2)


def build_table(dataset_results, value_key, models, comparisons, dataset_stats=None):
    if dataset_stats is None:
        dataset_stats = {}
    table = []
    for dataset, model_rows in sorted(dataset_results.items()):
        raw = {}
        for model in models:
            val = model_rows.get(model, {}).get(value_key, '')
            try:
                raw[model] = float(val) if val != '' else ''
            except ValueError:
                raw[model] = ''

        numeric = {m: v for m, v in raw.items() if v != ''}
        winner = min(numeric, key=numeric.get) if numeric else None

        row = {'dataset': dataset}
        for model in models:
            v = raw[model]
            if v == '':
                row[model] = ''
            else:
                cell = f'{v:.6f}'
                if model == winner:
                    cell += '***'
                row[model] = cell

        for base_model, imp_model, col in comparisons:
            row[col] = pct_improvement(raw.get(base_model, ''), raw.get(imp_model, ''))

        stats = dataset_stats.get(dataset, {})
        for col in STATS_COLS:
            row[col] = stats.get(col, '')

        table.append(row)
    return table


def write_table(table, path, models, comparisons):
    os.makedirs(path.parent, exist_ok=True)
    imp_cols = [col for _, _, col in comparisons]
    fieldnames = ['dataset'] + models + imp_cols + STATS_COLS
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved → {path}")


def main():
    dataset_results = {}
    dataset_stats = {}
    for entry in sorted(RESULTS_DIR.iterdir()):
        if not entry.is_dir() or entry.name == 'summary':
            continue
        csv_path = entry / 'final_results.csv'
        if not csv_path.exists():
            continue
        dataset_results[entry.name] = read_final_results(csv_path)
        dataset_stats[entry.name] = read_dataset_stats(
            entry / f'{entry.name}_stats.csv'
        )

    write_table(
        build_table(dataset_results, 'test_mse', MODELS_MAIN, COMPARISONS_MAIN, dataset_stats),
        OUT_DIR / 'mse_overall.csv',
        MODELS_MAIN, COMPARISONS_MAIN,
    )
    write_table(
        build_table(dataset_results, 'test_mse', MODELS_ASYM, COMPARISONS_ASYM, dataset_stats),
        OUT_DIR / 'mse_asym.csv',
        MODELS_ASYM, COMPARISONS_ASYM,
    )


if __name__ == '__main__':
    main()
