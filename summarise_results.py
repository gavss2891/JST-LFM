"""
summarise_results.py

Reads all results/<dataset>/final_results.csv files and writes comparison
tables to results/summary/:
  - mse_overall.csv        — main models, rows=dataset, columns=model
  - mse_asym.csv           — JST-LFM vs JST-LFM-asym comparison
  - mse_2sent_vs_3sent.csv — 2-sentiment vs 3-sentiment JST-LFM-asym comparison
"""

import csv
import os
from pathlib import Path

RESULTS_DIR = Path('results')
OUT_DIR = RESULTS_DIR / 'summary'

KS_TUNE_DIR  = Path('/Users/gavinshao/Desktop/Master Thesis/Code/results_tune_ks (S=3)')
KS_WIDE_PATH = KS_TUNE_DIR / 'Ks_summary_wide.csv'

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

MODELS_2SENT = ['JST-LFM-asym-2sent', 'JST-LFM-asym-2sent-Ks-tuned', 'JST-LFM', 'JST-LFM-asym']
COMPARISONS_2SENT = [
    ('JST-LFM-asym-2sent-Ks-tuned', 'JST-LFM-asym', 'imp_JST-LFM-asym-2sent-Ks-tuned→JST-LFM-asym'),
]

MODELS_ALL = ['Offset', 'LFM', 'LDAFirst', 'LDA-LFM', 'JSTFirst', 'JST-LFM', 'JST-LFM-asym']
COMPARISONS_ALL = [
    ('LDA-LFM',  'JST-LFM',      'imp_LDA-LFM→JST-LFM'),
    ('LDAFirst', 'JSTFirst',     'imp_LDAFirst→JSTFirst'),
    ('JSTFirst', 'JST-LFM',      'imp_JSTFirst→JST-LFM'),
    ('JST-LFM',  'JST-LFM-asym', 'imp_JST-LFM→JST-LFM-asym'),
    ('LDA-LFM',  'JST-LFM-asym', 'imp_LDA-LFM→JST-LFM-asym'),
]

STATS_COLS = ['n_reviews', 'n_users', 'n_items', 'density', 'n_tokens', 'rating_distribution']

MODELS_RUNTIME = [
    'Offset', 'LFM', 'LDAFirst', 'LDA-LFM',
    'JSTFirst', 'JST-LFM',
]


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
    for dataset, model_rows in dataset_results.items():
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
    table.sort(key=lambda r: int(r.get('n_tokens') or 0), reverse=True)
    return table


def average_row(table, models, comparisons):
    imp_cols = [col for _, _, col in comparisons]
    avg = {'dataset': 'AVERAGE'}
    for col in models + imp_cols:
        vals = []
        for row in table:
            v = row.get(col, '')
            if v == '':
                continue
            try:
                vals.append(float(str(v).replace('***', '')))
            except ValueError:
                pass
        avg[col] = round(sum(vals) / len(vals), 6) if vals else ''
    for col in STATS_COLS:
        avg[col] = ''
    return avg


def build_runtime_table(dataset_results, models, dataset_stats=None):
    if dataset_stats is None:
        dataset_stats = {}
    table = []
    for dataset, model_rows in dataset_results.items():
        row = {'dataset': dataset}
        for model in models:
            val = model_rows.get(model, {}).get('runtime_sec', '')
            try:
                row[model] = round(float(val), 1) if val != '' else ''
            except ValueError:
                row[model] = ''
        stats = dataset_stats.get(dataset, {})
        for col in STATS_COLS:
            row[col] = stats.get(col, '')
        table.append(row)
    table.sort(key=lambda r: int(r.get('n_tokens') or 0), reverse=True)
    return table


def average_runtime_row(table, models):
    avg = {'dataset': 'AVERAGE'}
    for model in models:
        vals = [float(r[model]) for r in table if r.get(model) not in ('', None)]
        avg[model] = round(sum(vals) / len(vals), 1) if vals else ''
    for col in STATS_COLS:
        avg[col] = ''
    return avg


def write_runtime_table(table, path, models):
    os.makedirs(path.parent, exist_ok=True)
    fieldnames = ['dataset'] + models + STATS_COLS
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
        writer.writerow(average_runtime_row(table, models))
    print(f"Saved → {path}")


def write_table(table, path, models, comparisons):
    os.makedirs(path.parent, exist_ok=True)
    imp_cols = [col for _, _, col in comparisons]
    fieldnames = ['dataset'] + models + imp_cols + STATS_COLS
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
        writer.writerow(average_row(table, models, comparisons))
    print(f"Saved → {path}")


def load_best_asym_mse():
    """Return {dataset: best_test_mse} from the Ks tuning wide summary."""
    best = {}
    if not KS_WIDE_PATH.exists():
        return best
    with open(KS_WIDE_PATH, newline='') as f:
        for row in csv.DictReader(f):
            ds = row['dataset']
            val = row.get('best_test_mse', '')
            if val:
                best[ds] = float(val)
    return best



def main():
    best_asym_mse = load_best_asym_mse()

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

    # Override JST-LFM-asym test_mse with best tuned Ks result where available.
    for ds, mse in best_asym_mse.items():
        if ds in dataset_results and 'JST-LFM-asym' in dataset_results[ds]:
            dataset_results[ds]['JST-LFM-asym']['test_mse'] = str(mse)

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
    write_table(
        build_table(dataset_results, 'test_mse', MODELS_ALL, COMPARISONS_ALL, dataset_stats),
        OUT_DIR / 'mse_all.csv',
        MODELS_ALL, COMPARISONS_ALL,
    )
    write_table(
        build_table(dataset_results, 'test_mse', MODELS_2SENT, COMPARISONS_2SENT, dataset_stats),
        OUT_DIR / 'mse_2sent_vs_3sent.csv',
        MODELS_2SENT, COMPARISONS_2SENT,
    )
    write_runtime_table(
        build_runtime_table(dataset_results, MODELS_RUNTIME, dataset_stats),
        OUT_DIR / 'runtime.csv',
        MODELS_RUNTIME,
    )


if __name__ == '__main__':
    main()
