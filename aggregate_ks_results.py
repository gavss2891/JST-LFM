"""
aggregate_ks_results.py

Reads existing JST_LFM_asym_Ks_*_result.csv files from results_tune_ks/
and produces two summary CSVs:

  Ks_summary_long.csv  – one row per (dataset, Ks): test_mse + hyperparams
  Ks_summary_wide.csv  – one row per dataset, one column per Ks config
                         (5,5,5) = JST-LFM symmetric included as first config
"""

import csv
from pathlib import Path

RESULTS_DIR  = Path('/Users/gavinshao/Desktop/Master Thesis/Code/results_tune_ks')
SUMMARY_PATH = Path('/Users/gavinshao/Desktop/Master Thesis/Code/results/summary/mse_overall.csv')
MSE_ALL_PATH = Path('/Users/gavinshao/Desktop/Master Thesis/Code/results/summary/mse_all.csv')

KS_CONFIGS = [
    (9, 1, 5),
    (9, 5, 1),
    (11, 2, 2),
    (7, 4, 4),
    (7, 2, 6),
    (7, 6, 2),
]

def ks_tag(Ks):
    return '_'.join(str(k) for k in Ks)

def read_result(result_path):
    """Read test_mse and hyperparams from a _result.csv file."""
    with open(result_path, newline='') as f:
        row = next(csv.DictReader(f))
    return float(row['test_mse']), float(row['test_mae']), row

# ── load n_tokens ordering and (5,5,5) baseline from existing summaries ───────
tokens_map = {}
with open(SUMMARY_PATH, newline='') as f:
    for row in csv.DictReader(f):
        if row['dataset'] != 'AVERAGE':
            tokens_map[row['dataset']] = int(float(row['n_tokens']))

def strip_stars(v):
    return float(str(v).replace('***', ''))

asym_555_map = {}  # dataset -> test_mse of JST-LFM (5,5,5) symmetric
asym_933_map = {}  # dataset -> test_mse of JST-LFM-asym (9,3,3)
with open(MSE_ALL_PATH, newline='') as f:
    for row in csv.DictReader(f):
        if row['dataset'] != 'AVERAGE':
            asym_555_map[row['dataset']] = strip_stars(row['JST-LFM'])
            asym_933_map[row['dataset']] = strip_stars(row['JST-LFM-asym'])

# ── collect results ──────────────────────────────────────────────────────────
datasets_unsorted = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
datasets = sorted(datasets_unsorted, key=lambda d: tokens_map.get(d, 0), reverse=True)

long_rows = []
wide_data = {}   # dataset -> {ks_label -> {mse}}

for ds in datasets:
    wide_data[ds] = {}
    for Ks in KS_CONFIGS:
        tag         = ks_tag(Ks)
        result_path = RESULTS_DIR / ds / f'JST_LFM_asym_Ks_{tag}_result.csv'

        if result_path.exists():
            test_mse, test_mae, row = read_result(result_path)
            lbl = str(Ks)
            long_rows.append({
                'dataset':   ds,
                'Ks':        lbl,
                'K_pos':     Ks[0],
                'K_neg':     Ks[1],
                'K_neu':     Ks[2],
                'D_total':   sum(Ks),
                'test_mse':  round(test_mse, 6),
                'test_mae':  round(test_mae, 6),
                'best_lr':   row['best_lr'],
                'best_reg':  row['best_reg'],
                'best_mu':   row['best_mu'],
                'best_epochs': row['best_epochs'],
            })
            wide_data[ds][str(Ks)] = {'mse': round(test_mse, 6)}

# ── write long-form ──────────────────────────────────────────────────────────
long_path = RESULTS_DIR / 'Ks_summary_long.csv'
long_fields = ['dataset', 'Ks', 'K_pos', 'K_neg', 'K_neu', 'D_total',
               'test_mse', 'test_mae', 'best_lr', 'best_reg', 'best_mu', 'best_epochs']
with open(long_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=long_fields)
    w.writeheader()
    w.writerows(long_rows)
print(f"Saved: {long_path}")

# ── write wide-form ──────────────────────────────────────────────────────────
SYM_LBL      = '(5,5,5)'
ASYM_LBL     = '(9,3,3)'
ks_labels    = [str(Ks) for Ks in KS_CONFIGS]
all_labels   = [SYM_LBL, ASYM_LBL] + ks_labels  # baselines first, then tuned configs
mse_cols     = [f'mse_{lbl}' for lbl in all_labels]
wide_fields  = ['dataset', 'n_tokens'] + mse_cols + ['best_Ks', 'best_test_mse',
                'imp_vs_(5,5,5) (%)']

wide_rows = []
for ds in datasets:
    row = {'dataset': ds, 'n_tokens': tokens_map.get(ds, '')}

    # build candidate pool including both baselines
    mse_555 = asym_555_map.get(ds)
    mse_933 = asym_933_map.get(ds)
    candidates = {}
    if mse_555:
        candidates[SYM_LBL] = round(mse_555, 6)
    if mse_933:
        candidates[ASYM_LBL] = round(mse_933, 6)
    for lbl in ks_labels:
        info = wide_data[ds].get(lbl)
        if info:
            candidates[lbl] = info['mse']

    # find overall best
    best_lbl = min(candidates, key=candidates.get) if candidates else None
    best_mse = candidates[best_lbl] if best_lbl else float('inf')

    # fill mse columns with *** on winner
    row[f'mse_{SYM_LBL}'] = (f'{round(mse_555, 6)}***' if best_lbl == SYM_LBL
                              else round(mse_555, 6)) if mse_555 else ''
    row[f'mse_{ASYM_LBL}'] = (f'{round(mse_933, 6)}***' if best_lbl == ASYM_LBL
                               else round(mse_933, 6)) if mse_933 else ''
    for lbl in ks_labels:
        info = wide_data[ds].get(lbl)
        if info:
            val = info['mse']
            row[f'mse_{lbl}'] = f'{val}***' if lbl == best_lbl else val
        else:
            row[f'mse_{lbl}'] = ''

    row['best_Ks']       = best_lbl or ''
    row['best_test_mse'] = round(best_mse, 6) if best_lbl else ''

    # imp vs (5,5,5) — null if (5,5,5) is the best
    if mse_555 and best_lbl and best_lbl != SYM_LBL:
        row['imp_vs_(5,5,5) (%)'] = round((mse_555 - best_mse) / mse_555 * 100, 2)
    else:
        row['imp_vs_(5,5,5) (%)'] = ''

    wide_rows.append(row)

wide_path = RESULTS_DIR / 'Ks_summary_wide.csv'
with open(wide_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=wide_fields)
    w.writeheader()
    w.writerows(wide_rows)
print(f"Saved: {wide_path}")

# ── val-selected summary: pick best Ks by val MSE, report its test MSE ───────
def best_val_mse_for_ks(ds, Ks):
    """Return the minimum val MSE across all combos in the tuning CSV, or None."""
    tag         = ks_tag(Ks)
    tuning_path = RESULTS_DIR / ds / f'JST_LFM_asym_Ks_{tag}_tuning.csv'
    if not tuning_path.exists():
        return None
    with open(tuning_path, newline='') as f:
        vals = [float(r['val_mse']) for r in csv.DictReader(f)]
    return min(vals) if vals else None

val_sel_fields = ['dataset', 'n_tokens',
                  'best_Ks', 'best_val_mse', 'test_mse',
                  f'test_mse_{SYM_LBL}', 'imp_vs_(5,5,5)_(%)']
val_sel_rows = []

for ds in datasets:
    # collect (val_mse, test_mse, ks_label) for every available Ks config
    candidates = []
    for Ks in KS_CONFIGS:
        val  = best_val_mse_for_ks(ds, Ks)
        lbl  = str(Ks)
        info = wide_data[ds].get(lbl)
        if val is not None and info:
            candidates.append((val, info['mse'], lbl))

    mse_555 = asym_555_map.get(ds)

    if candidates:
        best_val, best_test, best_lbl = min(candidates, key=lambda x: x[0])
    else:
        best_val = best_test = best_lbl = None

    imp = ''
    if mse_555 and best_test is not None:
        imp = round((mse_555 - best_test) / mse_555 * 100, 2)

    val_sel_rows.append({
        'dataset':            ds,
        'n_tokens':           tokens_map.get(ds, ''),
        'best_Ks':            best_lbl or '',
        'best_val_mse':       round(best_val,  6) if best_val  is not None else '',
        'test_mse':           round(best_test, 6) if best_test is not None else '',
        f'test_mse_{SYM_LBL}': round(mse_555, 6) if mse_555 else '',
        'imp_vs_(5,5,5)_(%)': imp,
    })

val_sel_path = RESULTS_DIR / 'Ks_summary_val_selected.csv'
with open(val_sel_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=val_sel_fields)
    w.writeheader()
    w.writerows(val_sel_rows)
print(f"Saved: {val_sel_path}")

print(f"\n{'dataset':<35}  {'best_Ks':<12}  {'best_val_mse':<14}  {'test_mse':<10}  {'mse_555':<10}  imp_555%")
for r in val_sel_rows:
    print(f"{r['dataset']:<35}  {str(r['best_Ks']):<12}  "
          f"{str(r['best_val_mse']):<14}  {str(r['test_mse']):<10}  "
          f"{str(r[f'test_mse_{SYM_LBL}']):<10}  {r['imp_vs_(5,5,5)_(%)']}")

# ── print table ──────────────────────────────────────────────────────────────
print(f"\n{'dataset':<22}", end='')
for lbl in all_labels:
    print(f"  {lbl:<12}", end='')
print(f"  {'best_Ks':<12}  {'best_test_mse':<12}  imp_555%")
for r in wide_rows:
    print(f"{r['dataset']:<22}", end='')
    for lbl in all_labels:
        val = r.get(f'mse_{lbl}', '')
        print(f"  {str(val):<13}", end='')
    print(f"  {r['best_Ks']:<12}  {r['best_test_mse']:<12}  {r['imp_vs_(5,5,5) (%)']}")
