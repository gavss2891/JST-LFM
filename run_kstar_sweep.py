"""
run_kstar_sweep.py

K* sweep for LDA-LFM (imported) and JST-LFM-asym-2sent (2-sentiment).

For each dataset:
  1. Import existing LDA-LFM k* results from results/<dataset>/kstar_sweep.csv
     (produced by the previous S=3 run — no re-computation needed).
  2. Read the fixed (K_pos, K_neg) for JST-LFM-asym-2sent from
     results/summary/ks_tuned_selection.csv (selected by validation MSE).
  3. Read the best (lr, reg, mu) for JST-LFM-asym-2sent-Ks-tuned from
     results/<dataset>/final_results.csv.
  4. Build the JST corpus once; run the k* sweep at K* in K_STAR_VALUES with
     those fixed hyperparameters.
  5. Write combined results to results/<dataset>/kstar_sweep_2sent.csv and
     a cross-dataset summary to results/kstar_summary_2sent.csv.

K* = 0 reproduces the base model exactly (reference point).
Hyperparameters are held fixed across K* values so that any change in
test MSE reflects the effect of the extra features, not re-tuning.
"""

import csv
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from data_preprocessing import load_amazon_gz, split_data, clean
from JST_LFM_asymmetric import run_jst_lfm_asym_kstar
from JST_LFM_asymmetric import build_corpus as jst_build_corpus
from JST_LFM_asymmetric import load_mpqa_lexicon

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR      = Path('/Users/gavinshao/Desktop/Master Thesis/Code/Data')
RESULTS_DIR   = Path('/Users/gavinshao/Desktop/Master Thesis/Code/results')
KS_SELECTION  = RESULTS_DIR / 'summary' / 'ks_tuned_selection.csv'
LEXICON_PATH  = 'MPQA_Subjectivity_Lexicon.tff'
SEED          = 42
GAMMA_2SENT   = (0.1, 1)     # one value per sentiment (pos, neg)

DATA_PATHS = [
    DATA_DIR / 'reviews_Musical_Instruments_5.json.gz',
    DATA_DIR / 'reviews_Automotive_5.json.gz',
    DATA_DIR / 'reviews_Patio_Lawn_and_Garden_5.json.gz',
    DATA_DIR / 'reviews_Amazon_Instant_Video_5.json.gz',
    DATA_DIR / 'reviews_Office_Products_5.json.gz',
    DATA_DIR / 'reviews_Digital_Music_5.json.gz',
    DATA_DIR / 'reviews_Pet_Supplies_5.json.gz',
    DATA_DIR / 'reviews_Baby_5.json.gz',
    DATA_DIR / 'reviews_Grocery_and_Gourmet_Food_5.json.gz',
    DATA_DIR / 'reviews_Tools_and_Home_Improvement_5.json.gz',
    DATA_DIR / 'reviews_Toys_and_Games_5.json.gz',
    DATA_DIR / 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
    DATA_DIR / 'reviews_Beauty_5.json.gz',
    DATA_DIR / 'reviews_Cell_Phones_and_Accessories_5.json.gz',
    DATA_DIR / 'reviews_Sports_and_Outdoors_5.json.gz',
    DATA_DIR / 'reviews_Apps_for_Android_5.json.gz',
    DATA_DIR / 'reviews_Health_and_Personal_Care_5.json.gz',
    DATA_DIR / 'reviews_Video_Games_5.json.gz',
    DATA_DIR / 'reviews_Home_and_Kitchen_5.json.gz',
    DATA_DIR / 'reviews_Kindle_Store_5.json.gz',
    DATA_DIR / 'reviews_CDs_and_Vinyl_5.json.gz',
    DATA_DIR / 'reviews_Electronics_5.json.gz',
    DATA_DIR / 'reviews_Movies_and_TV_5.json.gz',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dataset_name(path):
    name = Path(path).name
    return name.replace('reviews_', '').replace('.json.gz', '')


def load_best_ks_map(selection_path):
    """Return {dataset_name: (K_pos, K_neg)} from ks_tuned_selection.csv."""
    best_ks = {}
    if not os.path.exists(selection_path):
        return best_ks
    with open(selection_path, newline='') as f:
        for row in csv.DictReader(f):
            best_ks[row['dataset']] = (int(row['K_pos']), int(row['K_neg']))
    return best_ks


def read_lda_kstar_rows(sweep_csv):
    """Import existing LDA-LFM k* rows from a previously generated kstar_sweep.csv.

    Returns (rows, k_star_values) where k_star_values is the sorted list of
    k* values present in those rows — used to run the same sweep for JST.
    """
    rows = []
    if not os.path.exists(sweep_csv):
        return rows, []
    with open(sweep_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row['model'] == 'LDA-LFM':
                rows.append({
                    'dataset':     row['dataset'],
                    'model':       'LDA-LFM',
                    'k_star':      int(row['k_star']),
                    'test_mse':    float(row['test_mse']),
                    'test_mae':    float(row['test_mae']),
                    'val_mse':     float(row['val_mse']),
                    'best_epochs': int(row['best_epochs']),
                    'lr':          float(row['lr']),
                    'reg':         float(row['reg']),
                    'mu':          float(row['mu']),
                    'runtime_sec': float(row['runtime_sec']),
                })
    k_star_values = sorted({r['k_star'] for r in rows})
    return rows, k_star_values


def read_best_hparams(final_csv, model_name):
    """Read (lr, reg, mu) for model_name from final_results.csv."""
    if not os.path.exists(final_csv):
        raise FileNotFoundError(
            f"Missing {final_csv}. Run main.py first to produce it."
        )
    with open(final_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row['model'] == model_name:
                return float(row['best_lr']), float(row['best_reg']), float(row['best_mu'])
    raise ValueError(f"Model '{model_name}' not found in {final_csv}")


def read_jst_kstar0_row(final_csv, dataset_name):
    """Import the k*=0 JST-LFM-asym-2sent result from final_results.csv.

    K*=0 is identical to the base Ks-tuned run, so we reuse that result
    directly instead of re-running the model.
    """
    if not os.path.exists(final_csv):
        return None
    with open(final_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row['model'] == 'JST-LFM-asym-2sent-Ks-tuned':
                return {
                    'dataset':     dataset_name,
                    'model':       'JST-LFM-asym-2sent',
                    'k_star':      0,
                    'test_mse':    float(row['test_mse']),
                    'test_mae':    float(row['test_mae']),
                    'val_mse':     '',
                    'best_epochs': int(row['best_epochs']),
                    'lr':          float(row['best_lr']),
                    'reg':         float(row['best_reg']),
                    'mu':          float(row['best_mu']),
                    'runtime_sec': float(row['runtime_sec']),
                }
    return None


def write_sweep_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ['dataset', 'model', 'k_star', 'test_mse', 'test_mae',
                  'val_mse', 'best_epochs', 'lr', 'reg', 'mu', 'runtime_sec']
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Parallel worker helper (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _init_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _run_jst_kstar_job(args):
    (train, valid, test, uid2idx, sid2idx,
     lr, reg, mu_corpus, best_ks, k_star,
     jst_corpus, jst_lexicon, jst_dict) = args
    t0 = time.time()
    results, best_info, _, _ = run_jst_lfm_asym_kstar(
        train, valid, test, uid2idx, sid2idx,
        lr=lr, reg=reg, mu_corpus=mu_corpus,
        Ks=best_ks, k_star=k_star,
        gamma=GAMMA_2SENT,
        corpus=jst_corpus, lexicon=jst_lexicon, dictionary=jst_dict,
        verbose=False,
    )
    return k_star, results, best_info, round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def run_dataset(data_path, best_ks_map):
    name     = dataset_name(data_path)
    out_dir  = RESULTS_DIR / name
    final_csv   = out_dir / 'final_results.csv'
    old_sweep   = out_dir / 'kstar_sweep.csv'
    sweep_csv   = out_dir / 'kstar_sweep_2sent.csv'

    print(f"\n{'='*65}")
    print(f"Dataset: {name}")
    print(f"{'='*65}")

    data = load_amazon_gz(str(data_path))
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=SEED)

    dataset_rows = []

    # ------------------------------------------------------------------ LDA-LFM
    lda_rows, k_star_values = read_lda_kstar_rows(str(old_sweep))
    if lda_rows:
        print(f"\n[LDA-LFM] imported {len(lda_rows)} k* rows from {old_sweep} "
              f"(k*={k_star_values})")
        dataset_rows.extend(lda_rows)
        best_lda = min(lda_rows, key=lambda r: r['test_mse'])
        print(f"  best K*={best_lda['k_star']}  test MSE={best_lda['test_mse']:.4f}")
    else:
        print(f"\n[LDA-LFM] WARNING: no existing rows found in {old_sweep} — skipping")
        k_star_values = []

    # -------------------------------------------------------- JST-LFM-asym-2sent
    if name not in best_ks_map:
        print(f"\n[JST-LFM-asym-2sent] WARNING: {name} not in ks_tuned_selection — skipping")
        return dataset_rows
    best_ks = best_ks_map[name]
    lr_jst, reg_jst, mu_jst = read_best_hparams(
        str(final_csv), 'JST-LFM-asym-2sent-Ks-tuned'
    )
    run_k_star_values = [k for k in k_star_values if k != 0]
    print(f"\n[JST-LFM-asym-2sent] hparams: lr={lr_jst}, reg={reg_jst}, mu={mu_jst}, "
          f"Ks={best_ks}, k*={k_star_values} (k*=0 imported, running {run_k_star_values})")

    # k*=0 — reuse the result already stored in final_results.csv
    if 0 in k_star_values:
        kstar0_row = read_jst_kstar0_row(str(final_csv), name)
        if kstar0_row:
            dataset_rows.append(kstar0_row)
            print(f"  k*=0 imported: test MSE={kstar0_row['test_mse']:.4f}")
        else:
            print("  WARNING: could not import k*=0 from final_results.csv")

    if not run_k_star_values:
        write_sweep_csv(str(sweep_csv), dataset_rows)
        print(f"  Saved {sweep_csv}")
        return dataset_rows

    print("  Cleaning training reviews...", flush=True)
    with Pool(cpu_count()) as p:
        train['tokens'] = p.map(clean, train['reviewText'])

    print("  Building JST corpus (once)...", flush=True)
    _, jst_dict, all_words, all_docs, n_d, seen = \
        jst_build_corpus(train, sid2idx, n_vocab=5000)
    jst_corpus  = (all_words, all_docs, n_d, seen)
    jst_lexicon = load_mpqa_lexicon(LEXICON_PATH, jst_dict, min_freq=20)

    jst_jobs = [
        (train, valid, test, uid2idx, sid2idx,
         lr_jst, reg_jst, mu_jst, best_ks, k_star,
         jst_corpus, jst_lexicon, jst_dict)
        for k_star in run_k_star_values
    ]
    n_workers = min(len(run_k_star_values), cpu_count())
    with Pool(n_workers, initializer=_init_worker) as pool:
        jst_job_results = pool.map(_run_jst_kstar_job, jst_jobs)

    for k_star, results, best_info, elapsed in sorted(jst_job_results, key=lambda x: x[0]):
        dataset_rows.append({
            'dataset':     name,
            'model':       'JST-LFM-asym-2sent',
            'k_star':      k_star,
            'test_mse':    results['MSE'],
            'test_mae':    results['MAE'],
            'val_mse':     best_info['val_mse'],
            'best_epochs': best_info['epochs'],
            'lr':          lr_jst,
            'reg':         reg_jst,
            'mu':          mu_jst,
            'runtime_sec': elapsed,
        })

    jst_rows = [r for r in dataset_rows if r['model'] == 'JST-LFM-asym-2sent']
    best_jst = min(jst_rows, key=lambda r: r['test_mse'])
    print(f"  [JST-LFM-asym-2sent] best K*={best_jst['k_star']}  "
          f"test MSE={best_jst['test_mse']:.4f}")

    write_sweep_csv(str(sweep_csv), dataset_rows)
    print(f"  Saved {sweep_csv}")

    return dataset_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    best_ks_map = load_best_ks_map(KS_SELECTION)
    if best_ks_map:
        print(f"Loaded best (K_pos, K_neg) for {len(best_ks_map)} datasets "
              f"from {KS_SELECTION}")
    else:
        print(f"WARNING: no Ks selection found at {KS_SELECTION} — datasets will be skipped")

    all_rows = []
    for path in DATA_PATHS:
        rows = run_dataset(path, best_ks_map)
        all_rows.extend(rows)

    # Cross-dataset summary
    summary_path = RESULTS_DIR / 'kstar_summary_2sent.csv'
    fieldnames = ['dataset', 'model', 'k_star', 'test_mse', 'test_mae',
                  'val_mse', 'best_epochs', 'lr', 'reg', 'mu', 'runtime_sec']
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCross-dataset summary saved to {summary_path}")

    # Print best K* table
    by_ds_model = defaultdict(list)
    for r in all_rows:
        by_ds_model[(r['dataset'], r['model'])].append(r)

    print(f"\n{'dataset':<35} {'model':<22} {'best K*':<10} test MSE")
    print('-' * 76)
    for (ds, model), rows in sorted(by_ds_model.items()):
        best = min(rows, key=lambda r: r['test_mse'])
        print(f"{ds:<35} {model:<22} K*={best['k_star']:<7} {best['test_mse']:.4f}")
