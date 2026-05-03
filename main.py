"""
main.py

Loads and preprocesses Amazon review data, then runs all
recommender models. Results are collected and printed together in a
single summary table at the end. Per-model tuning grids and final
test metrics are written to CSV files under results/<dataset>/.

"""

import csv
import os
from collections import defaultdict
from pathlib import Path
import re
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_preprocessing import load_amazon, print_stats, split_data, filter_k_core, plot_distributions, save_dataset_stats
from LDAFirst import run_lda_first_tuned
from LFM import run_lfm_tuned
from LDA_LFM import run_lda_lfm_tuned, top_words_per_topic
from JSTFirst import run_jst_first_tuned
from JST_LFM import run_jst_lfm_tuned
from JST_LFM_asymmetric import run_jst_lfm_asym_tuned

# ---------------------------------------------------------------------------
# Config — add more paths to run over multiple datasets in one go
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
DATA_PATHS = [
    # DATA_DIR / 'reviews_Musical_Instruments_5.json.gz',
    # DATA_DIR / 'reviews_Amazon_Instant_Video_5.json.gz',
    # DATA_DIR / 'reviews_Digital_Music_5.json.gz',
    # DATA_DIR / 'reviews_Baby_5.json.gz',
    # DATA_DIR / 'reviews_Patio_Lawn_and_Garden_5.json.gz',
    # DATA_DIR / 'reviews_Pet_Supplies_5.json.gz', #-----assym starts here! 
    # DATA_DIR / 'reviews_Office_Products_5.json.gz',
    # DATA_DIR / 'reviews_Grocery_and_Gourmet_Food_5.json.gz',
    # DATA_DIR / 'reviews_Video_Games_5.json.gz',
    # DATA_DIR / 'reviews_Automotive_5.json.gz',
    # DATA_DIR / 'reviews_Tools_and_Home_Improvement_5.json.gz',
    # DATA_DIR / 'reviews_Beauty_5.json.gz',
    # DATA_DIR / 'reviews_Toys_and_Games_5.json.gz' #-----assym stops here!
    DATA_DIR / 'reviews_Apps_for_Android_5.json.gz', 
    DATA_DIR / 'reviews_Health_and_Personal_Care_5.json.gz',
    DATA_DIR / 'reviews_Kindle_Store_5.json.gz',
    DATA_DIR / 'reviews_Sports_and_Outdoors_5.json.gz',
    DATA_DIR / 'reviews_Cell_Phones_and_Accessories_5.json.gz',
    DATA_DIR / 'reviews_CDs_and_Vinyl_5.json.gz',
    DATA_DIR / 'reviews_Home_and_Kitchen_5.json.gz',
    DATA_DIR / 'reviews_Movies_and_TV_5.json.gz',
    DATA_DIR / 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
    DATA_DIR / 'reviews_Electronics_5.json.gz',
]

LEXICON_PATH = 'MPQA_Subjectivity_Lexicon.tff'
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dataset_name(path):
    """reviews_Beauty_5.json.gz  →  Beauty_5"""
    name = os.path.basename(path)
    name = re.sub(r'^reviews_', '', name)
    name = re.sub(r'\.json\.gz$', '', name)
    return name


def save_epoch_plot(tuning_rows, best_info, out_path, model_name):
    """Save a grid of subplots — one per hyperparameter combo — showing val MSE
    vs epoch. The best combo's subplot is highlighted with a red border and
    a dashed vertical line at its best epoch.
    """
    best_lr  = best_info['lr']
    best_reg = best_info['reg']
    best_ep  = best_info['epochs']
    best_mu  = best_info.get('mu', float('nan'))
    has_mu   = best_mu == best_mu  # False when best_mu is NaN

    # Group rows by hyperparameter combo, preserving insertion order
    combos = defaultdict(list)
    for row in tuning_rows:
        key = (row['lr'], row['reg'], row['mu']) if has_mu else (row['lr'], row['reg'])
        combos[key].append((row['n_epochs'], row['val_mse']))
    for key in combos:
        combos[key].sort()

    keys = sorted(combos.keys())
    n = len(keys)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    for ax, key in zip(axes, keys):
        history = combos[key]
        epochs = [ep  for ep, _   in history]
        mses   = [mse for _,  mse in history]

        if has_mu:
            lr, reg, mu = key
            is_best = (lr == best_lr and reg == best_reg and mu == best_mu)
            title = f'lr={lr}, reg={reg}, mu={mu}'
        else:
            lr, reg = key
            is_best = (lr == best_lr and reg == best_reg)
            title = f'lr={lr}, reg={reg}'

        color = 'red' if is_best else 'steelblue'
        ax.plot(epochs, mses, linewidth=1.2, color=color)
        if is_best:
            ax.axvline(best_ep, color='red', linestyle='--', linewidth=1,
                       label=f'best epoch={best_ep}')
            ax.legend(fontsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
            title += ' [BEST]'
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Epoch', fontsize=7)
        ax.set_ylabel('Val MSE', fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f'{model_name} — val MSE per hyperparameter combo', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def write_tuning_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lr', 'reg', 'mu', 'n_epochs', 'val_mse'])
        writer.writeheader()
        writer.writerows(rows)



_SENT_LABEL = {0: 'positive', 1: 'negative', 2: 'neutral'}


def write_jst_topics_csv(path, topic_words):
    """topic_words: {(k, l): [(word, prob), ...]} from top_words_per_topic_sentiment."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['topic', 'sentiment', 'rank', 'word', 'probability']
        )
        writer.writeheader()
        for (k, l), words in sorted(topic_words.items()):
            for rank, (word, prob) in enumerate(words, start=1):
                writer.writerow({
                    'topic': k, 'sentiment': _SENT_LABEL[l],
                    'rank': rank, 'word': word, 'probability': prob,
                })


def write_topics_csv(path, psi, dictionary, top_n=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    topics = top_words_per_topic(psi, dictionary, top_n=top_n)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['topic', 'rank', 'word', 'probability'])
        writer.writeheader()
        for k, words in topics.items():
            for rank, (word, prob) in enumerate(words, start=1):
                writer.writerow({'topic': k, 'rank': rank, 'word': word, 'probability': prob})


def write_final_csv(path, rows):
    """Upsert rows into the final results CSV keyed by model name.

    Existing rows whose model name matches a new row are overwritten in-place;
    models not yet in the file are appended. Rows for models not touched in
    this run are left untouched.
    """
    fieldnames = ['model', 'best_lr', 'best_reg', 'best_mu', 'best_epochs',
                  'test_mse', 'test_mae', 'runtime_sec']

    # Load existing rows, preserving their order
    existing = {}
    if os.path.exists(path):
        with open(path, 'r', newline='') as f:
            for row in csv.DictReader(f):
                existing[row['model']] = row

    # Upsert: overwrite matching model rows, append new ones
    for row in rows:
        existing[row['model']] = row

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing.values())


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE': float(np.mean(errors ** 2)),
        'MAE': float(np.mean(np.abs(errors))),
    }



# ---------------------------------------------------------------------------
# Model 1: Offset Model (predict global mean for all)
# ---------------------------------------------------------------------------
def offset_model(train, test):
    mu = train['overall'].mean()
    predictions = np.full(len(test), mu)
    res = evaluate(predictions, test['overall'].values)
    res['test_pred'] = predictions
    return res


# ---------------------------------------------------------------------------
# Model 2: Baseline Rating Model (global mean + user bias + item bias)
# ---------------------------------------------------------------------------
def baseline_rating_model(train, test):
    mu = train['overall'].mean()
    user_bias = train.groupby('user_idx')['overall'].mean() - mu
    item_bias = train.groupby('item_idx')['overall'].mean() - mu
    test_user_bias = test['user_idx'].map(user_bias).fillna(0).values
    test_item_bias = test['item_idx'].map(item_bias).fillna(0).values
    predictions = mu + test_user_bias + test_item_bias
    res = evaluate(predictions, test['overall'].values)
    res['test_pred'] = predictions
    return res


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------
def run_pipeline(data_path, k_core=5):
    name = dataset_name(data_path)
    out_dir = os.path.join('results', name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {name} ({k_core}-core)")
    print(f"{'='*60}")

    data = load_amazon(data_path)
    print("Pre-filter stats:")
    print_stats(data)
    plot_distributions(data, dataset_name=name, out_dir=out_dir)
    save_dataset_stats(data, dataset_name=name, out_dir=out_dir)

    # data = filter_k_core(data, k=k_core)
    # if len(data) == 0:
    #     print(f"\n[SKIPPED] {name}: dataset empty after {k_core}-core filtering")
    #     return
    # print("\nPost-filter stats:")
    # print_stats(data)

    train, valid, test, uid2idx, sid2idx = split_data(data, seed=SEED)
    final_rows = []

    # -- Offset Model -------------------------------------------------------
    t = time.time()
    res = offset_model(train, test)
    res['Time'] = time.time() - t
    print(f"Offset Model: MSE={res['MSE']:.4f}  MAE={res['MAE']:.4f}  "
          f"Time={res['Time']:.2f}s")
    final_rows.append({
        'model': 'Offset', 'best_lr': '', 'best_reg': '', 'best_mu': '',
        'best_epochs': '',
        'test_mse': res['MSE'], 'test_mae': res['MAE'],
        'runtime_sec': res['Time'],
    })

    # -- Baseline Rating ----------------------------------------------------
    t = time.time()
    res = baseline_rating_model(train, test)
    res['Time'] = time.time() - t
    print(f"Baseline Rating: MSE={res['MSE']:.4f}  MAE={res['MAE']:.4f}  "
          f"Time={res['Time']:.2f}s")
    final_rows.append({
        'model': 'Baseline', 'best_lr': '', 'best_reg': '', 'best_mu': '',
        'best_epochs': '',
        'test_mse': res['MSE'], 'test_mae': res['MAE'],
        'runtime_sec': res['Time'],
    })

    # -- LFM ----------------------------------------------------------------
    t = time.time()
    res, best, tuning, epoch_mse = run_lfm_tuned(train, valid, test, uid2idx, sid2idx, n_factors=15)
    elapsed = time.time() - t
    print(f"LFM: MSE={res['MSE']:.4f}  MAE={res['MAE']:.4f}  Time={elapsed:.2f}s")
    write_tuning_csv(os.path.join(out_dir, 'LFM_tuning.csv'), tuning)
    save_epoch_plot(tuning, best, os.path.join(out_dir, 'LFM_epoch_mse.png'), 'LFM')
    final_rows.append({
        'model': 'LFM',
        'best_lr': best['lr'], 'best_reg': best['reg'], 'best_mu': best['mu'],
        'best_epochs': best['epochs'],
        'test_mse': res['MSE'], 'test_mae': res['MAE'],
        'runtime_sec': elapsed,
    })

    # -- LDAFirst -----------------------------------------------------------
    t = time.time()
    res, best, tuning, epoch_mse = run_lda_first_tuned(
        train, valid, test, uid2idx, sid2idx, n_topics=15)
    elapsed = time.time() - t
    print(f"LDAFirst: MSE={res['MSE']:.4f}  MAE={res['MAE']:.4f}  Time={elapsed:.2f}s")
    write_tuning_csv(os.path.join(out_dir, 'LDAFirst_tuning.csv'), tuning)
    save_epoch_plot(tuning, best, os.path.join(out_dir, 'LDAFirst_epoch_mse.png'), 'LDAFirst')
    final_rows.append({
        'model': 'LDAFirst',
        'best_lr': best['lr'], 'best_reg': best['reg'], 'best_mu': best['mu'],
        'best_epochs': best['epochs'],
        'test_mse': res['MSE'], 'test_mae': res['MAE'],
        'runtime_sec': elapsed,
    })

    # -- LDA-LFM ------------------------------------------------------------
    t = time.time()
    res, best, tuning, epoch_mse, psi, _topic_words, dictionary = run_lda_lfm_tuned(
        train, valid, test, uid2idx, sid2idx, n_topics=15
    )
    elapsed = time.time() - t
    print(f"LDA-LFM: MSE={res['MSE']:.4f}  MAE={res['MAE']:.4f}  Time={elapsed:.2f}s")
    write_tuning_csv(os.path.join(out_dir, 'LDA_LFM_tuning.csv'), tuning)
    write_topics_csv(os.path.join(out_dir, 'LDA_LFM_topics.csv'), psi, dictionary)
    save_epoch_plot(tuning, best, os.path.join(out_dir, 'LDA_LFM_epoch_mse.png'), 'LDA-LFM')
    final_rows.append({
        'model': 'LDA-LFM',
        'best_lr': best['lr'], 'best_reg': best['reg'], 'best_mu': best['mu'],
        'best_epochs': best['epochs'],
        'test_mse': res['MSE'], 'test_mae': res['MAE'],
        'runtime_sec': elapsed,
    })

    # -- JSTFirst -----------------------------------------------------------
    t = time.time()
    res, best, tuning, epoch_mse, topic_words = run_jst_first_tuned(
        train, valid, test, uid2idx, sid2idx,
        lexicon_path=LEXICON_PATH, K=5,
    )
    elapsed = time.time() - t
    print(f"JSTFirst: MSE={res['MSE']:.4f}  MAE={res['MAE']:.4f}  Time={elapsed:.2f}s")
    write_tuning_csv(os.path.join(out_dir, 'JSTFirst_tuning.csv'), tuning)
    write_jst_topics_csv(os.path.join(out_dir, 'JSTFirst_topics.csv'), topic_words)
    save_epoch_plot(tuning, best, os.path.join(out_dir, 'JSTFirst_epoch_mse.png'), 'JSTFirst')
    final_rows.append({
        'model': 'JSTFirst',
        'best_lr': best['lr'], 'best_reg': best['reg'], 'best_mu': best['mu'],
        'best_epochs': best['epochs'],
        'test_mse': res['MSE'], 'test_mae': res['MAE'],
        'runtime_sec': elapsed,
    })

    # -- JST-LFM ------------------------------------------------------------
    t = time.time()
    res, best, tuning, epoch_mse, topic_words, dictionary = run_jst_lfm_tuned(
        train, valid, test, uid2idx, sid2idx,
        lexicon_path=LEXICON_PATH, K=5,
    )
    elapsed = time.time() - t
    print(f"JST-LFM: MSE={res['MSE']:.4f}  MAE={res['MAE']:.4f}  Time={elapsed:.2f}s")
    write_tuning_csv(os.path.join(out_dir, 'JST_LFM_tuning.csv'), tuning)
    write_jst_topics_csv(os.path.join(out_dir, 'JST_LFM_topics.csv'), topic_words)
    save_epoch_plot(tuning, best, os.path.join(out_dir, 'JST_LFM_epoch_mse.png'), 'JST-LFM')
    final_rows.append({
        'model': 'JST-LFM',
        'best_lr': best['lr'], 'best_reg': best['reg'], 'best_mu': best['mu'],
        'best_epochs': best['epochs'],
        'test_mse': res['MSE'], 'test_mae': res['MAE'],
        'runtime_sec': elapsed,
    })

    # -- JST-LFM asymmetric ---------------------------------------------------
    t = time.time()
    res, best, tuning, epoch_mse, topic_words, dictionary = run_jst_lfm_asym_tuned(
        train, valid, test, uid2idx, sid2idx,
        lexicon_path=LEXICON_PATH, Ks=(9, 3, 3),
    )
    elapsed = time.time() - t
    print(f"JST-LFM-asym: MSE={res['MSE']:.4f}  MAE={res['MAE']:.4f}  Time={elapsed:.2f}s")
    write_tuning_csv(os.path.join(out_dir, 'JST_LFM_asym_tuning.csv'), tuning)
    write_jst_topics_csv(os.path.join(out_dir, 'JST_LFM_asym_topics.csv'), topic_words)
    save_epoch_plot(tuning, best, os.path.join(out_dir, 'JST_LFM_asym_epoch_mse.png'), 'JST-LFM-asym')
    final_rows.append({
        'model': 'JST-LFM-asym',
        'best_lr': best['lr'], 'best_reg': best['reg'], 'best_mu': best['mu'],
        'best_epochs': best['epochs'],
        'test_mse': res['MSE'], 'test_mae': res['MAE'],
        'runtime_sec': elapsed,
    })


    # -- Write final CSV ----------------------------------------------------
    write_final_csv(os.path.join(out_dir, 'final_results.csv'), final_rows)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    for path in DATA_PATHS:
        run_pipeline(path)