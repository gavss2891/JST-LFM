"""
main.py

Loads and preprocesses Amazon review data, then runs all
recommender models. Results are collected and printed together in a
single summary table at the end.

Requires Python <= 3.12 and NumPy < 2.0 for scikit-surprise compatibility.
Install: pip install scikit-surprise pandas "numpy<2" nltk
"""

import time
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy

from data_preprocessing import load_amazon_gz, print_stats, split_data
from LDAFirst import run_lda_first_tuned
from LFM import run_lfm_tuned
from LDA_LFM import run_lda_lfm_tuned

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'
SEED = 42

# ---------------------------------------------------------------------------
# Load and split
# ---------------------------------------------------------------------------
data = load_amazon_gz(DATA_PATH)
print_stats(data)

train, valid, test, uid2idx, sid2idx = split_data(data, seed=SEED)

n_users = len(uid2idx)
n_items = len(sid2idx)

# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE':  np.mean(errors ** 2),
        'MAE':  np.mean(np.abs(errors)),
    }

# ---------------------------------------------------------------------------
# Model 1: Offset Model (predict global mean for all)
# ---------------------------------------------------------------------------
def offset_model(train, test):
    mu = train['overall'].mean()
    predictions = np.full(len(test), mu)
    return evaluate(predictions, test['overall'].values)

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
    return evaluate(predictions, test['overall'].values)

# ---------------------------------------------------------------------------
# Run all models and collect results
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    all_results = {}

    t = time.time()
    all_results['Offset Model'] = offset_model(train, test)
    all_results['Offset Model']['Time'] = time.time() - t
    print(f"Offset Model Completed: Time={all_results['Offset Model']['Time']:.2f}s")

    t = time.time()
    all_results['Baseline Rating'] = baseline_rating_model(train, test)
    all_results['Baseline Rating']['Time'] = time.time() - t
    print(f"Baseline Rating Completed: Time={all_results['Baseline Rating']['Time']:.2f}s")

    t = time.time()
    all_results['LFM'] = run_lfm_tuned(train, valid, test, uid2idx, sid2idx, n_factors=10)
    all_results['LFM']['Time'] = time.time() - t
    print(f"LFM Completed: Time={all_results['LFM']['Time']:.2f}s")
    
    t = time.time()
    lda_results, _, _ = run_lda_first_tuned(train, valid, test, uid2idx, sid2idx)
    all_results['LDAFirst'] = lda_results
    all_results['LDAFirst']['Time'] = time.time() - t
    print(f"LDAFirst Completed: Time={all_results['LDAFirst']['Time']:.2f}s")

    t = time.time()
    lda_lfm_results, _, _ = run_lda_lfm_tuned(train, valid, test, uid2idx, sid2idx)
    all_results['LDA-LFM'] = lda_lfm_results
    all_results['LDA-LFM']['Time'] = time.time() - t
    print(f"LDA-LFM Completed: Time={all_results['LDA-LFM']['Time']:.2f}s")

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    metrics = ['MSE', 'MAE', 'Time']
    col_w = 12
    name_w = 20

    header = f"{'Model':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("MODEL RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    for name, res in all_results.items():
        row = f"{name:<{name_w}}"
        for m in ['MSE', 'MAE']:
            row += f"{res[m]:>{col_w}.4f}"
        row += f"{res['Time']:>{col_w}.2f}s"
        print(row)
    print("=" * len(header))
    print(f"\nGlobal average rating: {train['overall'].mean():.4f}")
    print(f"Users: {n_users:,}  |  Items: {n_items:,}")

