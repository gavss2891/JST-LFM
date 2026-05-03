"""
1_data_preprocessing.py

Loads the Amazon 5-core review data, cleans review text, splits into
train/valid/test sets, and prepares documents for topic modelling.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import csv
import gzip
import json
import string

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOP_WORDS = set(stopwords.words("english"))
PUNCTUATION = set(string.punctuation)
LEMMATIZER = WordNetLemmatizer()
EXTRA_REMOVE = {"'s", "'re", "'d", "n't", "'ve", "ca", "it.i", '--', '...', 'mr.', "''", '``'}

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_amazon_gz(path):
    """Read a gzipped JSON-lines file into a DataFrame."""
    def _parse():
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)
    df = pd.DataFrame(_parse())
    df = df[['reviewerID', 'asin', 'overall', 'reviewText']].copy()
    return df

def load_amazon_2023(path):
    """
    Read a 2023 Amazon Reviews JSONL file (gzipped or plain) and return
    a DataFrame with columns ['reviewerID', 'asin', 'overall', 'reviewText']
    matching the schema produced by load_amazon_gz for 2014 data.
    """
    opener = gzip.open if str(path).endswith('.gz') else open
 
    rows = []
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                'reviewerID': obj['user_id'],
                'asin':       obj['parent_asin'],   # product level
                'overall':    float(obj['rating']),
                'reviewText': obj.get('text') or '',
            })
 
    return pd.DataFrame(rows)
 
def load_amazon(path):
    """
    Auto-detect 2014 vs 2023 format by filename suffix:
      *.json.gz   → 2014 format (load_amazon_gz)
      *.jsonl.gz  → 2023 format (load_amazon_2023)
      *.jsonl     → 2023 format (plain text)
    """
    s = str(path)
    if s.endswith('.jsonl') or s.endswith('.jsonl.gz'):
        return load_amazon_2023(path)
    if s.endswith('.json.gz'):
        # delayed import to avoid circularity if you put this in a separate file
        from data_preprocessing import load_amazon_gz
        return load_amazon_gz(path)
    raise ValueError(f"Unknown file format: {path}")
 
# ---------------------------------------------------------------------------
# 2. Descriptive statistics
# ---------------------------------------------------------------------------
def print_stats(data):
    """Print summary statistics for a review DataFrame."""
    n_users = data['reviewerID'].nunique()
    n_items = data['asin'].nunique()
    n_ratings = len(data)
    density = n_ratings / (n_users * n_items)

    print(f"{'='*50}")
    print(f"Reviews: {n_ratings:,}")
    print(f"Users:   {n_users:,}")
    print(f"Items:   {n_items:,}")
    print(f"Density: {density:.6f}")
    print(f"{'='*50}")

    print(f"\nRating distribution:")
    print(data['overall'].value_counts().sort_index().to_string())
    print(f"Average: {data['overall'].mean():.3f}")

    word_counts = data['reviewText'].dropna().str.split().str.len()
    print(f"\nReview length (words):")
    print(f"  Mean:   {word_counts.mean():.1f}")
    print(f"  Median: {word_counts.median():.1f}")
    print(f"  Min:    {word_counts.min()}")
    print(f"  Max:    {word_counts.max()}")

# ---------------------------------------------------------------------------
# 3. Text cleaning
# ---------------------------------------------------------------------------
def clean(text):
    """Tokenise, lowercase, remove punctuation/stopwords/numbers, lemmatise."""
    if not isinstance(text, str) or len(text) == 0:
        return []
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if len(w) > 1]
    tokens = [w for w in tokens if w not in PUNCTUATION]
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [w for w in tokens if not w.isdigit()]
    tokens = [w for w in tokens if w not in EXTRA_REMOVE]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return tokens

# ---------------------------------------------------------------------------
# 4. Train / valid / test split
# ---------------------------------------------------------------------------
def split_data(data, train_frac=0.8, valid_frac=0.1, seed=42):
    """
    Randomly split data into train/valid/test.
    Ensures every user and item in valid/test also appears in train.
    Returns DataFrames with integer indices user_idx and item_idx.
    """
    # Global mappings (created once, shared by all splits)
    unique_users = sorted(data['reviewerID'].unique())
    unique_items = sorted(data['asin'].unique())
    uid2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    sid2idx = {sid: idx for idx, sid in enumerate(unique_items)}

    data = data.copy()
    data['user_idx'] = data['reviewerID'].map(uid2idx)
    data['item_idx'] = data['asin'].map(sid2idx)

    # Random permutation
    rng = np.random.RandomState(seed)
    n = len(data)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_valid = int(valid_frac * n)

    train = data.iloc[idx[:n_train]]
    valid = data.iloc[idx[n_train:n_train + n_valid]]
    test = data.iloc[idx[n_train + n_valid:]]
    print(f"Before orphan check: train={len(train)}, valid={len(valid)}, test={len(test)}")

    # Move orphan users/items back to training
    train_users = set(train['reviewerID'])
    train_items = set(train['asin'])

    move_from_valid = ~valid['reviewerID'].isin(train_users) | ~valid['asin'].isin(train_items)
    move_from_test = ~test['reviewerID'].isin(train_users) | ~test['asin'].isin(train_items)

    train = pd.concat([train, valid[move_from_valid], test[move_from_test]])
    valid = valid[~move_from_valid]
    test = test[~move_from_test]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train):,} ({len(train)/n:.1%})")
    print(f"  Valid: {len(valid):,} ({len(valid)/n:.1%})")
    print(f"  Test:  {len(test):,} ({len(test)/n:.1%})")

    return train, valid, test, uid2idx, sid2idx


# ---------------------------------------------------------------------------
# 5. Build documents (one per item: concatenation of all cleaned reviews)
# ---------------------------------------------------------------------------
def build_documents(data):
    """
    For each item, concatenate all cleaned review tokens into one document.
    Returns a DataFrame with columns ['asin', 'tokens'].
    """
    data = data.copy()
    data['tokens'] = data['reviewText'].apply(clean)
    docs = data.groupby('asin')['tokens'].apply(lambda x: sum(x, [])).reset_index()
    docs.columns = ['asin', 'tokens']
    print(f"\nDocuments: {len(docs):,}")
    print(f"Total tokens: {docs['tokens'].apply(len).sum():,}")
    print(f"Avg tokens per document: {docs['tokens'].apply(len).mean():.0f}")
    return docs


# ---------------------------------------------------------------------------
# 6. K-core filtering to ensure every user/item has at least k reviews
# ---------------------------------------------------------------------------

def filter_k_core(data, k=5, verbose=True):
    n_before = len(data)
    n_users_before = data['reviewerID'].nunique()
    n_items_before = data['asin'].nunique()

    iteration = 0
    while True:
        iteration += 1
        n_start = len(data)

        user_counts = data['reviewerID'].value_counts()
        valid_users = user_counts[user_counts >= k].index
        data = data[data['reviewerID'].isin(valid_users)]

        item_counts = data['asin'].value_counts()
        valid_items = item_counts[item_counts >= k].index
        data = data[data['asin'].isin(valid_items)]

        if len(data) == n_start:
            break

        # Safety: dataset collapsed
        if len(data) == 0:
            if verbose:
                print(f"\n{k}-core filtering: dataset collapsed to empty after "
                      f"{iteration} iteration(s)")
            return data.reset_index(drop=True)

    if verbose:
        print(f"\n{k}-core filtering: converged in {iteration} iteration(s)")
        print(f"  Reviews: {n_before:,} → {len(data):,} "
              f"({len(data)/n_before:.1%} retained)")
        print(f"  Users:   {n_users_before:,} → {data['reviewerID'].nunique():,}")
        print(f"  Items:   {n_items_before:,} → {data['asin'].nunique():,}")

    return data.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 7. Dataset-level stats file
# ---------------------------------------------------------------------------
def save_dataset_stats(data, dataset_name, out_dir):
    """
    Save a one-row CSV with dataset-level statistics to
    <out_dir>/<dataset_name>_stats.csv.
    """
    n_reviews = len(data)
    n_users   = data['reviewerID'].nunique()
    n_items   = data['asin'].nunique()
    density   = n_reviews / (n_users * n_items)
    total_words = int(data['reviewText'].fillna('').str.split().str.len().sum())

    rating_counts = data['overall'].value_counts().sort_index()
    rating_str = ' | '.join(f'{int(r)}★:{int(c)}' for r, c in rating_counts.items())

    row = {
        'n_reviews':          n_reviews,
        'n_users':            n_users,
        'n_items':            n_items,
        'density':            round(density, 6),
        'total_words':        total_words,
        'rating_distribution': rating_str,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{dataset_name}_stats.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved dataset stats → {path}")


# ---------------------------------------------------------------------------
# 8. Distribution histograms + summary table
# ---------------------------------------------------------------------------
def _dist_stats(series, label):
    """Return a dict of summary stats for one distribution series."""
    return {
        'distribution': label,
        'count': int(len(series)),
        'mean':   round(float(series.mean()), 2),
        'median': round(float(series.median()), 2),
        'std':    round(float(series.std()), 2),
        'min':    int(series.min()),
        'p25':    round(float(series.quantile(0.25)), 2),
        'p75':    round(float(series.quantile(0.75)), 2),
        'p95':    round(float(series.quantile(0.95)), 2),
        'max':    int(series.max()),
    }


def plot_distributions(data, dataset_name, out_dir=None, bins=50):
    """
    Save a 1×4 histogram figure and a summary CSV for *data* (raw, pre-split):
      1. Words per review
      2. Words per item  (sum of all review words for that item)
      3. Reviews per user
      4. Reviews per item
    """
    word_counts       = data['reviewText'].fillna('').str.split().str.len()
    words_per_review  = word_counts
    words_per_item    = word_counts.groupby(data['asin']).sum()
    reviews_per_user  = data.groupby('reviewerID').size()
    reviews_per_item  = data.groupby('asin').size()

    panels = [
        (words_per_review, 'Words per review',  'Word count',  '# reviews',  'steelblue'),
        (words_per_item,   'Words per item',     'Word count',  '# items',    'cornflowerblue'),
        (reviews_per_user, 'Reviews per user',   '# reviews',  '# users',    'darkorange'),
        (reviews_per_item, 'Reviews per item',   '# reviews',  '# items',    'seagreen'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(dataset_name, fontsize=13)

    for ax, (series, title, xlabel, ylabel, color) in zip(axes, panels):
        ax.hist(series, bins=bins, color=color, edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        fig_path = os.path.join(out_dir, f'{dataset_name}_distributions.png')
        plt.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"Saved distribution plot → {fig_path}")

        stats_rows = [_dist_stats(s, lbl) for s, lbl, *_ in panels]
        csv_path = os.path.join(out_dir, f'{dataset_name}_distribution_stats.csv')
        fieldnames = ['distribution', 'count', 'mean', 'median', 'std',
                      'min', 'p25', 'p75', 'p95', 'max']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_rows)
        print(f"Saved distribution stats  → {csv_path}")

        hist_rows = []
        for series, label, *_ in panels:
            counts, edges = np.histogram(series, bins=bins)
            for i, cnt in enumerate(counts):
                hist_rows.append({
                    'distribution': label,
                    'bin_start': round(float(edges[i]), 2),
                    'bin_end':   round(float(edges[i + 1]), 2),
                    'count':     int(cnt),
                })
        hist_path = os.path.join(out_dir, f'{dataset_name}_distribution_bins.csv')
        with open(hist_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['distribution', 'bin_start', 'bin_end', 'count'])
            writer.writeheader()
            writer.writerows(hist_rows)
        print(f"Saved distribution bins   → {hist_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import re
    from pathlib import Path

    DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
    DATA_PATHS = [
        DATA_DIR / 'reviews_Musical_Instruments_5.json.gz',
        DATA_DIR / 'reviews_Amazon_Instant_Video_5.json.gz',
        DATA_DIR / 'reviews_Digital_Music_5.json.gz',
    ]

    for path in DATA_PATHS:
        name = re.sub(r'\.json\.gz$', '', re.sub(r'^reviews_', '', os.path.basename(path)))
        print(f"\n=== {name} ===")
        data = load_amazon(path)
        print_stats(data)
        plot_distributions(data, dataset_name=name, out_dir=os.path.join('results', name))