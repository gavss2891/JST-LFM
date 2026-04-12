"""
1_data_preprocessing.py

Loads the Amazon 5-core review data, cleans review text, splits into
train/valid/test sets, and prepares documents for topic modelling.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import gzip
import json
import string

import numpy as np
import pandas as pd

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
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    # Load
    data = load_amazon_gz(DATA_PATH)
    print_stats(data)

    # Split
    train, valid, test, uid2idx, sid2idx = split_data(data)

    # Build documents from training reviews only
    docs = build_documents(train)

    # Preview
    print(f"\nSample document (first 20 tokens):")
    print(docs.iloc[0]['tokens'][:20])