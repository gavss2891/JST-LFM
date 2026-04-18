"""
LDAFirst.py

LDAFirst baseline: run LDA on the review corpus to obtain topic distributions
theta_i for each item, then use these as fixed item factors in a latent
factor model. Only user factors P and biases are learned.

Prediction: mu + b_u + b_i + p_u^T theta_i

Unlike LDA-LFM, the topic model and rating model are NOT jointly optimised.
LDA runs once, then its output is fed into the rating model as fixed features.

Tuning matches LFM.py: vectorised batch Adam with a small grid search over
lr, reg, and n_epochs on the validation set; the best model is then evaluated
on the test set. LDA is built only once and reused across all grid combos.
"""

import numpy as np
import gensim
from gensim import corpora

from data_preprocessing import load_amazon_gz, split_data, clean


# ---------------------------------------------------------------------------
# 1. Build LDA topic distributions (run once, before tuning)
# ---------------------------------------------------------------------------
def build_lda(train, sid2idx, n_topics=10, n_vocab=5000, passes=1):
    """
    Run LDA on training reviews grouped by item.

    Returns
    -------
    theta : np.ndarray of shape (n_items, n_topics)
        Each row is the topic distribution for item i. Items with no training
        reviews receive the mean theta over seen items (a valid distribution,
        unlike the zero vector).
    lda_model, dictionary : fitted gensim objects
    """
    train = train.copy()
    train['tokens'] = train['reviewText'].apply(clean)
    docs = train.groupby('asin')['tokens'].apply(lambda x: sum(x, [])).reset_index()
    docs.columns = ['asin', 'tokens']

    dictionary = corpora.Dictionary(docs['tokens'])
    dictionary.filter_extremes(keep_n=n_vocab)
    corpus = [dictionary.doc2bow(doc) for doc in docs['tokens']]

    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        chunksize=1000,
        passes=passes,
    )

    n_items = len(sid2idx)
    theta = np.zeros((n_items, n_topics), dtype=np.float64)
    seen = np.zeros(n_items, dtype=bool)

    for _, row in docs.iterrows():
        item_idx = sid2idx[row['asin']]
        bow = dictionary.doc2bow(row['tokens'])
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in topic_dist:
            theta[item_idx, topic_id] = prob
        seen[item_idx] = True

    # Cold items: replace the zero vector with the mean of seen items'
    # distributions so p_u^T theta_i is a meaningful contribution.
    if (~seen).any():
        if seen.any():
            theta[~seen] = theta[seen].mean(axis=0)
        else:
            theta[~seen] = 1.0 / n_topics  # uniform fallback

    print(f"LDA complete: {n_topics} topics, vocab {len(dictionary)}, "
          f"{seen.sum()}/{n_items} items have training reviews "
          f"(cold items filled with mean theta)")
    return theta, lda_model, dictionary


# ---------------------------------------------------------------------------
# 2. Train rating model (fixed theta) with vectorised batch Adam
# ---------------------------------------------------------------------------
def train_lfm_fixed_q(train, theta, uid2idx, sid2idx, n_epochs=35,
                      lr=0.005, reg=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
    """Train LFM with item factors fixed to theta. Returns mu, P, b_u, b_i."""
    n_users = len(uid2idx)
    n_items = len(sid2idx)
    n_topics = theta.shape[1]
    mu = train['overall'].mean()

    rng = np.random.RandomState(42)
    P = rng.normal(0, 0.01, (n_users, n_topics)).astype(np.float64)
    b_u = np.zeros(n_users, dtype=np.float64)
    b_i = np.zeros(n_items, dtype=np.float64)

    # Adam state
    adam = {name: {'m': np.zeros_like(p), 'v': np.zeros_like(p)}
            for name, p in [('P', P), ('b_u', b_u), ('b_i', b_i)]}

    users = train['user_idx'].values
    items = train['item_idx'].values
    ratings = train['overall'].values.astype(np.float64)
    n_ratings = len(ratings)

    for epoch in range(n_epochs):
        pred = mu + b_u[users] + b_i[items] + np.sum(P[users] * theta[items], axis=1)
        err = pred - ratings

        # Gradients
        err_2 = 2 * err / n_ratings
        g_bu = np.zeros_like(b_u)
        g_bi = np.zeros_like(b_i)
        g_P = np.zeros_like(P)

        np.add.at(g_bu, users, err_2)
        np.add.at(g_bi, items, err_2)
        np.add.at(g_P, users, err_2[:, None] * theta[items])

        g_bu += 2 * reg * b_u
        g_bi += 2 * reg * b_i
        g_P += 2 * reg * P

        # Adam updates
        for name, grad in [('b_u', g_bu), ('b_i', g_bi), ('P', g_P)]:
            adam[name]['m'] = beta1 * adam[name]['m'] + (1 - beta1) * grad
            adam[name]['v'] = beta2 * adam[name]['v'] + (1 - beta2) * grad ** 2

        b_u = b_u - lr * adam['b_u']['m'] / (np.sqrt(adam['b_u']['v']) + eps)
        b_i = b_i - lr * adam['b_i']['m'] / (np.sqrt(adam['b_i']['v']) + eps)
        P = P - lr * adam['P']['m'] / (np.sqrt(adam['P']['v']) + eps)

    return mu, P, b_u, b_i


# ---------------------------------------------------------------------------
# 3. Predict and evaluate (vectorised)
# ---------------------------------------------------------------------------
def predict_lfm_fixed_q(data, mu, P, b_u, b_i, theta):
    users = data['user_idx'].values
    items = data['item_idx'].values
    return mu + b_u[users] + b_i[items] + np.sum(P[users] * theta[items], axis=1)


def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE':  np.mean(errors ** 2),
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'MAE':  np.mean(np.abs(errors)),
    }


# ---------------------------------------------------------------------------
# 4. Full pipeline with grid-search tuning
# ---------------------------------------------------------------------------
def run_lda_first_tuned(train, valid, test, uid2idx, sid2idx,
                        n_topics=10, n_vocab=5000, lda_passes=1):
    """
    Build LDA features once, then grid search lr, reg, n_epochs on validation
    set and evaluate the best configuration on test set. Mirrors run_lfm_tuned.
    """
    print("Running LDA...")
    theta, lda_model, dictionary = build_lda(
        train, sid2idx, n_topics, n_vocab, passes=lda_passes
    )

    lr_grid = [0.005, 0.01, 0.05]
    reg_grid = [0.02, 1.0, 10.0]
    epoch_grid = [5, 20, 50]

    best_val_mse = np.inf
    best_params = None

    for lr in lr_grid:
        for reg in reg_grid:
            for n_epochs in epoch_grid:
                mu, P, b_u, b_i = train_lfm_fixed_q(
                    train, theta, uid2idx, sid2idx, n_epochs, lr, reg
                )
                val_pred = predict_lfm_fixed_q(valid, mu, P, b_u, b_i, theta)
                val_mse = np.mean((val_pred - valid['overall'].values) ** 2)

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_params = (lr, reg, n_epochs, mu, P, b_u, b_i)

    lr, reg, n_epochs, mu, P, b_u, b_i = best_params
    print(f"  Best LDAFirst: lr={lr}, reg={reg}, epochs={n_epochs}, "
          f"val MSE={best_val_mse:.4f}")

    test_pred = predict_lfm_fixed_q(test, mu, P, b_u, b_i, theta)
    results = evaluate(test_pred, test['overall'].values)
    return results, lda_model, dictionary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results, _, _ = run_lda_first_tuned(
        train, valid, test, uid2idx, sid2idx, n_topics=10
    )
    elapsed = time.time() - t

    print(f"\nLDAFirst Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Time: {elapsed:.2f}s")