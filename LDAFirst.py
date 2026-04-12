"""
LDAFirst.py

LDAFirst baseline: run LDA on review corpus to obtain topic distributions
theta_i for each item, then use these as fixed item factors in a latent
factor model. Only user factors P and biases are learned.

Prediction: mu + b_u + b_i + p_u^T theta_i

Unlike LDA-LFM, the topic model and rating model are NOT jointly optimised.
LDA runs once, then its output is fed into the rating model as fixed features.
"""

import numpy as np
import gensim
from gensim import corpora

from data_preprocessing import load_amazon_gz, split_data, clean

# ---------------------------------------------------------------------------
# 1. Build LDA topic distributions
# ---------------------------------------------------------------------------
def build_lda(train, sid2idx, n_topics=10, n_vocab=5000):
    """
    Run LDA on training reviews grouped by item.
    Returns theta: np.array of shape (n_items, n_topics)
    """
    # Group reviews by item, clean and concatenate
    train = train.copy()
    train['tokens'] = train['reviewText'].apply(clean)
    docs = train.groupby('asin')['tokens'].apply(lambda x: sum(x, [])).reset_index()
    docs.columns = ['asin', 'tokens']

    # Build dictionary and corpus
    dictionary = corpora.Dictionary(docs['tokens'])
    dictionary.filter_extremes(keep_n=n_vocab)
    corpus = [dictionary.doc2bow(doc) for doc in docs['tokens']]

    # Run LDA
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        chunksize=1000,
        passes=1
    )

    # Extract topic distributions for each item
    n_items = len(sid2idx)
    theta = np.zeros((n_items, n_topics), dtype=np.float32)

    for _, row in docs.iterrows():
        item_idx = sid2idx[row['asin']]
        bow = dictionary.doc2bow(row['tokens'])
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in topic_dist:
            theta[item_idx, topic_id] = prob

    print(f"LDA complete: {n_topics} topics, vocab size {len(dictionary)}")
    return theta, lda_model, dictionary

# ---------------------------------------------------------------------------
# 2. Rating model with fixed item factors (SGD)
# ---------------------------------------------------------------------------
def train_lfm_fixed_q(train, valid, theta, uid2idx, sid2idx,
                      n_epochs=50, lr=0.005, reg=0.02,
                      beta1=0.9, beta2=0.999, eps=1e-8,
                      patience=5):

    n_users = len(uid2idx)
    n_items = len(sid2idx)
    n_topics = theta.shape[1]
    mu = train['overall'].mean()

    rng = np.random.RandomState(42)
    P = rng.normal(0, 0.01, (n_users, n_topics)).astype(np.float32)
    b_u = np.zeros(n_users, dtype=np.float32)
    b_i = np.zeros(n_items, dtype=np.float32)

    m_P = np.zeros_like(P)
    v_P = np.zeros_like(P)
    m_bu = np.zeros_like(b_u)
    v_bu = np.zeros_like(b_u)
    m_bi = np.zeros_like(b_i)
    v_bi = np.zeros_like(b_i)

    users = train['user_idx'].values
    items = train['item_idx'].values
    ratings = train['overall'].values.astype(np.float32)
    n_ratings = len(ratings)

    # Early stopping state
    best_val_mse = np.inf
    best_epoch = 0
    best_P = P.copy()
    best_bu = b_u.copy()
    best_bi = b_i.copy()
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        order = rng.permutation(n_ratings)
        total_loss = 0.0

        for idx in order:
            u = users[idx]
            i = items[idx]
            r = ratings[idx]

            pred = mu + b_u[u] + b_i[i] + P[u] @ theta[i]
            err = r - pred
            total_loss += err ** 2

            g_bu = -err + reg * b_u[u]
            g_bi = -err + reg * b_i[i]
            g_P  = -err * theta[i] + reg * P[u]

            m_bu[u] = beta1 * m_bu[u] + (1 - beta1) * g_bu
            v_bu[u] = beta2 * v_bu[u] + (1 - beta2) * g_bu ** 2
            b_u[u] -= lr * m_bu[u] / (np.sqrt(v_bu[u]) + eps)

            m_bi[i] = beta1 * m_bi[i] + (1 - beta1) * g_bi
            v_bi[i] = beta2 * v_bi[i] + (1 - beta2) * g_bi ** 2
            b_i[i] -= lr * m_bi[i] / (np.sqrt(v_bi[i]) + eps)

            m_P[u] = beta1 * m_P[u] + (1 - beta1) * g_P
            v_P[u] = beta2 * v_P[u] + (1 - beta2) * g_P ** 2
            P[u] -= lr * m_P[u] / (np.sqrt(v_P[u]) + eps)

        # Validation check
        val_pred = predict_lfm_fixed_q(valid, mu, P, b_u, b_i, theta)
        val_mse = np.mean((val_pred - valid['overall'].values) ** 2)
        train_mse = total_loss / n_ratings

        print(f"  Epoch {epoch+1}/{n_epochs}, train MSE: {train_mse:.4f}, valid MSE: {val_mse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            best_P = P.copy()
            best_bu = b_u.copy()
            best_bi = b_i.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1}. Best epoch: {best_epoch} (valid MSE: {best_val_mse:.4f})")
                break

    return mu, best_P, best_bu, best_bi

# ---------------------------------------------------------------------------
# 3. Predict and evaluate
# ---------------------------------------------------------------------------
def predict_lfm_fixed_q(test, mu, P, b_u, b_i, theta):
    """Generate predictions for test set."""
    users = test['user_idx'].values
    items = test['item_idx'].values

    predictions = np.array([
        mu + b_u[u] + b_i[i] + P[u] @ theta[i]
        for u, i in zip(users, items)
    ], dtype=np.float32)

    return predictions


def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE':  np.mean(errors ** 2),
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'MAE':  np.mean(np.abs(errors)),
    }

# ---------------------------------------------------------------------------
# 4. Full pipeline
# ---------------------------------------------------------------------------
def run_lda_first(train, valid, test, uid2idx, sid2idx,
                  n_topics=10, n_vocab=5000, n_epochs=50, lr=0.005, reg=0.02):

    print("Running LDA...")
    theta, lda_model, dictionary = build_lda(train, sid2idx, n_topics, n_vocab)

    mu, P, b_u, b_i = train_lfm_fixed_q(
        train, valid, theta, uid2idx, sid2idx, n_epochs, lr, reg
    )

    predictions = predict_lfm_fixed_q(test, mu, P, b_u, b_i, theta)
    results = evaluate(predictions, test['overall'].values)

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
    results, _, _ = run_lda_first(train, valid, test, uid2idx, sid2idx, n_topics=10)
    elapsed = time.time() - t

    print(f"\nLDAFirst Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  RMSE: {results['RMSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Time: {elapsed:.2f}s")