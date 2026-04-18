"""
LDA_LFM.py

LDA-LFM model: jointly optimises latent factor rating prediction with
LDA-based topic modelling. Item factors q_i are linked to topic
distributions theta_i via softmax: theta_{i,k} = softmax(kappa * q_{i,k}).

Prediction: mu + b_u + b_i + p_u^T q_i

The corpus log-likelihood and rating loss are jointly optimised so that
the rating signal informs the topics and the topics regularise the item
factors.

Tuning matches the structure of LFM.py: a grid search over lr, reg, and
mu_corpus on the validation set, using the "checkpoint trick" to evaluate
multiple epoch counts in a single training run. The corpus log-likelihood
gradient is divided by N_total so that mu_corpus ~ 1 is a neutral default.
"""

import numpy as np
from gensim import corpora

from data_preprocessing import load_amazon_gz, split_data, clean


# ---------------------------------------------------------------------------
# 1. Build corpus: one document per item, stored as flat arrays for speed
# ---------------------------------------------------------------------------
def build_corpus(train, sid2idx, n_vocab=5000):
    """
    Group reviews by item, clean, build vocabulary, convert to word indices.
    Returns doc_words (list of arrays, one per item) and the gensim dictionary.
    """
    train = train.copy()
    train['tokens'] = train['reviewText'].apply(clean)
    docs = train.groupby('asin')['tokens'].apply(lambda x: sum(x, [])).reset_index()
    docs.columns = ['asin', 'tokens']

    dictionary = corpora.Dictionary(docs['tokens'])
    dictionary.filter_extremes(keep_n=n_vocab)

    n_items = len(sid2idx)
    doc_words = [np.array([], dtype=np.int32) for _ in range(n_items)]

    for _, row in docs.iterrows():
        item_idx = sid2idx[row['asin']]
        word_ids = [dictionary.token2id[w] for w in row['tokens']
                    if w in dictionary.token2id]
        doc_words[item_idx] = np.array(word_ids, dtype=np.int32)

    n_words_total = sum(len(d) for d in doc_words)
    print(f"Corpus: {n_items} documents, {n_words_total:,} tokens, "
          f"vocab size {len(dictionary)}")

    return doc_words, dictionary


# ---------------------------------------------------------------------------
# 2. Softmax (numerically stable)
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 3. Flatten doc_words into contiguous arrays for vectorised ops
# ---------------------------------------------------------------------------
def flatten_corpus(doc_words):
    """
    Returns
    -------
    all_words : (N_total,) int array of word ids
    all_docs  : (N_total,) int array of doc ids (parallel to all_words)
    n_d       : (n_items,) int array of words per document
    """
    n_d = np.array([len(d) for d in doc_words], dtype=np.int64)
    all_words = np.concatenate(doc_words).astype(np.int64) if n_d.sum() > 0 \
                else np.array([], dtype=np.int64)
    all_docs = np.repeat(np.arange(len(doc_words), dtype=np.int64), n_d)
    return all_words, all_docs, n_d


# ---------------------------------------------------------------------------
# 4. Gibbs sampling
# ---------------------------------------------------------------------------
def gibbs_sample_vec(all_words, all_docs, theta, phi, rng):
    """
    Sample a topic for every word in the corpus from
        p(z=k) proportional to theta[i,k] * phi[k, w].

    Returns
    -------
    all_topics : (N_total,) int array of sampled topics
    lk         : corpus log-likelihood under the sampled assignments
    """
    # raw[j, k] = theta[all_docs[j], k] * phi[k, all_words[j]]
    raw = theta[all_docs] * phi.T[all_words]          # (N_total, K)
    probs = raw / raw.sum(axis=1, keepdims=True)

    # Inverse-CDF categorical sampling 
    cumprobs = probs.cumsum(axis=1)
    u = rng.uniform(size=(len(all_words), 1))
    all_topics = (cumprobs < u).sum(axis=1).astype(np.int64)

    # Log-likelihood under the sampled assignments
    lk = np.log(raw[np.arange(len(all_words)), all_topics] + 1e-30).sum()
    return all_topics, lk


# ---------------------------------------------------------------------------
# 5. Corpus gradients
# ---------------------------------------------------------------------------
def compute_corpus_gradients_vec(all_words, all_docs, all_topics,
                                 theta, phi, q, kappa,
                                 n_items, n_topics, n_vocab, n_d):
    """
    Analytical gradients of the LDA-LFM corpus log-likelihood:
        grad_q[i,k]  = kappa * (N_{k,i} - N_i * theta[i,k])
        grad_psi[k,w] = N_{k,w} - N_k * phi[k,w]
        grad_kappa   = sum_{i,j} (q[i, z_{i,j}] - E_theta[q_i])
    """
    # Document-topic counts: N_{k,i}
    topic_counts = np.bincount(
        all_docs * n_topics + all_topics,
        minlength=n_items * n_topics,
    ).reshape(n_items, n_topics).astype(np.float64)

    # Topic-word counts: N_{k,w}
    word_topic_counts = np.bincount(
        all_topics * n_vocab + all_words,
        minlength=n_topics * n_vocab,
    ).reshape(n_topics, n_vocab).astype(np.float64)

    # grad_q
    grad_q = kappa * (topic_counts - n_d[:, None] * theta)

    # grad_psi
    N_k = word_topic_counts.sum(axis=1, keepdims=True)   # (n_topics, 1)
    grad_psi = word_topic_counts - N_k * phi

    # grad_kappa
    eq = (theta * q).sum(axis=1)                          # E_theta[q_i]
    grad_kappa = float(q[all_docs, all_topics].sum() - eq[all_docs].sum())

    return grad_q, grad_psi, grad_kappa


# ---------------------------------------------------------------------------
# 6. Prediction and evaluation
# ---------------------------------------------------------------------------
def predict_ratings(data, mu, P, Q, b_u, b_i):
    users = data['user_idx'].values
    items = data['item_idx'].values
    return mu + b_u[users] + b_i[items] + np.sum(P[users] * Q[items], axis=1)


def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE':  np.mean(errors ** 2),
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'MAE':  np.mean(np.abs(errors)),
    }


# ---------------------------------------------------------------------------
# 7. Core fit: single run with checkpointed validation evaluations
# ---------------------------------------------------------------------------
def fit_lda_lfm(train, valid, doc_words, uid2idx, sid2idx,
                n_topics=10, n_vocab=5000,
                lr=0.005, reg=0.02, mu_corpus=1.0, kappa_init=1.0,
                beta1=0.9, beta2=0.999, eps=1e-8,
                checkpoint_epochs=(5, 20, 50)):
    """
    Train LDA-LFM with Adam. Runs for max(checkpoint_epochs) epochs and
    returns a dict mapping each checkpoint epoch to (val_mse, params_copy).
    """
    n_users = len(uid2idx)
    n_items = len(sid2idx)
    n_epochs = max(checkpoint_epochs)
    checkpoint_set = set(checkpoint_epochs)

    rng = np.random.RandomState(42)
    mu = train['overall'].mean()

    # Parameters
    Q   = rng.normal(0, 0.01, (n_items, n_topics)).astype(np.float64)
    P   = rng.normal(0, 0.01, (n_users, n_topics)).astype(np.float64)
    b_u = np.zeros(n_users, dtype=np.float64)
    b_i = np.zeros(n_items, dtype=np.float64)
    psi = rng.normal(0, 0.01, (n_topics, n_vocab)).astype(np.float64)
    kappa = float(kappa_init)

    # Flat corpus arrays (built once per fit)
    all_words, all_docs, n_d = flatten_corpus(doc_words)
    total_words = int(n_d.sum())

    # Adam state
    adam = {name: {'m': np.zeros_like(param), 'v': np.zeros_like(param)}
            for name, param in [('Q', Q), ('P', P), ('b_u', b_u),
                                ('b_i', b_i), ('psi', psi)]}
    adam['kappa'] = {'m': 0.0, 'v': 0.0}

    # Rating arrays
    users = train['user_idx'].values
    items = train['item_idx'].values
    ratings = train['overall'].values.astype(np.float64)
    n_ratings = len(ratings)

    checkpoints = {}

    for epoch in range(n_epochs):

        # 1) Compute theta and phi from current parameters
        theta = softmax(kappa * Q)                         # (n_items, K)
        phi   = softmax(psi)                               # (K, V)

        # 2) Vectorised Gibbs sampling
        all_topics, _lk = gibbs_sample_vec(all_words, all_docs, theta, phi, rng)

        # 3) Vectorised corpus gradients
        grad_q_corpus, grad_psi_corpus, grad_kappa_corpus = \
            compute_corpus_gradients_vec(all_words, all_docs, all_topics,
                                         theta, phi, Q, kappa,
                                         n_items, n_topics, n_vocab, n_d)

        # 4) Rating gradients (vectorised batch)
        pred = predict_ratings(train, mu, P, Q, b_u, b_i)
        err = pred - ratings
        err_2 = 2 * err / n_ratings

        grad_P       = np.zeros_like(P)
        grad_Q_rating = np.zeros_like(Q)
        grad_bu      = np.zeros_like(b_u)
        grad_bi      = np.zeros_like(b_i)

        np.add.at(grad_bu, users, err_2)
        np.add.at(grad_bi, items, err_2)
        np.add.at(grad_P,  users, err_2[:, None] * Q[items])
        np.add.at(grad_Q_rating, items, err_2[:, None] * P[users])

        grad_bu += 2 * reg * b_u
        grad_bi += 2 * reg * b_i
        grad_P  += 2 * reg * P

        # 5) Combine rating and corpus gradients
        #    Corpus gradient is divided by total_words so that mu_corpus ~ 1
        #    is a neutral, per-observation trade-off
        grad_Q     = grad_Q_rating - mu_corpus * grad_q_corpus   / total_words
        grad_psi   =                - mu_corpus * grad_psi_corpus / total_words
        grad_kappa =                - mu_corpus * grad_kappa_corpus / total_words

        # 6) Adam updates
        def adam_update(name, param, grad):
            adam[name]['m'] = beta1 * adam[name]['m'] + (1 - beta1) * grad
            adam[name]['v'] = beta2 * adam[name]['v'] + (1 - beta2) * grad ** 2
            return param - lr * adam[name]['m'] / (np.sqrt(adam[name]['v']) + eps)

        P   = adam_update('P',   P,   grad_P)
        Q   = adam_update('Q',   Q,   grad_Q)
        b_u = adam_update('b_u', b_u, grad_bu)
        b_i = adam_update('b_i', b_i, grad_bi)
        psi = adam_update('psi', psi, grad_psi)

        # Scalar Adam for kappa
        adam['kappa']['m'] = beta1 * adam['kappa']['m'] + (1 - beta1) * grad_kappa
        adam['kappa']['v'] = beta2 * adam['kappa']['v'] + (1 - beta2) * grad_kappa ** 2
        kappa = kappa - lr * adam['kappa']['m'] / (np.sqrt(adam['kappa']['v']) + eps)

        # 7) Checkpoint: evaluate on validation and snapshot
        if (epoch + 1) in checkpoint_set:
            val_pred = predict_ratings(valid, mu, P, Q, b_u, b_i)
            val_mse = float(np.mean((val_pred - valid['overall'].values) ** 2))
            checkpoints[epoch + 1] = (
                val_mse,
                (mu, P.copy(), Q.copy(), b_u.copy(), b_i.copy(),
                 psi.copy(), float(kappa)),
            )

    return checkpoints


# ---------------------------------------------------------------------------
# 8. Full pipeline with grid-search tuning
# ---------------------------------------------------------------------------
def run_lda_lfm_tuned(train, valid, test, uid2idx, sid2idx,
                      n_topics=10, n_vocab=5000):
    """
    Build the corpus once, then grid-search lr, reg, mu_corpus on the
    validation set using the checkpoint trick for n_epochs. Evaluate the
    best configuration on the test set. Mirrors run_lfm_tuned in LFM.py.
    """
    print("Building corpus...")
    doc_words, dictionary = build_corpus(train, sid2idx, n_vocab)
    actual_n_vocab = len(dictionary)

    lr_grid    = [0.05, 0.1, 0.15]
    reg_grid   = [0.02, 1.0, 10.0]
    mu_grid    = [1.0, 10.0, 100]
    epoch_grid = [5, 20, 50, 100, 200]

    best_val_mse = np.inf
    best = None  # (lr, reg, mu_c, n_ep, params)

    print("Tuning LDA-LFM...")
    for lr in lr_grid:
        for reg in reg_grid:
            for mu_c in mu_grid:
                ckpts = fit_lda_lfm(
                    train, valid, doc_words, uid2idx, sid2idx,
                    n_topics=n_topics, n_vocab=actual_n_vocab,
                    lr=lr, reg=reg, mu_corpus=mu_c,
                    checkpoint_epochs=epoch_grid,
                )

                for n_ep, (vmse, params) in ckpts.items():
                    if vmse < best_val_mse:
                        best_val_mse = vmse
                        best = (lr, reg, mu_c, n_ep, params)

                ckpts_str = " ".join(
                    f"@{ep}={ckpts[ep][0]:.4f}" for ep in sorted(ckpts.keys())
                )
                print(f"  lr={lr}, reg={reg}, mu={mu_c}: val {ckpts_str}")

    lr, reg, mu_c, n_ep, params = best
    print(f"\n  Best LDA-LFM: lr={lr}, reg={reg}, mu={mu_c}, "
          f"epochs={n_ep}, val MSE={best_val_mse:.4f}")

    mu, P, Q, b_u, b_i, psi, kappa = params
    test_pred = predict_ratings(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)

    return results, params, dictionary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results, params, dictionary = run_lda_lfm_tuned(
        train, valid, test, uid2idx, sid2idx, n_topics=10,
    )
    elapsed = time.time() - t

    print(f"\nLDA-LFM Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  RMSE: {results['RMSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Time: {elapsed:.2f}s")