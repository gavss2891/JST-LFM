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

Gibbs sweep implementation: Numba-JIT sequential pass that, for each token,
computes p_z(k) = theta[d,k] * phi[k,w], inverse-CDF samples z, and
increments count matrices in place. Faithful to Algorithm 2 of the
methodology (parametric sampling distribution fixed within each iteration);
only the bookkeeping is compiled. First call JIT-compiles (~5-10s); every
later call runs at compiled speed thanks to cache=True.
"""

import numpy as np
import numba
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
    if 'tokens' not in train.columns:
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
# 4. Numba-JIT sweep: sample + count + log-likelihood + grad_kappa accumulator
# ---------------------------------------------------------------------------
@numba.njit(cache=True, fastmath=True)
def _ldalfm_sweep(all_words, all_docs, theta, phi, q, eq,
                  K, rand_uniforms,
                  N_ki, N_kw, N_k, all_topics):
    """
    One sequential pass over the corpus.

    For each token position j (with d = all_docs[j], w = all_words[j]):
        p_z(k)      = theta[d, k] * phi[k, w]             for k = 0..K-1
        z_j         ~  Multinomial(p_z / sum p_z)          via inverse CDF
        all_topics[j] <- z_j
        N_ki[d, z_j] += 1     (document-topic count)
        N_kw[z_j, w] += 1     (topic-word count)
        N_k[z_j]     += 1     (topic count)
        lk          += log(p_z[z_j])                       (corpus log-likelihood)
        grad_kappa  += q[d, z_j] - eq[d]                   (kappa gradient accumulator)

    theta, phi, q, eq are held fixed throughout the sweep — this preserves
    the parametric sampling distribution of Algorithm 2. The count matrices
    N_ki, N_kw, N_k and the output array all_topics are modified in place.

    Parameters
    ----------
    rand_uniforms : (N_total,) float64 array of uniforms pre-drawn from the
                    numpy MT19937 stream. Passed in so the Numba path uses
                    the same random sequence the numpy path would have used.

    Returns
    -------
    lk          : float, corpus log-likelihood under the sampled assignments
    grad_kappa  : float, sum_j (q[d_j, z_j] - E_theta[q_{d_j}])
    """
    N_total = all_words.shape[0]
    probs = np.empty(K, dtype=np.float64)
    lk = 0.0
    grad_kappa = 0.0

    for j in range(N_total):
        d = all_docs[j]
        w = all_words[j]

        total = 0.0
        for k in range(K):
            p = theta[d, k] * phi[k, w]
            probs[k] = p
            total += p

        r = rand_uniforms[j] * total
        cum = 0.0
        z = K - 1  # fallback guard against FP underflow
        for k in range(K):
            cum += probs[k]
            if r < cum:
                z = k
                break

        all_topics[j] = z
        N_ki[d, z] += 1.0
        N_kw[z, w] += 1.0
        N_k[z]     += 1.0

        lk         += np.log(probs[z] + 1e-30)
        grad_kappa += q[d, z] - eq[d]

    return lk, grad_kappa


# ---------------------------------------------------------------------------
# 5. Prediction and evaluation
# ---------------------------------------------------------------------------
def predict_ratings(data, mu, P, Q, b_u, b_i):
    users = data['user_idx'].values
    items = data['item_idx'].values
    return mu + b_u[users] + b_i[items] + np.sum(P[users] * Q[items], axis=1)


def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE':  np.mean(errors ** 2),
        'MAE':  np.mean(np.abs(errors)),
    }


# ---------------------------------------------------------------------------
# 6. Top words per topic (for diagnostic / reporting)
# ---------------------------------------------------------------------------
def top_words_per_topic(psi, dictionary, top_n=10):
    """
    Return the top-N highest-probability words for each topic, computed
    from phi = softmax(psi).

    Returns
    -------
    dict {topic_k: [(word, probability), ...]}
    """
    phi = softmax(psi)
    n_topics = phi.shape[0]
    result = {}
    for k in range(n_topics):
        top_idx = np.argsort(-phi[k])[:top_n]
        result[k] = [(dictionary[int(i)], float(phi[k, int(i)]))
                     for i in top_idx]
    return result


# ---------------------------------------------------------------------------
# 7. Core fit: single run with checkpointed validation evaluations
# ---------------------------------------------------------------------------
def fit_lda_lfm(train, valid, doc_words, uid2idx, sid2idx,
                n_topics=10, n_vocab=5000,
                lr=0.005, reg=0.02, mu_corpus=1.0, kappa_init=1.0,
                beta1=0.9, beta2=0.999, eps=1e-8,
                n_epochs=300, verbose=False):
    """
    Train LDA-LFM with Adam. Evaluates val MSE every epoch and returns the
    best epoch's params plus the full MSE history.

    If verbose, prints a one-line summary per epoch (useful outside the
    grid search; the grid handles its own per-combo progress reporting).
    """
    import time as _time

    n_users = len(uid2idx)
    n_items = len(sid2idx)

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
    N_total = int(all_words.shape[0])

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

    # Preallocated sweep buffers (zeroed each epoch, reused across epochs)
    N_ki_buf = np.zeros((n_items, n_topics), dtype=np.float64)
    N_kw_buf = np.zeros((n_topics, n_vocab), dtype=np.float64)
    N_k_buf  = np.zeros(n_topics, dtype=np.float64)
    all_topics = np.zeros(N_total, dtype=np.int64)

    mse_history = []
    _best_vmse = np.inf
    _best_epoch = -1
    _best_params = None
    _patience_counter = 0
    _prev_vmse = np.inf
    _loss_diff_counter = 0

    for epoch in range(n_epochs):
        t_epoch = _time.time()

        # 1) Compute theta and phi from current parameters
        theta = softmax(kappa * Q)                         # (n_items, K)
        phi   = softmax(psi)                               # (K, V)
        eq    = np.sum(theta * Q, axis=1)                  # (n_items,) E_theta[q_i]

        # 2) Zero sweep buffers and run JIT-compiled Gibbs sweep
        N_ki_buf.fill(0.0)
        N_kw_buf.fill(0.0)
        N_k_buf.fill(0.0)
        rand_uniforms = rng.random_sample(N_total)
        lk, grad_kappa_corpus = _ldalfm_sweep(
            all_words, all_docs, theta, phi, Q, eq,
            n_topics, rand_uniforms,
            N_ki_buf, N_kw_buf, N_k_buf, all_topics,
        )

        # 3) Corpus gradients from counts (cheap, vectorised)
        grad_q_corpus   = kappa * (N_ki_buf - n_d[:, None] * theta)
        grad_psi_corpus = N_kw_buf - N_k_buf[:, None] * phi
        # grad_kappa_corpus already returned from the sweep

        # 4) Rating gradients (vectorised batch)
        pred = predict_ratings(train, mu, P, Q, b_u, b_i)
        err = pred - ratings
        err_2 = 2 * err / n_ratings

        grad_P        = np.zeros_like(P)
        grad_Q_rating = np.zeros_like(Q)
        grad_bu       = np.zeros_like(b_u)
        grad_bi       = np.zeros_like(b_i)

        np.add.at(grad_bu, users, err_2)
        np.add.at(grad_bi, items, err_2)
        np.add.at(grad_P,  users, err_2[:, None] * Q[items])
        np.add.at(grad_Q_rating, items, err_2[:, None] * P[users])

        grad_bu += 2 * reg * b_u
        grad_bi += 2 * reg * b_i
        grad_P  += 2 * reg * P
        # Q is NOT regularised by lambda; the corpus term does that job.

        # 5) Combine rating and corpus gradients
        #    Corpus gradient is divided by total_words so that mu_corpus ~ 1
        #    is a neutral, per-observation trade-off (see methodology note).
        grad_Q     = grad_Q_rating - mu_corpus * grad_q_corpus   / total_words
        grad_psi   =                - mu_corpus * grad_psi_corpus / total_words
        grad_kappa =                - mu_corpus * grad_kappa_corpus / total_words

        # 6) Adam updates with bias correction
        t = epoch + 1
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        def adam_update(name, param, grad):
            adam[name]['m'] = beta1 * adam[name]['m'] + (1 - beta1) * grad
            adam[name]['v'] = beta2 * adam[name]['v'] + (1 - beta2) * grad ** 2
            return param - lr * (adam[name]['m'] / bc1) / (np.sqrt(adam[name]['v'] / bc2) + eps)

        P   = adam_update('P',   P,   grad_P)
        Q   = adam_update('Q',   Q,   grad_Q)
        b_u = adam_update('b_u', b_u, grad_bu)
        b_i = adam_update('b_i', b_i, grad_bi)
        psi = adam_update('psi', psi, grad_psi)

        adam['kappa']['m'] = beta1 * adam['kappa']['m'] + (1 - beta1) * grad_kappa
        adam['kappa']['v'] = beta2 * adam['kappa']['v'] + (1 - beta2) * grad_kappa ** 2
        kappa = kappa - lr * (adam['kappa']['m'] / bc1) / (np.sqrt(adam['kappa']['v'] / bc2) + eps)

        # 7) Evaluate on validation every epoch; keep best params
        val_pred = predict_ratings(valid, mu, P, Q, b_u, b_i)
        val_mse = float(np.mean((val_pred - valid['overall'].values) ** 2))
        mse_history.append((epoch + 1, val_mse))

        if abs(val_mse - _prev_vmse) < 1e-6:
            _loss_diff_counter += 1
            if _loss_diff_counter >= 10:
                break
        else:
            _loss_diff_counter = 0

        if val_mse < _best_vmse:
            _best_vmse = val_mse
            _best_epoch = epoch + 1
            _best_params = (mu, P.copy(), Q.copy(), b_u.copy(), b_i.copy(),
                            psi.copy(), float(kappa))
            _patience_counter = 0
        else:
            _patience_counter += 1
            if _patience_counter >= 100:
                break

        _prev_vmse = val_mse

        if verbose:
            dt = _time.time() - t_epoch
            train_mse = float(np.mean(err ** 2))
            print(f"      epoch {epoch+1}/{n_epochs}  "
                  f"train MSE {train_mse:.4f}  "
                  f"kappa {kappa:+.3f}  "
                  f"lk {lk:.1f}  "
                  f"dt {dt:.2f}s",
                  flush=True)

    return _best_epoch, _best_vmse, _best_params, mse_history


# ---------------------------------------------------------------------------
# 8. Full pipeline with grid-search tuning
# ---------------------------------------------------------------------------
def run_lda_lfm_tuned(train, valid, test, uid2idx, sid2idx,
                      n_topics=10, n_vocab=5000):
    """
    Build the corpus once, then grid-search lr, reg, mu_corpus on the
    validation set using the checkpoint trick for n_epochs. Evaluate the
    best configuration on the test set. Mirrors run_lfm_tuned in LFM.py.

    Returns
    -------
    results     : dict with MSE, RMSE, MAE on test set
    best_info   : dict with best lr, reg, mu, epochs
    tuning_rows : list of dicts, one per (lr, reg, mu, n_epochs, val_mse)
    topic_words : dict {topic_k: [(word, prob), ...]} from the best model
    dictionary  : gensim Dictionary
    """
    import time as _time

    print("Building corpus...")
    doc_words, dictionary = build_corpus(train, sid2idx, n_vocab)
    actual_n_vocab = len(dictionary)

    lr_grid  = [0.01, 0.02]
    reg_grid = [0.001]
    mu_grid  = [100.0, 200.0]
    n_epochs = 1000

    best_val_mse = np.inf
    best = None  # (lr, reg, mu_c, n_ep, params)
    tuning_rows = []
    best_mse_history = None

    n_combos = len(lr_grid) * len(reg_grid) * len(mu_grid)
    print(f"Tuning LDA-LFM ({n_combos} combos; first combo JIT-compiles, "
          f"~5-10s)...", flush=True)
    t_tune = _time.time()
    combo_idx = 0

    for lr in lr_grid:
        for reg in reg_grid:
            for mu_c in mu_grid:
                combo_idx += 1
                t_combo = _time.time()
                best_ep, best_vmse, params, mse_hist = fit_lda_lfm(
                    train, valid, doc_words, uid2idx, sid2idx,
                    n_topics=n_topics, n_vocab=actual_n_vocab,
                    lr=lr, reg=reg, mu_corpus=mu_c,
                    n_epochs=n_epochs, verbose=False,
                )

                for ep, vmse in mse_hist:
                    tuning_rows.append({
                        'lr': lr, 'reg': reg, 'mu': mu_c,
                        'n_epochs': ep, 'val_mse': vmse,
                    })
                if best_vmse < best_val_mse:
                    best_val_mse = best_vmse
                    best = (lr, reg, mu_c, best_ep, params)
                    best_mse_history = mse_hist

                dt_combo = _time.time() - t_combo
                elapsed  = _time.time() - t_tune
                print(f"  [{combo_idx}/{n_combos}] lr={lr}, reg={reg}, mu={mu_c}: "
                      f"best val {best_vmse:.4f} @ epoch {best_ep}  "
                      f"(combo {dt_combo:.1f}s, elapsed {elapsed:.1f}s)",
                      flush=True)

    lr, reg, mu_c, n_ep, params = best
    print(f"\n  Best LDA-LFM: lr={lr}, reg={reg}, mu={mu_c}, "
          f"epochs={n_ep}, val MSE={best_val_mse:.4f}")

    mu, P, Q, b_u, b_i, psi, kappa = params
    test_pred = predict_ratings(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)
    results['test_pred'] = test_pred
    best_info = {'lr': lr, 'reg': reg, 'mu': mu_c, 'epochs': n_ep}

    topic_words = top_words_per_topic(psi, dictionary, top_n=10)

    return results, best_info, tuning_rows, best_mse_history, psi, topic_words, dictionary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results, best_info, _, _, _, _ = run_lda_lfm_tuned(
        train, valid, test, uid2idx, sid2idx, n_topics=10,
    )
    elapsed = time.time() - t

    print(f"\nLDA-LFM Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Best: lr={best_info['lr']}, reg={best_info['reg']}, "
          f"mu={best_info['mu']}, epochs={best_info['epochs']}")
    print(f"  Time: {elapsed:.2f}s")