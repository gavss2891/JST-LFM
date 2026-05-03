"""
JST_LFM.py

JST-LFM model: jointly optimises latent factor rating prediction with
sentiment-aware topic modelling. Item factors q_{i,l} are sentiment-specific
and linked to topic distributions theta_{d,k,l} via softmax over the topic
axis (one softmax per sentiment):

    theta[i, l, k] = softmax_k(kappa * q[i, l, k])

Word distributions phi[l, k, w] = softmax_w(psi[l, k, w]) are jointly
optimised. The sentiment distribution pi[d, l] is COUNT-BASED and NOT
parameterised — it is recomputed at the start of each outer iteration
from the current sentiment counts (see methodology Eq. 24).

Rating prediction (methodology Eq. 23):
    r_hat = mu + b_u + b_i + sum_l p_{u,l}^T q_{i,l}
          = mu + b_u + b_i + p_u @ q_i   (using flat S*K-dim vectors)

Gibbs sampler implementation: Numba-JIT sequential pass that, for each
token, computes p(k, l) = pi[d, l] * theta[d, l, k] * phi[l, k, w],
samples (z_j, l_j) jointly via inverse-CDF, and increments count matrices
in place. Faithful to Algorithm 3 of the methodology (parametric theta/phi
and count-based pi fixed within each iteration); only the bookkeeping is
compiled. First call JIT-compiles (~10-15s); subsequent calls run at
compiled speed thanks to cache=True.

Sentiment labels: 0 = positive, 1 = negative, 2 = neutral.
MPQA lexicon seeds initial sentiment assignments at iteration 1 only;
after that, the joint rating-corpus gradient drives sentiment discovery.
"""

import re
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
    Returns doc_words (list of arrays), dictionary, and flat sweep arrays.
    """
    train = train.copy()
    if 'tokens' not in train.columns:
        train['tokens'] = train['reviewText'].apply(clean)
    docs = train.groupby('asin')['tokens'].apply(lambda x: sum(x, [])).reset_index()
    docs.columns = ['asin', 'tokens']

    dictionary = corpora.Dictionary(docs['tokens'])
    dictionary.filter_extremes(keep_n=n_vocab)

    n_items = len(sid2idx)
    doc_words = [np.array([], dtype=np.int64) for _ in range(n_items)]
    seen = np.zeros(n_items, dtype=bool)

    for _, row in docs.iterrows():
        item_idx = sid2idx[row['asin']]
        word_ids = [dictionary.token2id[w] for w in row['tokens']
                    if w in dictionary.token2id]
        doc_words[item_idx] = np.array(word_ids, dtype=np.int64)
        seen[item_idx] = True

    n_d = np.array([len(d) for d in doc_words], dtype=np.int64)
    all_words = np.concatenate(doc_words) if n_d.sum() > 0 \
                else np.array([], dtype=np.int64)
    all_docs = np.repeat(np.arange(n_items, dtype=np.int64), n_d)

    print(f"Corpus: {n_items} items, {int(n_d.sum()):,} tokens, "
          f"vocab {len(dictionary)}, "
          f"{int(seen.sum())}/{n_items} items have reviews")

    return doc_words, dictionary, all_words, all_docs, n_d, seen


# ---------------------------------------------------------------------------
# 2. MPQA lexicon loading and filtering
# ---------------------------------------------------------------------------
def load_mpqa_lexicon(path, dictionary, min_freq=20):
    """
    Parse an MPQA-format .tff file and return {word_id: sentiment_label}
    for vocabulary words with corpus frequency >= min_freq.
    Labels: 0 = positive, 1 = negative. Neutral and conflicting polarities
    are skipped.
    """
    kv = re.compile(r'(\w+)=([^\s]+)')
    raw_polarity = {}

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pairs = dict(kv.findall(line))
            word = pairs.get('word1')
            polarity = pairs.get('priorpolarity')
            if word is None or polarity is None:
                continue
            if polarity not in ('positive', 'negative'):
                continue
            raw_polarity.setdefault(word, set()).add(polarity)

    lexicon = {}
    for word, polarities in raw_polarity.items():
        if len(polarities) != 1:
            continue
        if word not in dictionary.token2id:
            continue
        wid = dictionary.token2id[word]
        if dictionary.dfs.get(wid, 0) < min_freq:
            continue
        polarity = next(iter(polarities))
        lexicon[wid] = 0 if polarity == 'positive' else 1

    return lexicon


# ---------------------------------------------------------------------------
# 3. Softmax (numerically stable) and init helpers
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def build_initial_counts(all_words, all_docs, all_topics, all_sents,
                         n_items, n_vocab, K, S):
    """
    Build count matrices from current (z, l) assignments. Used once at
    initialisation; subsequent iterations maintain these via the sweep.
    """
    N_klI = np.zeros((n_items, S, K), dtype=np.float64)    # doc-sentiment-topic
    N_lkw = np.zeros((S, K, n_vocab), dtype=np.float64)    # sentiment-topic-word
    N_kl  = np.zeros((S, K), dtype=np.float64)             # sentiment-topic
    N_lI  = np.zeros((n_items, S), dtype=np.float64)       # doc-sentiment

    N_total = all_words.shape[0]
    for t in range(N_total):
        w = all_words[t]; d = all_docs[t]
        k = all_topics[t]; l = all_sents[t]
        N_klI[d, l, k] += 1.0
        N_lkw[l, k, w] += 1.0
        N_kl[l, k]     += 1.0
        N_lI[d, l]     += 1.0

    return N_klI, N_lkw, N_kl, N_lI


# ---------------------------------------------------------------------------
# 4. Numba-JIT sweep: joint (z, l) sampling + counts + lk + grad_kappa
# ---------------------------------------------------------------------------
@numba.njit(cache=True, fastmath=True)
def _jst_lfm_sweep(all_words, all_docs,
                   theta, phi, pi_d, Q_block, eq_block,
                   K, S, rand_uniforms,
                   N_klI, N_lkw, N_kl, N_lI,
                   all_topics, all_sents):
    """
    One sequential sweep over the corpus for JST-LFM.

    For each token j (with d = all_docs[j], w = all_words[j]):
        probs[k*S + l] = pi_d[d, l] * theta[d, l, k] * phi[l, k, w]
        (z_j, l_j)     ~ Multinomial(probs / sum probs)   via inverse CDF

        all_topics[j] <- z_j
        all_sents[j]  <- l_j
        N_klI[d, l_j, z_j] += 1
        N_lkw[l_j, z_j, w] += 1
        N_kl[l_j, z_j]     += 1
        N_lI[d, l_j]       += 1
        lk          += log(probs[z_j * S + l_j])
        grad_kappa  += Q_block[d, l_j, z_j] - eq_block[d, l_j]

    theta, phi, pi_d, Q_block, eq_block are held fixed throughout the
    sweep — matching Algorithm 3 of the methodology. Count matrices and
    assignment arrays are modified in place; they must be zeroed (counts)
    by the caller before calling this function.

    Returns
    -------
    lk         : corpus log-likelihood contribution from theta and phi
                 (pi contribution is count-based and has no gradient, so
                 it's NOT added here; caller can track log(pi) separately
                 if needed for monitoring the full likelihood)
    grad_kappa : scalar gradient accumulator for kappa
    """
    N_total = all_words.shape[0]
    probs = np.empty(K * S, dtype=np.float64)
    lk = 0.0
    grad_kappa = 0.0

    for j in range(N_total):
        d = all_docs[j]
        w = all_words[j]

        total = 0.0
        for k in range(K):
            for l in range(S):
                p = pi_d[d, l] * theta[d, l, k] * phi[l, k, w]
                probs[k * S + l] = p
                total += p

        r = rand_uniforms[j] * total
        cum = 0.0
        idx = K * S - 1  # fallback against floating-point underflow
        for ii in range(K * S):
            cum += probs[ii]
            if r < cum:
                idx = ii
                break

        z_new = idx // S
        l_new = idx % S

        all_topics[j] = z_new
        all_sents[j]  = l_new

        N_klI[d, l_new, z_new] += 1.0
        N_lkw[l_new, z_new, w] += 1.0
        N_kl[l_new, z_new]     += 1.0
        N_lI[d, l_new]         += 1.0

        lk         += np.log(probs[idx] + 1e-30)
        grad_kappa += Q_block[d, l_new, z_new] - eq_block[d, l_new]

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
# 6. Top words per (topic, sentiment) from learned psi
# ---------------------------------------------------------------------------
def top_words_per_topic_sentiment(psi, dictionary, top_n=10):
    """
    Top-N highest-probability words for each (topic, sentiment) pair,
    computed from phi = softmax(psi) along the vocabulary axis.

    Returns {(k, l): [(word, prob), ...]}  where l: 0=pos, 1=neg, 2=neu.
    """
    phi = softmax(psi, axis=2)                # (S, K, V)
    S, K, _ = phi.shape
    result = {}
    for k in range(K):
        for l in range(S):
            col = phi[l, k, :]
            top_idx = np.argsort(-col)[:top_n]
            result[(k, l)] = [(dictionary[int(i)], float(col[int(i)]))
                              for i in top_idx]
    return result


# ---------------------------------------------------------------------------
# 7. Core fit: single run with checkpointed validation evaluations
# ---------------------------------------------------------------------------
def fit_jst_lfm(train, valid, all_words, all_docs, n_d, seen,
                lexicon, uid2idx, sid2idx,
                n_vocab, K=10, S=3,
                alpha=5.0, beta=0.01, gamma=(0.1, 1, 10),
                lr=0.005, reg=0.02, mu_corpus=1.0, kappa_init=1.0,
                beta1=0.9, beta2=0.99, eps=1e-8,
                n_epochs=300, seed=42, verbose=False):
    """
    Train JST-LFM with Adam and Numba-JIT sequential Gibbs. Evaluates val MSE
    every epoch; keeps only the best epoch's params.

    Parameter dimensionality is D = S * K (no extra K_star features in
    this implementation; rating prediction uses the full S*K vector).
    """
    import time as _time

    n_users = len(uid2idx)
    n_items = len(sid2idx)
    D = S * K

    gamma_arr = np.asarray(gamma, dtype=np.float64)
    gamma_sum = float(gamma_arr.sum())

    rng = np.random.RandomState(seed)
    mu = train['overall'].mean()

    # --- Parameters ---
    # Q flat (n_items, D=S*K), sentiment-major: Q[i, l*K + k]
    Q   = rng.normal(0, 0.01, (n_items, D)).astype(np.float64)
    P   = rng.normal(0, 0.01, (n_users, D)).astype(np.float64)
    b_u = np.zeros(n_users, dtype=np.float64)
    b_i = np.zeros(n_items, dtype=np.float64)
    psi = rng.normal(0, 0.01, (S, K, n_vocab)).astype(np.float64)
    kappa = float(kappa_init)

    # --- Gibbs assignments: init random topic, lexicon-seeded sentiment ---
    N_total = int(all_words.shape[0])
    all_topics = rng.randint(0, K, size=N_total).astype(np.int64)
    all_sents  = rng.randint(0, S, size=N_total).astype(np.int64)

    if lexicon:
        lex_ids = np.fromiter(lexicon.keys(), dtype=np.int64)
        lex_labels = np.fromiter(lexicon.values(), dtype=np.int64)
        lookup = -np.ones(n_vocab, dtype=np.int64)
        lookup[lex_ids] = lex_labels
        seeded = lookup[all_words]
        mask = seeded >= 0
        all_sents[mask] = seeded[mask]

    # --- Initial count matrices (once at init; maintained per-iteration) ---
    N_klI, N_lkw, N_kl, N_lI = build_initial_counts(
        all_words, all_docs, all_topics, all_sents, n_items, n_vocab, K, S
    )

    # --- Adam state ---
    adam = {name: {'m': np.zeros_like(param), 'v': np.zeros_like(param)}
            for name, param in [('Q', Q), ('P', P), ('b_u', b_u),
                                ('b_i', b_i), ('psi', psi)]}
    adam['kappa'] = {'m': 0.0, 'v': 0.0}

    # --- Rating arrays ---
    users = train['user_idx'].values
    items = train['item_idx'].values
    ratings = train['overall'].values.astype(np.float64)
    n_ratings = len(ratings)
    total_words = int(n_d.sum())

    mse_history = []
    _best_vmse = np.inf
    _best_epoch = -1
    _best_params = None
    _patience_counter = 0
    _prev_vmse = np.inf
    _loss_diff_counter = 0

    for epoch in range(n_epochs):
        t_epoch = _time.time()

        # (A) Compute pi from current sentiment counts (FIXED during this sweep)
        N_d_vec = N_lI.sum(axis=1)                               # (n_items,)
        pi_d = (N_lI + gamma_arr[None, :]) / (N_d_vec[:, None] + gamma_sum)

        # (B) Compute theta, phi from current Q, psi
        Q_block = Q.reshape(n_items, S, K)                       # sentiment-major view
        theta = softmax(kappa * Q_block, axis=2)                 # (n_items, S, K)
        phi   = softmax(psi, axis=2)                             # (S, K, V)
        eq_block = np.sum(theta * Q_block, axis=2)               # (n_items, S)

        # (C) Zero count matrices; Numba sweep resamples all tokens
        N_klI.fill(0.0); N_lkw.fill(0.0); N_kl.fill(0.0); N_lI.fill(0.0)
        rand_uniforms = rng.random_sample(N_total)
        lk, grad_kappa_corpus = _jst_lfm_sweep(
            all_words, all_docs, theta, phi, pi_d, Q_block, eq_block,
            K, S, rand_uniforms,
            N_klI, N_lkw, N_kl, N_lI, all_topics, all_sents,
        )

        # (D) Corpus gradients from counts (vectorised, cheap)
        #     grad_q[i, l, k] = kappa * (N_klI[i, l, k] - N_lI[i, l] * theta[i, l, k])
        #     grad_psi[l, k, w] = N_lkw[l, k, w] - N_kl[l, k] * phi[l, k, w]
        grad_q_corpus_block = kappa * (N_klI - N_lI[:, :, None] * theta)  # (n_items, S, K)
        grad_q_corpus_flat  = grad_q_corpus_block.reshape(n_items, D)     # flat, matches Q
        grad_psi_corpus     = N_lkw - N_kl[:, :, None] * phi              # (S, K, V)

        # (E) Rating gradients
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

        # (F) Combine (corpus gradient normalised by total_words; mu_corpus ~1 neutral)
        grad_Q     = grad_Q_rating - mu_corpus * grad_q_corpus_flat / total_words
        grad_psi   =                - mu_corpus * grad_psi_corpus   / total_words
        grad_kappa =                - mu_corpus * grad_kappa_corpus / total_words

        # (G) Adam updates with bias correction
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

        # (H) Evaluate on validation every epoch; keep best params
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
                  f"dt {dt:.2f}s", flush=True)

    return _best_epoch, _best_vmse, _best_params, mse_history


# ---------------------------------------------------------------------------
# 8. Full pipeline with grid-search tuning
# ---------------------------------------------------------------------------
def run_jst_lfm_tuned(train, valid, test, uid2idx, sid2idx,
                     lexicon_path='MPQA_Subjectivity_Lexicon.tff',
                     K=10, S=3, n_vocab=5000,
                     alpha=5.0, beta=0.01, gamma=(0.1, 1, 10),
                     min_freq=20, seed=42):
    """
    Build the corpus once, load MPQA lexicon once, then grid-search
    (lr, reg, mu_corpus) × checkpoint epochs on validation. Evaluate the
    best configuration on the test set.

    Returns
    -------
    results     : dict with MSE, MAE on test set
    best_info   : dict with best lr, reg, mu, epochs
    tuning_rows : list of dicts, one per (lr, reg, mu, n_epochs, val_mse)
    topic_words : dict {(k, l): [(word, prob), ...]} from best model's psi
    dictionary  : gensim Dictionary
    """
    import time as _time

    print("Building corpus...")
    doc_words, dictionary, all_words, all_docs, n_d, seen = \
        build_corpus(train, sid2idx, n_vocab)
    actual_n_vocab = len(dictionary)

    lexicon = load_mpqa_lexicon(lexicon_path, dictionary, min_freq=min_freq)
    print(f"Lexicon: {len(lexicon):,} sentiment-tagged words (positive/negative)")

    lr_grid  = [0.01, 0.02]
    reg_grid = [0.001]
    mu_grid  = [300.0, 600.0]
    n_epochs = 1000

    n_combos = len(lr_grid) * len(reg_grid) * len(mu_grid)
    print(f"Tuning JST-LFM ({n_combos} combos × {n_epochs} epochs; "
          f"first combo JIT-compiles, ~10-15s)...", flush=True)
    t_tune = _time.time()
    combo_idx = 0

    best_val_mse = np.inf
    best = None  # (lr, reg, mu_c, n_ep, params)
    tuning_rows = []
    best_mse_history = None

    for lr in lr_grid:
        for reg in reg_grid:
            for mu_c in mu_grid:
                combo_idx += 1
                t_combo = _time.time()
                best_ep, best_vmse, params, mse_hist = fit_jst_lfm(
                    train, valid, all_words, all_docs, n_d, seen,
                    lexicon, uid2idx, sid2idx,
                    n_vocab=actual_n_vocab,
                    K=K, S=S, alpha=alpha, beta=beta, gamma=gamma,
                    lr=lr, reg=reg, mu_corpus=mu_c,
                    n_epochs=n_epochs, seed=seed, verbose=False,
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
                print(f"  [{combo_idx}/{n_combos}] lr={lr}, reg={reg}, mu={mu_c}  "
                      f"best val {best_vmse:.4f} @ epoch {best_ep}  "
                      f"(combo {dt_combo:.1f}s, elapsed {elapsed:.1f}s)",
                      flush=True)

    lr, reg, mu_c, n_ep, params = best
    print(f"\n  Best JST-LFM: lr={lr}, reg={reg}, mu={mu_c}, "
          f"epochs={n_ep}, val MSE={best_val_mse:.4f}")

    mu, P, Q, b_u, b_i, psi, kappa = params
    test_pred = predict_ratings(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)
    results['test_pred'] = test_pred
    best_info = {'lr': lr, 'reg': reg, 'mu': mu_c, 'epochs': n_ep}

    topic_words = top_words_per_topic_sentiment(psi, dictionary, top_n=10)

    return results, best_info, tuning_rows, best_mse_history, topic_words, dictionary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results, best_info, _, _, _ = run_jst_lfm_tuned(
        train, valid, test, uid2idx, sid2idx,
    )
    elapsed = time.time() - t

    print(f"\nJST-LFM Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Best: lr={best_info['lr']}, reg={best_info['reg']}, "
          f"mu={best_info['mu']}, epochs={best_info['epochs']}")
    print(f"  Time: {elapsed:.2f}s")