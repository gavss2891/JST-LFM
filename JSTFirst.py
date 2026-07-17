"""
JSTFirst.py

JSTFirst baseline: runs the Joint Sentiment/Topic model on the review
corpus to obtain sentiment-specific topic distributions theta_{d,k,l}, then
flattens these to a fixed item feature vector used as fixed item factors in
a latent factor model. Only user factors P and biases are learned; JST and
the rating model are NOT jointly optimised. JST runs once (seeded with the
MPQA subjectivity lexicon), then its output is fed into the rating model as
fixed features.

Prediction: mu + b_u + b_i + p_u^T theta_flat_i

Tuning mirrors LDAFirst: JST is built once, then the LFM second stage is
grid-searched over lr, reg, and n_epochs using the epoch-checkpoint trick.

Gibbs sampler: sequential collapsed Gibbs (Lin & He 2009, Eq. 5) with
per-token exclusion, JIT-compiled via Numba; first call JIT-compiles
(~5-10s).
"""

import re
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numba
from gensim import corpora

from data_preprocessing import load_amazon_gz, split_data, clean


# ---------------------------------------------------------------------------
# 1. Build corpus: one document per item (list of word indices)
# ---------------------------------------------------------------------------
def build_corpus(train, sid2idx, n_vocab=5000):
    """
    Group reviews by item, clean, build vocabulary, convert to word indices.
    Returns doc_words (list of arrays), dictionary, flat arrays for sampling.
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
    Parse an MPQA-format .tff file and return a dict {word_id: sentiment_label}
    for words in our vocabulary with corpus frequency >= min_freq.

    Sentiment labels: 0 = positive, 1 = negative. Neutral words and words with
    conflicting polarities across POS tags are skipped (label stays random at
    Gibbs initialisation time).

    Expected line format (one attribute=value pair per space-separated token):
        type=weaksubj len=1 word1=abdicate pos1=verb stemmed1=y priorpolarity=negative
    """
    kv = re.compile(r'(\w+)=([^\s]+)')
    raw_polarity = {}  # word_str -> set of polarities seen

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pairs = dict(kv.findall(line))
            word = pairs.get('word1')
            polarity = pairs.get('priorpolarity')
            if word is None or polarity is None:
                continue
            if polarity == 'neutral':
                continue
            if polarity not in ('positive', 'negative'):
                continue
            raw_polarity.setdefault(word, set()).add(polarity)

    lexicon = {}
    for word, polarities in raw_polarity.items():
        if len(polarities) != 1:
            continue  # conflicting — skip
        if word not in dictionary.token2id:
            continue
        wid = dictionary.token2id[word]
        if dictionary.dfs.get(wid, 0) < min_freq:
            continue
        polarity = next(iter(polarities))
        lexicon[wid] = 0 if polarity == 'positive' else 1

    return lexicon


# ---------------------------------------------------------------------------
# 3. Numba-JIT sequential collapsed Gibbs sampler (inner sweep)
# ---------------------------------------------------------------------------
@numba.njit(cache=True, fastmath=True)
def _gibbs_sweep(all_words, all_docs, all_topics, all_sents,
                 N_wkl, N_kl, N_kld, N_ld, N_d,
                 K, S, V, alpha, beta, gamma_arr, gamma_sum,
                 rand_uniforms):
    """
    One full Gibbs sweep over all tokens: for each token, decrements its
    current assignment, resamples (topic, sentiment) jointly from the
    collapsed posterior (Lin & He 2009, Eq. 5) via inverse-CDF, then
    increments the new assignment. Count buffers are modified in place.
    Lexicon-seeded tokens are still resampled here — the lexicon only
    biases initialisation.
    """
    N_total = all_words.shape[0]
    prob = np.empty(K * S, dtype=np.float64)

    for t in range(N_total):
        w = all_words[t]
        d = all_docs[t]
        k_old = all_topics[t]
        l_old = all_sents[t]

        # Decrement current assignment
        N_wkl[w, k_old, l_old] -= 1.0
        N_kl[k_old, l_old]     -= 1.0
        N_kld[k_old, l_old, d] -= 1.0
        N_ld[l_old, d]         -= 1.0
        N_d[d]                 -= 1.0

        # Compute joint probability grid over (k, l)
        total = 0.0
        Nd_d = N_d[d] + gamma_sum
        for l in range(S):
            Nld_d = N_ld[l, d]
            denom_klalpha = Nld_d + K * alpha
            f3 = (Nld_d + gamma_arr[l]) / Nd_d
            for k in range(K):
                f1 = (N_wkl[w, k, l] + beta) / (N_kl[k, l] + V * beta)
                f2 = (N_kld[k, l, d] + alpha) / denom_klalpha
                p = f1 * f2 * f3
                prob[k * S + l] = p
                total += p

        # Inverse-CDF sampling
        r = rand_uniforms[t] * total
        cum = 0.0
        sampled = K * S - 1  # fallback guard
        for idx in range(K * S):
            cum += prob[idx]
            if r < cum:
                sampled = idx
                break

        k_new = sampled // S
        l_new = sampled % S

        # Increment new assignment
        N_wkl[w, k_new, l_new] += 1.0
        N_kl[k_new, l_new]     += 1.0
        N_kld[k_new, l_new, d] += 1.0
        N_ld[l_new, d]         += 1.0
        N_d[d]                 += 1.0

        all_topics[t] = k_new
        all_sents[t]  = l_new


# ---------------------------------------------------------------------------
# 4. JST sequential Gibbs sampler (outer wrapper)
# ---------------------------------------------------------------------------
def run_jst_gibbs(all_words, all_docs, n_items, n_vocab, lexicon,
                  K=10, S=3, alpha=5.0, beta=0.01,
                  gamma=(0.01, 5.0, 1.0), n_iter=500, seed=42):
    """
    Sequential collapsed Gibbs sampler for JST (Lin & He 2009), JIT-compiled
    via Numba.

    Sentiment labels: 0 = positive, 1 = negative, 2 = neutral. Lexicon seeds
    positive/negative words at initialisation; all others (and every topic
    assignment) are initialised uniformly at random.

    Returns
    -------
    all_topics : (N_total,) int64
    all_sents  : (N_total,) int64
    """
    rng = np.random.RandomState(seed)
    N_total = len(all_words)
    V = int(n_vocab)
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    gamma_sum = float(gamma_arr.sum())

    # --- Initialisation ---
    all_topics = rng.randint(0, K, size=N_total).astype(np.int64)
    all_sents = rng.randint(0, S, size=N_total).astype(np.int64)

    if lexicon:
        lex_ids = np.fromiter(lexicon.keys(), dtype=np.int64)
        lex_labels = np.fromiter(lexicon.values(), dtype=np.int64)
        lookup = -np.ones(V, dtype=np.int64)
        lookup[lex_ids] = lex_labels
        seeded = lookup[all_words]
        mask = seeded >= 0
        all_sents[mask] = seeded[mask]

    # --- Build initial count matrices once; updated in place during sweeps ---
    N_wkl = np.zeros((V, K, S), dtype=np.float64)
    N_kl  = np.zeros((K, S), dtype=np.float64)
    N_kld = np.zeros((K, S, n_items), dtype=np.float64)
    N_ld  = np.zeros((S, n_items), dtype=np.float64)
    N_d   = np.zeros(n_items, dtype=np.float64)

    for t in range(N_total):
        w = all_words[t]
        d = all_docs[t]
        k = all_topics[t]
        l = all_sents[t]
        N_wkl[w, k, l] += 1.0
        N_kl[k, l]     += 1.0
        N_kld[k, l, d] += 1.0
        N_ld[l, d]     += 1.0
        N_d[d]         += 1.0

    print(f"  Gibbs init done: {N_total:,} tokens, "
          f"V={V}, K={K}, S={S}. Starting sweeps "
          f"(first call JIT-compiles, ~5-10s)...", flush=True)

    # --- Gibbs iterations ---
    import time as _time
    t_start = _time.time()
    report_every = max(1, n_iter // 20)  # ~20 progress lines total

    for it in range(n_iter):
        rand_uniforms = rng.random_sample(N_total)
        _gibbs_sweep(all_words, all_docs, all_topics, all_sents,
                     N_wkl, N_kl, N_kld, N_ld, N_d,
                     K, S, V, float(alpha), float(beta),
                     gamma_arr, gamma_sum, rand_uniforms)

        if it == 0 or (it + 1) % report_every == 0 or (it + 1) == n_iter:
            elapsed = _time.time() - t_start
            per_iter = elapsed / (it + 1)
            eta = per_iter * (n_iter - it - 1)
            print(f"    iter {it+1}/{n_iter}  "
                  f"elapsed {elapsed:6.1f}s  "
                  f"per-iter {per_iter:5.2f}s  "
                  f"eta {eta:6.1f}s",
                  flush=True)

    return all_topics, all_sents


# ---------------------------------------------------------------------------
# 5. Feature extraction: theta_{d,k,l} -> (n_items, S*K) flattened
# ---------------------------------------------------------------------------
def extract_jst_features(all_topics, all_sents, all_docs, n_items,
                         K=10, S=3, alpha=5.0, seen=None):
    """
    Compute theta_{d,k,l} from final Gibbs counts and flatten to a
    sentiment-major (n_items, S*K) feature matrix.

    Ordering:  [theta_{d,0,pos}, ..., theta_{d,K-1,pos},
                theta_{d,0,neg}, ..., theta_{d,K-1,neg},
                theta_{d,0,neu}, ..., theta_{d,K-1,neu}]

    Cold items (no training reviews) are filled with the mean feature vector
    over items that do have reviews.
    """
    flat_kld = (all_topics * S * n_items) + (all_sents * n_items) + all_docs
    N_kld = np.bincount(flat_kld, minlength=K * S * n_items) \
              .reshape(K, S, n_items).astype(np.float64)
    N_ld = N_kld.sum(axis=0)                                      # (S, n_items)

    # theta[k, l, d] = (N_kld[k,l,d] + alpha) / (N_ld[l,d] + K*alpha)
    theta = (N_kld + alpha) / (N_ld[None, :, :] + K * alpha)      # (K, S, n_items)

    # Flatten to (n_items, S*K), sentiment-major
    # features[d, l*K + k] = theta[k, l, d]
    features = theta.transpose(2, 1, 0).reshape(n_items, S * K)

    # Cold-item handling
    if seen is not None and (~seen).any() and seen.any():
        features[~seen] = features[seen].mean(axis=0)

    return features


# ---------------------------------------------------------------------------
# 6. LFM second stage: train with fixed item features, vectorised batch Adam
# ---------------------------------------------------------------------------
def _run_combo_jst_first(args):
    lr, reg, n_epochs, train, valid, features, uid2idx, sid2idx, verbose = args
    best_ep, best_vmse, params, mse_hist = train_lfm_fixed_q_checkpoint(
        train, valid, features, uid2idx, sid2idx,
        lr=lr, reg=reg, n_epochs=n_epochs, verbose=verbose,
    )
    return lr, reg, best_ep, best_vmse, params, mse_hist


def train_lfm_fixed_q_checkpoint(train, valid, features, uid2idx, sid2idx,
                                 lr=0.005, reg=0.02,
                                 beta1=0.9, beta2=0.999, eps=1e-8,
                                 n_epochs=300, verbose=False):
    """
    Train LFM with item factors fixed to `features`. Evaluates val MSE every
    epoch; keeps only the best epoch's params.

    Returns
    -------
    (best_epoch, best_val_mse, best_params, mse_history)
    """
    n_users = len(uid2idx)
    n_items = len(sid2idx)
    n_factors = features.shape[1]

    mu = train['overall'].mean()
    rng = np.random.RandomState(42)
    P = rng.normal(0, 0.01, (n_users, n_factors)).astype(np.float64)
    b_u = np.zeros(n_users, dtype=np.float64)
    b_i = np.zeros(n_items, dtype=np.float64)

    adam = {name: {'m': np.zeros_like(p), 'v': np.zeros_like(p)}
            for name, p in [('P', P), ('b_u', b_u), ('b_i', b_i)]}

    users = train['user_idx'].values
    items = train['item_idx'].values
    ratings = train['overall'].values.astype(np.float64)
    n_ratings = len(ratings)

    valid_users = valid['user_idx'].values
    valid_items = valid['item_idx'].values
    valid_ratings = valid['overall'].values

    mse_history = []
    _best_vmse = np.inf
    _best_epoch = -1
    _best_params = None
    _patience_counter = 0
    _prev_vmse = np.inf
    _loss_diff_counter = 0

    for epoch in range(n_epochs):
        pred = mu + b_u[users] + b_i[items] \
               + np.sum(P[users] * features[items], axis=1)
        err = pred - ratings

        err_2 = 2 * err / n_ratings
        g_bu = np.zeros_like(b_u)
        g_bi = np.zeros_like(b_i)
        g_P = np.zeros_like(P)

        np.add.at(g_bu, users, err_2)
        np.add.at(g_bi, items, err_2)
        np.add.at(g_P, users, err_2[:, None] * features[items])

        g_bu += 2 * reg * b_u
        g_bi += 2 * reg * b_i
        g_P += 2 * reg * P
        # Features are fixed, not regularised.

        t = epoch + 1
        for name, grad in [('b_u', g_bu), ('b_i', g_bi), ('P', g_P)]:
            adam[name]['m'] = beta1 * adam[name]['m'] + (1 - beta1) * grad
            adam[name]['v'] = beta2 * adam[name]['v'] + (1 - beta2) * grad ** 2

        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        b_u = b_u - lr * (adam['b_u']['m'] / bc1) / (np.sqrt(adam['b_u']['v'] / bc2) + eps)
        b_i = b_i - lr * (adam['b_i']['m'] / bc1) / (np.sqrt(adam['b_i']['v'] / bc2) + eps)
        P = P - lr * (adam['P']['m'] / bc1) / (np.sqrt(adam['P']['v'] / bc2) + eps)

        val_pred = mu + b_u[valid_users] + b_i[valid_items] \
                   + np.sum(P[valid_users] * features[valid_items], axis=1)
        train_mse = float(np.mean(err ** 2))
        val_mse = float(np.mean((val_pred - valid_ratings) ** 2))
        mse_history.append((epoch + 1, train_mse, val_mse))

        if epoch >= 200:
            if val_mse < _best_vmse:
                _best_vmse = val_mse
                _best_epoch = epoch + 1
                _best_params = (mu, P.copy(), b_u.copy(), b_i.copy())
                _patience_counter = 0
            else:
                _patience_counter += 1
                if _patience_counter >= 100:
                    break
            if abs(val_mse - _prev_vmse) < 1e-6:
                _loss_diff_counter += 1
                if _loss_diff_counter >= 10:
                    break
            else:
                _loss_diff_counter = 0

        _prev_vmse = val_mse

        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(f"      [JSTFirst lr={lr} reg={reg}] epoch {epoch+1}/{n_epochs}  "
                  f"train MSE {train_mse:.4f}  val MSE {val_mse:.4f}", flush=True)

    return _best_epoch, _best_vmse, _best_params, mse_history


# ---------------------------------------------------------------------------
# 7. Prediction and evaluation
# ---------------------------------------------------------------------------
def predict_lfm_fixed_q(data, mu, P, b_u, b_i, features):
    users = data['user_idx'].values
    items = data['item_idx'].values
    return mu + b_u[users] + b_i[items] \
           + np.sum(P[users] * features[items], axis=1)


def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE': np.mean(errors ** 2),
        'MAE': np.mean(np.abs(errors)),
    }


# ---------------------------------------------------------------------------
# 8. Top words per (topic, sentiment)
# ---------------------------------------------------------------------------
def top_words_per_topic_sentiment(all_words, all_topics, all_sents,
                                  dictionary, K=10, S=3,
                                  beta=0.01, top_n=10):
    """
    Return the top-N highest-probability words for each (topic, sentiment)
    combination, computed from the final Gibbs counts via

        phi[w, k, l] = (N_{w,k,l} + beta) / (N_{k,l} + V*beta)

    Returns
    -------
    dict {(topic_k, sentiment_l): [(word, probability), ...]}
    sentiment_l :  0 = positive, 1 = negative, 2 = neutral
    """
    n_vocab = len(dictionary)
    flat_wkl = (all_words * K * S) + (all_topics * S) + all_sents
    N_wkl = np.bincount(flat_wkl, minlength=n_vocab * K * S) \
              .reshape(n_vocab, K, S).astype(np.float64)
    N_kl = N_wkl.sum(axis=0)                                      # (K, S)

    phi = (N_wkl + beta) / (N_kl + n_vocab * beta)                # (V, K, S)

    result = {}
    for k in range(K):
        for l in range(S):
            col = phi[:, k, l]
            top_idx = np.argsort(-col)[:top_n]
            result[(k, l)] = [(dictionary[int(i)], float(col[int(i)]))
                              for i in top_idx]
    return result


# ---------------------------------------------------------------------------
# 9. Full pipeline with grid search
# ---------------------------------------------------------------------------
def run_jst_first_tuned(train, valid, test, uid2idx, sid2idx,
                        lexicon_path='MPQA_Subjectivity_Lexicon.tff',
                        K=10, S=3, n_vocab=5000,
                        alpha=5.0, beta=0.01, gamma=(0.01, 5.0, 1.0),
                        n_gibbs_iter=500, min_freq=20, seed=42, verbose=False):
    """
    1. Build corpus
    2. Load MPQA lexicon
    3. Run JST sequential Gibbs (once, Numba-JIT)
    4. Extract theta features
    5. Grid-search LFM second stage with epoch-checkpoint trick
    6. Evaluate best config on test set

    Returns
    -------
    results    : dict with MSE, MAE on test set
    best_info  : dict with best lr, reg, mu, epochs
    tuning_rows: list of dicts for every grid point (lr, reg, mu, n_epochs, val_mse)
    topic_words: dict {(k, l): [(word, prob), ...]}
    """
    # --- Stage 1: JST (runs once) ---
    print("Building corpus...")
    doc_words, dictionary, all_words, all_docs, n_d, seen = \
        build_corpus(train, sid2idx, n_vocab)
    actual_n_vocab = len(dictionary)
    n_items = len(sid2idx)

    lexicon = load_mpqa_lexicon(lexicon_path, dictionary, min_freq=min_freq)
    print(f"Lexicon: {len(lexicon):,} sentiment-tagged words (positive/negative)")

    print(f"Running JST ({n_gibbs_iter} Gibbs iterations, "
          f"K={K} topics, S={S} sentiments, Numba sequential)...")
    all_topics, all_sents = run_jst_gibbs(
        all_words, all_docs, n_items, actual_n_vocab, lexicon,
        K=K, S=S, alpha=alpha, beta=beta, gamma=gamma,
        n_iter=n_gibbs_iter, seed=seed,
    )
    print("JST complete.")

    features = extract_jst_features(
        all_topics, all_sents, all_docs, n_items,
        K=K, S=S, alpha=alpha, seen=seen,
    )

    # --- Stage 2: LFM grid search with per-epoch tracking ---
    lr_grid  = [0.01]
    reg_grid = [0.001]
    n_epochs = 5000

    combos = [
        (lr, reg, n_epochs, train, valid, features, uid2idx, sid2idx, verbose)
        for lr in lr_grid for reg in reg_grid
    ]
    n_combos = len(combos)

    best_val_mse = np.inf
    best = None  # (lr, reg, n_ep, params)
    tuning_rows = []
    best_mse_history = None

    print(f"Tuning JSTFirst ({n_combos} combos in parallel)...", flush=True)
    t_tune = _time.time()
    with ProcessPoolExecutor(max_workers=n_combos) as ex:
        futures = {ex.submit(_run_combo_jst_first, c): c for c in combos}
        for i, fut in enumerate(as_completed(futures), 1):
            lr, reg, best_ep, best_vmse, params, mse_hist = fut.result()
            elapsed = _time.time() - t_tune
            print(f"  [{i}/{n_combos}] lr={lr}, reg={reg}  "
                  f"best val {best_vmse:.4f} @ epoch {best_ep}  "
                  f"(elapsed {elapsed:.1f}s)", flush=True)
            for ep, tmse, vmse in mse_hist:
                tuning_rows.append({
                    'lr': lr, 'reg': reg, 'mu': float('nan'),
                    'n_epochs': ep, 'train_mse': tmse, 'val_mse': vmse,
                })
            if best_vmse < best_val_mse:
                best_val_mse = best_vmse
                best = (lr, reg, best_ep, params)
                best_mse_history = mse_hist

    lr, reg, n_ep, (mu, P, b_u, b_i) = best
    print(f"  Best JSTFirst: lr={lr}, reg={reg}, epochs={n_ep}, "
          f"val MSE={best_val_mse:.4f}")

    test_pred = predict_lfm_fixed_q(test, mu, P, b_u, b_i, features)
    results = evaluate(test_pred, test['overall'].values)
    results['test_pred'] = test_pred
    best_info = {'lr': lr, 'reg': reg, 'mu': float('nan'), 'epochs': n_ep}

    topic_words = top_words_per_topic_sentiment(
        all_words, all_topics, all_sents, dictionary, K=K, S=S, beta=beta
    )

    return results, best_info, tuning_rows, best_mse_history, topic_words


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results, _, _, _ = run_jst_first_tuned(
        train, valid, test, uid2idx, sid2idx,
    )
    elapsed = time.time() - t

    print(f"\nJSTFirst Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Time: {elapsed:.2f}s")