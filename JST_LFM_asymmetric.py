"""
JST_LFM_asymmetric.py

Asymmetric variant of JST-LFM (see JST_LFM.py for the base model). Where
the symmetric version uses K topics per sentiment, this variant allows the
number of topics to differ across sentiments via a per-sentiment topic
count vector Ks, stored in a flat sentiment-major layout.

Prediction: mu + b_u + b_i + p_u^T q_i

Gibbs sweep: Numba-JIT sequential pass over tokens with per-sentiment
topic counts Ks, cache=True; first call JIT-compiles.
"""

import re
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Identical to JST_LFM.py.build_corpus.
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
    seen = np.zeros(n_items, dtype=bool)

    for _, row in docs.iterrows():
        item_idx = sid2idx[row['asin']]
        word_ids = [dictionary.token2id[w] for w in row['tokens']
                    if w in dictionary.token2id]
        doc_words[item_idx] = np.array(word_ids, dtype=np.int32)
        seen[item_idx] = True

    n_d = np.array([len(d) for d in doc_words], dtype=np.int32)
    all_words = np.concatenate(doc_words) if n_d.sum() > 0 \
                else np.array([], dtype=np.int32)
    all_docs = np.repeat(np.arange(n_items, dtype=np.int32), n_d)

    print(f"Corpus: {n_items} items, {int(n_d.sum()):,} tokens, "
          f"vocab {len(dictionary)}, "
          f"{int(seen.sum())}/{n_items} items have reviews")

    return doc_words, dictionary, all_words, all_docs, n_d, seen


# ---------------------------------------------------------------------------
# 2. MPQA lexicon loading and filtering (identical to symmetric version)
# ---------------------------------------------------------------------------
def load_mpqa_lexicon(path, dictionary, min_freq=20):
    """
    Parse an MPQA-format .tff file and return {word_id: sentiment_label}
    for vocabulary words with corpus frequency >= min_freq.
    Labels: 0 = positive, 2 = negative (sentiment index 1 is neutral and
    is never seeded from the lexicon). Neutral and conflicting polarities
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
        lexicon[wid] = 0 if polarity == 'positive' else 2

    return lexicon


# ---------------------------------------------------------------------------
# 3. Softmax (numerically stable) along an explicit axis range
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_per_sentiment_block(arr_flat, Ks, offsets):
    """
    Apply softmax independently within each sentiment's topic slice.

    arr_flat : (n_docs, D) where D = sum(Ks)
    Ks       : (S,) ints
    offsets  : (S+1,) cumulative offsets so block l is arr_flat[:, offsets[l]:offsets[l+1]]

    Returns array of same shape with each sentiment block softmax-normalised.
    """
    out = np.empty_like(arr_flat)
    for l in range(len(Ks)):
        block = arr_flat[:, offsets[l]:offsets[l + 1]]
        out[:, offsets[l]:offsets[l + 1]] = softmax(block, axis=1)
    return out


def softmax_psi_per_sentiment_topic(psi_flat, Ks, offsets, n_vocab):
    """
    Apply softmax along the vocabulary axis for each (sentiment, topic-in-l)
    cell. psi_flat has shape (D, V) where D = sum(Ks); each row is a
    (sentiment, topic) cell's word logits. Output has same shape.

    Cells are ordered sentiment-major: rows offsets[l]..offsets[l+1]-1 are
    the K_l word distributions for sentiment l.
    """
    return softmax(psi_flat, axis=1)  # row-wise softmax over V


# ---------------------------------------------------------------------------
# 4. Initial counts in flat (sentiment-major) layout
# ---------------------------------------------------------------------------
def build_initial_counts(all_words, all_docs, all_topics_flat, all_sents,
                         n_items, n_vocab, Ks, offsets):
    """
    Build count matrices from current (z_flat, l) assignments.

    all_topics_flat : (N_total,) int64 -- the FLAT topic index in
                      [offsets[l], offsets[l+1]). I.e. each token's topic
                      already encodes its sentiment via which block it lies in.
                      The l index in `all_sents` is redundant (derivable from
                      all_topics_flat) but kept for clarity and faster lookup.

    Returns
    -------
    N_kI  : (n_items, D) doc-topic counts (D = sum(Ks)). For sentiment l,
            columns offsets[l]:offsets[l+1] are that sentiment's topic counts.
    N_kw  : (D, n_vocab) topic-word counts (sentiment encoded by row offset).
    N_k   : (D,) total tokens per (sentiment, topic) cell.
    N_lI  : (n_items, S) doc-sentiment counts. Just collapses N_kI per block.
    """
    D = int(np.sum(Ks))
    S = len(Ks)
    N_kI = np.zeros((n_items, D), dtype=np.float64)
    N_kw = np.zeros((D, n_vocab), dtype=np.float64)
    N_k  = np.zeros(D, dtype=np.float64)
    N_lI = np.zeros((n_items, S), dtype=np.float64)

    N_total = all_words.shape[0]
    for t in range(N_total):
        w = all_words[t]; d = all_docs[t]
        kf = all_topics_flat[t]   # already flat-indexed (sentiment-major)
        l  = all_sents[t]
        N_kI[d, kf] += 1.0
        N_kw[kf, w] += 1.0
        N_k[kf]     += 1.0
        N_lI[d, l]  += 1.0

    return N_kI, N_kw, N_k, N_lI


# ---------------------------------------------------------------------------
# 5. Numba-JIT sweep: joint (z, l) sampling with asymmetric Ks
# ---------------------------------------------------------------------------
@numba.njit(cache=True, fastmath=True)
def _jst_lfm_asym_sweep(all_words, all_docs,
                        theta_flat, phi_flat, pi_d, Q_flat, eq_block,
                        Ks, offsets, D,
                        rand_uniforms,
                        N_kI, N_kw, N_k, N_lI,
                        all_topics_flat, all_sents):
    """
    One sequential Gibbs sweep with per-sentiment topic counts Ks: for each
    token, samples a flat (sentiment, topic) index via inverse-CDF over
    pi_d * theta_flat * phi_flat, decodes its sentiment from the offsets,
    updates the count matrices in place, and accumulates the corpus
    log-likelihood and kappa gradient.

    theta_flat : (n_items, D) sentiment-major-blocked softmax over Q
    phi_flat   : (D, V) per-(sentiment,topic) word distribution
    Q_flat     : (n_items, D) flat factor matrix (matches Q)
    eq_block   : (n_items, S) E_theta[q] within each sentiment block
    """
    N_total = all_words.shape[0]
    probs = np.empty(D, dtype=np.float64)
    lk = 0.0
    grad_kappa = 0.0

    for j in range(N_total):
        d = all_docs[j]
        w = all_words[j]

        # Build the full (l, k) probability vector in flat layout.
        # For each sentiment block l, columns offsets[l]..offsets[l+1]-1
        # carry that block's K_l topic probabilities under sentiment l.
        total = 0.0
        for l in range(Ks.shape[0]):
            pi_dl = pi_d[d, l]
            base = offsets[l]
            K_l = Ks[l]
            for k in range(K_l):
                kf = base + k
                p = pi_dl * theta_flat[d, kf] * phi_flat[kf, w]
                probs[kf] = p
                total += p

        # Inverse-CDF sample
        r = rand_uniforms[j] * total
        cum = 0.0
        kf_new = D - 1   # fallback against floating-point underflow
        for kf in range(D):
            cum += probs[kf]
            if r < cum:
                kf_new = kf
                break

        # Decode sentiment from flat index by binary lookup over offsets.
        # With S=3 a linear scan is fastest.
        l_new = 0
        for l in range(Ks.shape[0]):
            if kf_new < offsets[l + 1]:
                l_new = l
                break

        all_topics_flat[j] = kf_new
        all_sents[j]       = l_new

        N_kI[d, kf_new] += 1.0
        N_kw[kf_new, w] += 1.0
        N_k[kf_new]     += 1.0
        N_lI[d, l_new]  += 1.0

        lk         += np.log(probs[kf_new] + 1e-30)
        grad_kappa += Q_flat[d, kf_new] - eq_block[d, l_new]

    return lk, grad_kappa


# ---------------------------------------------------------------------------
# 6. Prediction and evaluation (identical to symmetric version)
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
# 7. Top words per (topic, sentiment) from learned psi (asymmetric layout)
# ---------------------------------------------------------------------------
def top_words_per_topic_sentiment(psi_flat, Ks, offsets, dictionary, top_n=10):
    """
    Return {(k_in_l, l): [(word, prob), ...]} for each valid (l, k_in_l) pair.
    psi_flat has shape (D, V) where D = sum(Ks); rows are sentiment-blocked.

    The output dict's keys preserve compatibility with the symmetric version's
    write_jst_topics_csv: (k, l) where k is the topic-within-sentiment index.
    """
    phi_flat = softmax(psi_flat, axis=1)
    result = {}
    for l in range(len(Ks)):
        K_l = Ks[l]
        base = offsets[l]
        for k_in_l in range(K_l):
            kf = base + k_in_l
            col = phi_flat[kf, :]
            top_idx = np.argsort(-col)[:top_n]
            result[(k_in_l, l)] = [(dictionary[int(i)], float(col[int(i)]))
                                   for i in top_idx]
    return result


# ---------------------------------------------------------------------------
# 8. Core fit: single run with checkpointed validation evaluations
# ---------------------------------------------------------------------------
def fit_jst_lfm_asym(train, valid, all_words, all_docs, n_d, seen,
                     lexicon, uid2idx, sid2idx,
                     n_vocab, Ks=(9, 3, 3), k_star=0,
                     alpha=5.0, beta=0.01, gamma=(0.1, 1, 10),
                     lr=0.005, reg=0.02, mu_corpus=1.0, kappa_init=1.0,
                     beta1=0.9, beta2=0.99, eps=1e-8,
                     n_epochs=300, seed=42, verbose=False):
    """
    Train asymmetric JST-LFM with Adam and Numba-JIT sequential Gibbs.
    Evaluates val MSE every epoch; keeps only the best epoch's params.

    Ks : tuple of S ints, one per sentiment label.
         Sentiment order: index 0 = positive, 1 = neutral, 2 = negative.
         Default (9, 3, 3) gives D = 15 to capacity-match LDA-LFM(K=15).

    k_star : int, default 0
        Number of extra rating-only latent dimensions appended to Q and P.
        Cols 0..D-1 (where D = sum(Ks)) are topic-linked (corpus gradient,
        no lambda reg). Cols D..D+k_star-1 are rating-only (rating gradient
        + lambda reg, never enter the Gibbs sweep).
        k_star=0 reproduces the base model exactly.

    The factor matrix Q has shape (n_items, D + k_star). Rating prediction
    uses the full inner product over all D + k_star dimensions. The topic
    model operates in S separate softmax blocks of size K_l each, joined
    into the flat sentiment-major layout described at the top of this module.
    """
    import time as _time

    Ks = np.asarray(Ks, dtype=np.int64)
    S = len(Ks)
    D = int(Ks.sum())
    Dfull = D + k_star               # full factor dimensionality
    offsets = np.concatenate([[0], np.cumsum(Ks)]).astype(np.int64)  # (S+1,)

    n_users = len(uid2idx)
    n_items = len(sid2idx)

    gamma_arr = np.asarray(gamma, dtype=np.float64)
    if gamma_arr.shape[0] != S:
        raise ValueError(f"gamma must have length S={S}, got {gamma_arr.shape[0]}")
    gamma_sum = float(gamma_arr.sum())

    rng = np.random.RandomState(seed)
    mu = train['overall'].mean()

    # --- Parameters: full dimensionality Dfull = D + k_star ---
    # Q, P flat (n_*, Dfull), sentiment-major blocked layout + optional extra block.
    Q   = rng.normal(0, 0.01, (n_items, Dfull)).astype(np.float64)
    P   = rng.normal(0, 0.01, (n_users, Dfull)).astype(np.float64)
    b_u = np.zeros(n_users, dtype=np.float64)
    b_i = np.zeros(n_items, dtype=np.float64)
    psi = rng.normal(0, 0.01, (D, n_vocab)).astype(np.float64)   # topic block only
    kappa = float(kappa_init)

    # --- Gibbs assignments: random topic-within-sentiment + lexicon-seeded l ---
    N_total = int(all_words.shape[0])
    all_sents = rng.randint(0, S, size=N_total).astype(np.int32)
    if lexicon:
        lex_ids = np.fromiter(lexicon.keys(), dtype=np.int64)
        lex_labels = np.fromiter(lexicon.values(), dtype=np.int64)
        lookup = -np.ones(n_vocab, dtype=np.int32)
        lookup[lex_ids] = lex_labels
        seeded = lookup[all_words]
        mask = seeded >= 0
        all_sents[mask] = seeded[mask]

    # Random topic within each token's assigned sentiment block, then convert
    # to the flat sentiment-major index.
    all_topics_flat = np.empty(N_total, dtype=np.int32)
    for l in range(S):
        mask_l = (all_sents == l)
        n_l = int(mask_l.sum())
        if n_l > 0:
            within = rng.randint(0, Ks[l], size=n_l)
            all_topics_flat[mask_l] = offsets[l] + within

    # --- Initial count matrices (once at init; maintained by sweeps) ---
    N_kI, N_kw, N_k, N_lI = build_initial_counts(
        all_words, all_docs, all_topics_flat, all_sents,
        n_items, n_vocab, Ks, offsets,
    )

    # --- Adam state (sized to Dfull for Q and P) ---
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

        # (A) pi from current sentiment counts (FIXED during this sweep)
        N_d_vec = N_lI.sum(axis=1)                                  # (n_items,)
        pi_d = (N_lI + gamma_arr[None, :]) / (N_d_vec[:, None] + gamma_sum)

        # (B) theta and phi from the TOPIC-LINKED block of Q only (first D cols)
        Q_topic    = Q[:, :D]                                            # (n_items, D)
        theta_flat = softmax_per_sentiment_block(kappa * Q_topic, Ks, offsets)
        phi_flat   = softmax(psi, axis=1)                               # (D, V)

        # eq_block[i, l] = sum_{k in block l} theta[i, kf] * Q_topic[i, kf]
        eq_block = np.zeros((n_items, S), dtype=np.float64)
        for l in range(S):
            sl = slice(offsets[l], offsets[l + 1])
            eq_block[:, l] = np.sum(theta_flat[:, sl] * Q_topic[:, sl], axis=1)

        # (C) Zero counts; Numba sweep resamples all tokens.
        #     Pass contiguous copies of the D-column topic views so Numba
        #     sees C-contiguous arrays.
        N_kI.fill(0.0); N_kw.fill(0.0); N_k.fill(0.0); N_lI.fill(0.0)
        rand_uniforms = rng.random_sample(N_total)
        theta_c   = np.ascontiguousarray(theta_flat)
        Q_topic_c = np.ascontiguousarray(Q_topic)
        lk, grad_kappa_corpus = _jst_lfm_asym_sweep(
            all_words, all_docs,
            theta_c, phi_flat, pi_d, Q_topic_c, eq_block,
            Ks, offsets, D,
            rand_uniforms,
            N_kI, N_kw, N_k, N_lI,
            all_topics_flat, all_sents,
        )

        # (D) Corpus gradients from counts (topic-linked block only)
        # grad_q[i, kf]  = kappa * (N_kI[i, kf] - N_lI[i, l(kf)] * theta_flat[i, kf])
        # grad_psi[kf, w] = N_kw[kf, w] - N_k[kf] * phi_flat[kf, w]
        grad_q_corpus = np.empty((n_items, D), dtype=np.float64)
        for l in range(S):
            sl = slice(offsets[l], offsets[l + 1])
            grad_q_corpus[:, sl] = kappa * (
                N_kI[:, sl] - N_lI[:, l:l + 1] * theta_flat[:, sl]
            )
        grad_psi_corpus = N_kw - N_k[:, None] * phi_flat                  # (D, V)

        # (E) Rating gradients over the FULL Dfull dimensions
        pred = predict_ratings(train, mu, P, Q, b_u, b_i)
        err = pred - ratings
        err_2 = 2 * err / n_ratings

        grad_P        = np.zeros_like(P)            # (n_users, Dfull)
        grad_Q_rating = np.zeros_like(Q)            # (n_items, Dfull)
        grad_bu       = np.zeros_like(b_u)
        grad_bi       = np.zeros_like(b_i)

        np.add.at(grad_bu, users, err_2)
        np.add.at(grad_bi, items, err_2)
        np.add.at(grad_P,  users, err_2[:, None] * Q[items])
        np.add.at(grad_Q_rating, items, err_2[:, None] * P[users])

        grad_bu += 2 * reg * b_u
        grad_bi += 2 * reg * b_i
        grad_P  += 2 * reg * P                      # all Dfull dims of P regularised
        # Topic-linked block of Q is NOT regularised by lambda; the corpus term
        # plays that role. Extra block is regularised below.

        # (F) Assemble the full Q gradient.
        #     - Topic-linked block (first D cols): rating grad + corpus grad,
        #       NOT lambda-regularised (corpus term plays that role).
        #     - Extra block (last k_star cols): rating grad + lambda, NO corpus.
        grad_Q = grad_Q_rating.copy()                               # (n_items, Dfull)
        grad_Q[:, :D] -= mu_corpus * grad_q_corpus / total_words
        if k_star > 0:
            grad_Q[:, D:] += 2 * reg * Q[:, D:]

        grad_psi   = - mu_corpus * grad_psi_corpus   / total_words
        grad_kappa = - mu_corpus * grad_kappa_corpus / total_words

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

        # (H) Validation evaluation; keep best params
        train_mse = float(np.mean(err ** 2))
        val_pred = predict_ratings(valid, mu, P, Q, b_u, b_i)
        val_mse = float(np.mean((val_pred - valid['overall'].values) ** 2))
        mse_history.append((epoch + 1, train_mse, val_mse))

        if epoch >= 1:
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
                if _patience_counter >= 300:
                    break

        _prev_vmse = val_mse

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            dt = _time.time() - t_epoch
            kstar_str = f" K*={k_star}" if k_star > 0 else ""
            print(f"      [JST-LFM-asym{kstar_str} lr={lr} reg={reg} mu={mu_corpus}] epoch {epoch+1}/{n_epochs}  "
                  f"train MSE {train_mse:.4f}  val MSE {val_mse:.4f}  "
                  f"kappa {kappa:+.3f}  lk {lk:.1f}  dt {dt:.2f}s", flush=True)

    return _best_epoch, _best_vmse, _best_params, mse_history


# ---------------------------------------------------------------------------
# 9. Single-configuration runner (fixed Ks, fixed hyperparameters, one k_star)
# ---------------------------------------------------------------------------
def run_jst_lfm_asym_kstar(train, valid, test, uid2idx, sid2idx,
                           lr, reg, mu_corpus,
                           Ks=(9, 3, 3), k_star=0,
                           lexicon_path='MPQA_Subjectivity_Lexicon.tff',
                           n_vocab=5000, alpha=5.0, beta=0.01, gamma=(0.1, 1, 10),
                           min_freq=20, n_epochs=5000, seed=42,
                           corpus=None, lexicon=None, dictionary=None,
                           verbose=False):
    """
    Train asymmetric JST-LFM with k_star extra features at fixed Ks and
    fixed (lr, reg, mu_corpus); evaluate on the test set.

    For the K* sweep, Ks should be the best allocation chosen for the dataset
    and (lr, reg, mu_corpus) the values selected at k_star = 0. The corpus,
    lexicon, and dictionary can be built once and passed in to avoid
    rebuilding them for every K*.

    `corpus`, if provided, is the tuple
    (all_words, all_docs, n_d, seen) returned by build_corpus (last 4 items).

    Returns
    -------
    results     : dict with MSE, MAE, test_pred
    best_info   : dict with lr, reg, mu, Ks, k_star, epochs, val_mse
    mse_history : list of (epoch, train_mse, val_mse)
    topic_words : dict {(k_in_l, l): [(word, prob), ...]} from the best model
    """
    if corpus is None or lexicon is None or dictionary is None:
        print(f"Building corpus (asymmetric Ks={tuple(Ks)})...")
        doc_words, dictionary, all_words, all_docs, n_d, seen = \
            build_corpus(train, sid2idx, n_vocab)
        lexicon = load_mpqa_lexicon(lexicon_path, dictionary, min_freq=min_freq)
    else:
        all_words, all_docs, n_d, seen = corpus
    actual_n_vocab = len(dictionary)

    Ks_arr = np.asarray(Ks, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(Ks_arr)]).astype(np.int64)

    best_ep, best_vmse, params, mse_hist = fit_jst_lfm_asym(
        train, valid, all_words, all_docs, n_d, seen,
        lexicon, uid2idx, sid2idx,
        n_vocab=actual_n_vocab, Ks=Ks, k_star=k_star,
        alpha=alpha, beta=beta, gamma=gamma,
        lr=lr, reg=reg, mu_corpus=mu_corpus,
        n_epochs=n_epochs, seed=seed, verbose=verbose,
    )

    mu, P, Q, b_u, b_i, psi, kappa = params
    test_pred = predict_ratings(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)
    results['test_pred'] = test_pred

    best_info = {'lr': lr, 'reg': reg, 'mu': mu_corpus, 'Ks': tuple(Ks),
                 'k_star': k_star, 'epochs': best_ep, 'val_mse': best_vmse}
    topic_words = top_words_per_topic_sentiment(
        psi, Ks_arr, offsets, dictionary, top_n=10
    )

    print(f"  JST-LFM-asym Ks={tuple(Ks)} K*={k_star}: val MSE={best_vmse:.4f} "
          f"@ epoch {best_ep}, test MSE={results['MSE']:.4f}")

    return results, best_info, mse_hist, topic_words


# ---------------------------------------------------------------------------
# 10. Full pipeline with grid-search tuning
# ---------------------------------------------------------------------------
def _run_combo_jst_lfm_asym(args):
    (lr, reg, mu_c, train, valid, all_words, all_docs, n_d, seen,
     lexicon, uid2idx, sid2idx, n_vocab, Ks, alpha, beta, gamma,
     n_epochs, seed, verbose) = args
    best_ep, best_vmse, params, mse_hist = fit_jst_lfm_asym(
        train, valid, all_words, all_docs, n_d, seen,
        lexicon, uid2idx, sid2idx,
        n_vocab=n_vocab, Ks=Ks, alpha=alpha, beta=beta, gamma=gamma,
        lr=lr, reg=reg, mu_corpus=mu_c,
        n_epochs=n_epochs, seed=seed, verbose=verbose,
    )
    return lr, reg, mu_c, best_ep, best_vmse, params, mse_hist


def run_jst_lfm_asym_tuned(train, valid, test, uid2idx, sid2idx,
                           lexicon_path='MPQA_Subjectivity_Lexicon.tff',
                           Ks=(9, 3, 3), n_vocab=5000,
                           alpha=5.0, beta=0.01, gamma=(0.1, 1, 10),
                           min_freq=20, seed=42, verbose=False,
                           n_workers=None, mu_grid=None):
    """
    Build corpus once, load MPQA lexicon once, then grid-search
    (lr, reg, mu_corpus) × checkpoint epochs on validation. Evaluate the
    best configuration on the test set.

    Mirror of run_jst_lfm_tuned but with per-sentiment topic counts Ks.

    Returns
    -------
    results     : dict with MSE, MAE on test set
    best_info   : dict with best lr, reg, mu, epochs, plus 'Ks' for the record
    tuning_rows : list of dicts, one per (lr, reg, mu, n_epochs, val_mse)
    topic_words : dict {(k_in_l, l): [(word, prob), ...]} from best model's psi
    dictionary  : gensim Dictionary
    """
    print(f"Building corpus (asymmetric Ks={tuple(Ks)})...")
    doc_words, dictionary, all_words, all_docs, n_d, seen = \
        build_corpus(train, sid2idx, n_vocab)
    actual_n_vocab = len(dictionary)

    lexicon = load_mpqa_lexicon(lexicon_path, dictionary, min_freq=min_freq)
    print(f"Lexicon: {len(lexicon):,} sentiment-tagged words (positive/negative)")

    Ks_arr = np.asarray(Ks, dtype=np.int64)
    S = len(Ks_arr)
    D = int(Ks_arr.sum())
    offsets = np.concatenate([[0], np.cumsum(Ks_arr)]).astype(np.int64)
    sent_names = ['K_pos', 'K_neu', 'K_neg']
    sent_str = ', '.join(f'{sent_names[l] if l < len(sent_names) else f"K_s{l}"}={Ks[l]}' for l in range(S))
    print(f"  Per-sentiment topic counts: {sent_str}")
    print(f"  Total Q dim D = {D}, sentiment-major offsets = {tuple(offsets.tolist())}")

    lr_grid  = [0.01]
    reg_grid = [0.001]
    if mu_grid is None:
        mu_grid = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1000]
    n_epochs = 5000

    combos = sorted([
        (lr, reg, mu_c, train, valid, all_words, all_docs, n_d, seen,
         lexicon, uid2idx, sid2idx, actual_n_vocab,
         Ks, alpha, beta, gamma, n_epochs, seed, verbose)
        for lr in lr_grid for reg in reg_grid for mu_c in mu_grid
    ], key=lambda c: -c[2])
    n_combos = len(combos)

    best_val_mse = np.inf
    best = None
    tuning_rows = []
    best_mse_history = None

    _n_workers = n_workers if n_workers is not None else n_combos
    print(f"Tuning JST-LFM-asym ({n_combos} combos, n_workers={_n_workers}; "
          f"Numba cache=True, JIT on first run only)...", flush=True)
    t_tune = _time.time()
    i = 0
    # Submit in batches of _n_workers to avoid pickling all combos simultaneously.
    with ProcessPoolExecutor(max_workers=_n_workers) as ex:
        for batch_start in range(0, n_combos, _n_workers):
            batch = combos[batch_start:batch_start + _n_workers]
            futures = {ex.submit(_run_combo_jst_lfm_asym, c): c for c in batch}
            for fut in as_completed(futures):
                i += 1
                lr, reg, mu_c, best_ep, best_vmse, params, mse_hist = fut.result()
                elapsed = _time.time() - t_tune
                print(f"  [{i}/{n_combos}] lr={lr}, reg={reg}, mu={mu_c}  "
                      f"best val {best_vmse:.4f} @ epoch {best_ep}  "
                      f"(elapsed {elapsed:.1f}s)", flush=True)
                for ep, tmse, vmse in mse_hist:
                    tuning_rows.append({
                        'lr': lr, 'reg': reg, 'mu': mu_c,
                        'n_epochs': ep, 'train_mse': tmse, 'val_mse': vmse,
                    })
                if best_vmse < best_val_mse:
                    best_val_mse = best_vmse
                    best = (lr, reg, mu_c, best_ep, params)
                    best_mse_history = mse_hist

    lr, reg, mu_c, n_ep, params = best
    print(f"\n  Best JST-LFM-asym Ks={tuple(Ks)}: lr={lr}, reg={reg}, "
          f"mu={mu_c}, epochs={n_ep}, val MSE={best_val_mse:.4f}")

    mu, P, Q, b_u, b_i, psi, kappa = params
    test_pred = predict_ratings(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)
    results['test_pred'] = test_pred
    best_info = {'lr': lr, 'reg': reg, 'mu': mu_c, 'epochs': n_ep,
                 'Ks': tuple(Ks)}

    topic_words = top_words_per_topic_sentiment(
        psi, Ks_arr, offsets, dictionary, top_n=10
    )

    return results, best_info, tuning_rows, best_mse_history, topic_words, dictionary


# ---------------------------------------------------------------------------
# 10. Ks-search: try multiple (K_pos, K_neg) allocations, pick best by val MSE
# ---------------------------------------------------------------------------
def run_jst_lfm_asym_ks_tuned(train, valid, test, uid2idx, sid2idx,
                                  lexicon_path='MPQA_Subjectivity_Lexicon.tff',
                                  ks_list=((5, 5), (7, 8), (9, 6), (11, 4), (13, 2)),
                                  n_vocab=5000,
                                  alpha=5.0, beta=0.01, gamma=(0.1, 1),
                                  min_freq=20, seed=42, verbose=False,
                                  n_workers=None, mu_grid=None):
    """
    2-sentiment JST-LFM-asym with Ks search. Builds corpus once, then for
    each (K_pos, K_neg) in ks_list runs the full mu-grid tuning and tracks
    the best validation MSE. The (Ks, mu) combination with the lowest
    validation MSE is used to evaluate on the test set.

    Returns same signature as run_jst_lfm_asym_tuned, with best_info['Ks']
    recording the winning allocation.
    """
    print("Building corpus (2-sentiment Ks sweep)...")
    doc_words, dictionary, all_words, all_docs, n_d, seen = \
        build_corpus(train, sid2idx, n_vocab)
    actual_n_vocab = len(dictionary)

    lexicon = load_mpqa_lexicon(lexicon_path, dictionary, min_freq=min_freq)
    print(f"Lexicon: {len(lexicon):,} sentiment-tagged words (positive/negative)")

    lr_grid = [0.01]
    reg_grid = [0.001]
    if mu_grid is None:
        mu_grid = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1000]
    n_epochs = 5000

    overall_best_val = np.inf
    overall_best = None       # (lr, reg, mu_c, n_ep, params)
    overall_best_Ks = None
    all_tuning_rows = []
    overall_best_mse_hist = None

    for Ks in ks_list:
        Ks_arr = np.asarray(Ks, dtype=np.int64)
        D = int(Ks_arr.sum())
        print(f"\n  Trying Ks={Ks} (K_pos={Ks[0]}, K_neg={Ks[1]}, D={D})...", flush=True)

        combos = sorted([
            (lr, reg, mu_c, train, valid, all_words, all_docs, n_d, seen,
             lexicon, uid2idx, sid2idx, actual_n_vocab,
             Ks, alpha, beta, gamma, n_epochs, seed, verbose)
            for lr in lr_grid for reg in reg_grid for mu_c in mu_grid
        ], key=lambda c: -c[2])
        n_combos = len(combos)

        w = n_workers if n_workers is not None else n_combos
        print(f"  Tuning {n_combos} combos (n_workers={w})...", flush=True)
        t_tune = _time.time()
        i = 0
        ks_best_val = np.inf
        ks_best = None
        ks_best_mse_hist = None

        with ProcessPoolExecutor(max_workers=w) as ex:
            for batch_start in range(0, n_combos, w):
                batch = combos[batch_start:batch_start + w]
                futures = {ex.submit(_run_combo_jst_lfm_asym, c): c for c in batch}
                for fut in as_completed(futures):
                    i += 1
                    lr, reg, mu_c, best_ep, best_vmse, params, mse_hist = fut.result()
                    elapsed = _time.time() - t_tune
                    print(f"    [{i}/{n_combos}] Ks={Ks} lr={lr}, reg={reg}, mu={mu_c}  "
                          f"best val {best_vmse:.4f} @ epoch {best_ep}  "
                          f"(elapsed {elapsed:.1f}s)", flush=True)
                    for ep, tmse, vmse in mse_hist:
                        all_tuning_rows.append({
                            'lr': lr, 'reg': reg, 'mu': mu_c,
                            'n_epochs': ep, 'train_mse': tmse, 'val_mse': vmse,
                        })
                    if best_vmse < ks_best_val:
                        ks_best_val = best_vmse
                        ks_best = (lr, reg, mu_c, best_ep, params)
                        ks_best_mse_hist = mse_hist

        print(f"  Ks={Ks}: best val MSE = {ks_best_val:.4f}", flush=True)
        if ks_best_val < overall_best_val:
            overall_best_val = ks_best_val
            overall_best = ks_best
            overall_best_Ks = Ks
            overall_best_mse_hist = ks_best_mse_hist

    print(f"\n  Best: Ks={overall_best_Ks}, val MSE={overall_best_val:.4f}")

    lr, reg, mu_c, n_ep, params = overall_best
    Ks_arr = np.asarray(overall_best_Ks, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(Ks_arr)]).astype(np.int64)

    mu, P, Q, b_u, b_i, psi, kappa = params
    test_pred = predict_ratings(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)
    results['test_pred'] = test_pred
    best_info = {'lr': lr, 'reg': reg, 'mu': mu_c, 'epochs': n_ep,
                 'Ks': tuple(overall_best_Ks)}

    topic_words = top_words_per_topic_sentiment(
        psi, Ks_arr, offsets, dictionary, top_n=10
    )

    return results, best_info, all_tuning_rows, overall_best_mse_hist, topic_words, dictionary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results, best_info, _, _, _, _ = run_jst_lfm_asym_tuned(
        train, valid, test, uid2idx, sid2idx, Ks=(9, 3, 3),
    )
    elapsed = time.time() - t

    print(f"\nJST-LFM-asym Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Best: Ks={best_info['Ks']}, lr={best_info['lr']}, "
          f"reg={best_info['reg']}, mu={best_info['mu']}, "
          f"epochs={best_info['epochs']}")
    print(f"  Time: {elapsed:.2f}s")