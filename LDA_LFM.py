"""
LDA_LFM.py

LDA-LFM model: jointly optimises latent factor rating prediction with
LDA-based topic modelling. Item factors q_i are linked to topic
distributions theta_i via softmax: theta_{i,k} = softmax(kappa * q_{i,k}).

Prediction: mu + b_u + b_i + p_u^T q_i

The corpus log-likelihood and rating loss are jointly optimised so that
the rating signal informs the topics and the topics regularise the item
factors.
"""

import numpy as np
from gensim import corpora

from data_preprocessing import load_amazon_gz, split_data, clean

# ---------------------------------------------------------------------------
# 1. Build corpus: one document per item (list of word indices)
# ---------------------------------------------------------------------------
def build_corpus(train, sid2idx, n_vocab=5000):
    """
    Group reviews by item, clean, build vocabulary, convert to word indices.
    Returns doc_words (list of arrays) and dictionary.
    """
    train = train.copy()
    train['tokens'] = train['reviewText'].apply(clean)
    docs = train.groupby('asin')['tokens'].apply(lambda x: sum(x, [])).reset_index()
    docs.columns = ['asin', 'tokens']

    # Build vocabulary
    dictionary = corpora.Dictionary(docs['tokens'])
    dictionary.filter_extremes(keep_n=n_vocab)

    # Convert to word index arrays per item
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
    """Compute softmax along given axis with numerical stability."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 3. Gibbs sampling: sample topic for each word
# ---------------------------------------------------------------------------
def gibbs_sample(doc_words, theta, phi, doc_topics, rng):
    """
    For each word in each document, sample a topic from:
        p(z=k) proportional to theta[i,k] * phi[k, w]

    doc_topics is modified in place.
    Returns corpus log-likelihood.
    """
    lk = 0.0
    for i in range(len(doc_words)):
        words = doc_words[i]
        if len(words) == 0:
            continue
        for j in range(len(words)):
            w = words[j]
            probs = theta[i] * phi[:, w]
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs /= prob_sum
            else:
                probs = np.ones_like(probs) / len(probs)
            z = rng.multinomial(1, probs).argmax()
            doc_topics[i][j] = z
            lk += np.log(theta[i, z] * phi[z, w] + 1e-30)
    return lk


# ---------------------------------------------------------------------------
# 4. Compute corpus gradients
# ---------------------------------------------------------------------------
def compute_corpus_gradients(doc_words, doc_topics, theta, phi, q, kappa,
                             n_topics, n_vocab):
    """
    Pre-compute gradients of corpus log-likelihood w.r.t. q, psi, kappa.

    grad_q[i,k] = kappa * (N_{k,i} - N_i * theta_{i,k})
    grad_kappa  = sum_d sum_j (q_{d,z_j} - E_theta[q_d])
    grad_psi[k,w] = sum_d sum_{j: z_j=k} (1[w_j=w] - phi[k,w])
    """
    n_items = len(doc_words)
    grad_q = np.zeros_like(q)       # (n_items, n_topics)
    grad_psi = np.zeros((n_topics, n_vocab))
    grad_kappa = 0.0

    # Expected q under theta for each item: sum_k theta[i,k] * q[i,k]
    eq = np.sum(theta * q, axis=1)  # (n_items,)

    for i in range(n_items):
        words = doc_words[i]
        topics = doc_topics[i]
        n_d = len(words)
        if n_d == 0:
            continue

        # Count topics in this document
        topic_counts = np.bincount(topics, minlength=n_topics).astype(np.float64)

        # grad_q[i,k] = kappa * (N_{k,i} - N_i * theta[i,k])
        grad_q[i] = kappa * (topic_counts - n_d * theta[i])

        # grad_kappa: sum_j (q[i, z_j] - E_theta[q_i])
        for j in range(n_d):
            z = topics[j]
            grad_kappa += q[i, z] - eq[i]

        # grad_psi[k,w]: for each word assigned to topic k
        for j in range(n_d):
            z = topics[j]
            w = words[j]
            grad_psi[z, w] += 1.0  # count

    # Adjust grad_psi: sum (1[w_j=w] - phi[k,w]) for each (k,w)
    # We accumulated the counts, now subtract the expected
    for k in range(n_topics):
        total_k = grad_psi[k].sum()  # total words assigned to topic k
        grad_psi[k] = grad_psi[k] - total_k * phi[k]

    return grad_q, grad_psi, grad_kappa


# ---------------------------------------------------------------------------
# 5. Prediction
# ---------------------------------------------------------------------------
def predict_ratings(data, mu, P, Q, b_u, b_i):
    """Predict ratings for all (user, item) pairs in data."""
    users = data['user_idx'].values
    items = data['item_idx'].values
    pred = mu + b_u[users] + b_i[items] + np.sum(P[users] * Q[items], axis=1)
    return pred


def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE':  np.mean(errors ** 2),
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'MAE':  np.mean(np.abs(errors)),
    }


# ---------------------------------------------------------------------------
# 6. Main training loop
# ---------------------------------------------------------------------------
def fit_lda_lfm(train, valid, doc_words, uid2idx, sid2idx,
                n_topics=10, n_vocab=5000, n_epochs=100,
                lr=0.005, reg=0.02, mu_corpus=1.0, kappa_init=1.0,
                beta1=0.9, beta2=0.999, eps=1e-8, patience=5):
    """
    Train LDA-LFM with Adam optimiser and early stopping.
    """
    n_users = len(uid2idx)
    n_items = len(sid2idx)

    rng = np.random.RandomState(42)
    mu = train['overall'].mean()

    # Initialise model parameters
    Q = rng.normal(0, 0.01, (n_items, n_topics)).astype(np.float64)
    P = rng.normal(0, 0.01, (n_users, n_topics)).astype(np.float64)
    b_u = np.zeros(n_users, dtype=np.float64)
    b_i = np.zeros(n_items, dtype=np.float64)
    psi = rng.normal(0, 0.01, (n_topics, n_vocab)).astype(np.float64)
    kappa = float(kappa_init)

    # Initialise topic assignments randomly
    doc_topics = [rng.randint(0, n_topics, size=len(doc_words[i]))
                  for i in range(n_items)]

    # Adam state for each parameter
    adam = {name: {'m': np.zeros_like(param), 'v': np.zeros_like(param)}
            for name, param in [('Q', Q), ('P', P), ('b_u', b_u),
                                ('b_i', b_i), ('psi', psi)]}
    adam['kappa'] = {'m': 0.0, 'v': 0.0}

    # Training arrays
    users = train['user_idx'].values
    items = train['item_idx'].values
    ratings = train['overall'].values.astype(np.float64)
    n_ratings = len(ratings)

    # Early stopping state
    best_val_mse = np.inf
    best_epoch = 0
    best_params = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):

        # --- Step 1: Compute theta and phi from current parameters ---
        theta = softmax(kappa * Q)                     # (n_items, n_topics)
        phi = softmax(psi)                             # (n_topics, n_vocab)

        # --- Step 2: Gibbs sampling ---
        lk = gibbs_sample(doc_words, theta, phi, doc_topics, rng)

        # --- Step 3: Compute corpus gradients ---
        grad_q_corpus, grad_psi_corpus, grad_kappa_corpus = \
            compute_corpus_gradients(doc_words, doc_topics, theta, phi, Q,
                                     kappa, n_topics, n_vocab)

        # --- Step 4: Compute rating predictions and gradients ---
        pred = predict_ratings(train, mu, P, Q, b_u, b_i)
        err = pred - ratings  # (n_ratings,)
        rating_loss = np.mean(err ** 2)

        # Accumulate rating gradients per user / item (vectorised)
        grad_P = np.zeros_like(P)
        grad_Q_rating = np.zeros_like(Q)
        grad_bu = np.zeros_like(b_u)
        grad_bi = np.zeros_like(b_i)

        err_2 = 2 * err / n_ratings  # (n_ratings,)
        np.add.at(grad_bu, users, err_2)
        np.add.at(grad_bi, items, err_2)
        np.add.at(grad_P, users, err_2[:, None] * Q[items])
        np.add.at(grad_Q_rating, items, err_2[:, None] * P[users])

        # Add regularisation
        grad_bu += 2 * reg * b_u
        grad_bi += 2 * reg * b_i
        grad_P  += 2 * reg * P
        # Q is regularised by the corpus term, not by lambda

        # --- Step 5: Combine rating and corpus gradients for Q ---
        total_words = sum(len(d) for d in doc_words)
        grad_Q = grad_Q_rating - mu_corpus * grad_q_corpus / total_words
        grad_psi = -mu_corpus * grad_psi_corpus / total_words
        grad_kappa = -mu_corpus * grad_kappa_corpus / total_words

        # --- Step 6: Adam updates ---
        def adam_update(name, param, grad):
            adam[name]['m'] = beta1 * adam[name]['m'] + (1 - beta1) * grad
            adam[name]['v'] = beta2 * adam[name]['v'] + (1 - beta2) * grad ** 2
            return param - lr * adam[name]['m'] / (np.sqrt(adam[name]['v']) + eps)

        P = adam_update('P', P, grad_P)
        Q = adam_update('Q', Q, grad_Q)
        b_u = adam_update('b_u', b_u, grad_bu)
        b_i = adam_update('b_i', b_i, grad_bi)
        psi = adam_update('psi', psi, grad_psi)

        # Scalar Adam for kappa
        adam['kappa']['m'] = beta1 * adam['kappa']['m'] + (1 - beta1) * grad_kappa
        adam['kappa']['v'] = beta2 * adam['kappa']['v'] + (1 - beta2) * grad_kappa ** 2
        kappa = kappa - lr * adam['kappa']['m'] / (np.sqrt(adam['kappa']['v']) + eps)

        # --- Step 7: Validation and early stopping ---
        val_pred = predict_ratings(valid, mu, P, Q, b_u, b_i)
        val_mse = np.mean((val_pred - valid['overall'].values) ** 2)

        print(f"  Epoch {epoch+1}/{n_epochs}, "
              f"train MSE: {rating_loss:.4f}, "
              f"valid MSE: {val_mse:.4f}, "
              f"lk: {lk:.1f}, "
              f"kappa: {kappa:.3f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            best_params = (mu, P.copy(), Q.copy(), b_u.copy(), b_i.copy(),
                           psi.copy(), kappa)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}. "
                      f"Best epoch: {best_epoch} (valid MSE: {best_val_mse:.4f})")
                break

    return best_params


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------
def run_lda_lfm(train, valid, test, uid2idx, sid2idx,
                n_topics=10, n_vocab=5000, n_epochs=100,
                lr=0.005, reg=0.02, mu_corpus=1.0, kappa_init=1.0,
                patience=5):
    """Run the full LDA-LFM pipeline."""

    print("Building corpus...")
    doc_words, dictionary = build_corpus(train, sid2idx, n_vocab)

    print("Fitting LDA-LFM...")
    params = fit_lda_lfm(
        train, valid, doc_words, uid2idx, sid2idx,
        n_topics=n_topics, n_vocab=len(dictionary),
        n_epochs=n_epochs, lr=lr, reg=reg,
        mu_corpus=mu_corpus, kappa_init=kappa_init,
        patience=patience
    )

    mu, P, Q, b_u, b_i, psi, kappa = params

    print("Evaluating on test set...")
    test_pred = predict_ratings(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)

    return results, params, dictionary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Electronics_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results, params, dictionary = run_lda_lfm(
        train, valid, test, uid2idx, sid2idx,
        n_topics=10, mu_corpus=1, reg=0.02, n_epochs=50
    )
    elapsed = time.time() - t

    print(f"\nLDA-LFM Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  RMSE: {results['RMSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Time: {elapsed:.2f}s")