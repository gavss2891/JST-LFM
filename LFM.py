"""
LFM.py

Latent Factor Model with Adam optimiser and grid-search hyperparameter tuning.

Prediction: mu + b_u + b_i + p_u^T q_i

Both user factors P and item factors Q are learned from ratings.
A small grid search over lr, reg, and n_epochs is performed on the
validation set; the best model is then evaluated on the test set.
"""

import numpy as np

from data_preprocessing import load_amazon_gz, split_data


# ---------------------------------------------------------------------------
# 1. Train LFM with Adam
# ---------------------------------------------------------------------------
def train_lfm(train, uid2idx, sid2idx, n_factors=10, n_epochs=35,
              lr=0.005, reg=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
    """Train LFM with Adam. Returns mu, P, Q, b_u, b_i."""
    n_users = len(uid2idx)
    n_items = len(sid2idx)
    mu = train['overall'].mean()

    rng = np.random.RandomState(42)
    P = rng.normal(0, 0.01, (n_users, n_factors)).astype(np.float64)
    Q = rng.normal(0, 0.01, (n_items, n_factors)).astype(np.float64)
    b_u = np.zeros(n_users, dtype=np.float64)
    b_i = np.zeros(n_items, dtype=np.float64)

    # Adam state
    adam = {name: {'m': np.zeros_like(p), 'v': np.zeros_like(p)}
            for name, p in [('P', P), ('Q', Q), ('b_u', b_u), ('b_i', b_i)]}

    users = train['user_idx'].values
    items = train['item_idx'].values
    ratings = train['overall'].values.astype(np.float64)
    n_ratings = len(ratings)

    for epoch in range(n_epochs):
        pred = mu + b_u[users] + b_i[items] + np.sum(P[users] * Q[items], axis=1)
        err = pred - ratings

        # Gradients (normalised by n_ratings)
        err_2 = 2 * err / n_ratings
        g_bu = np.zeros_like(b_u)
        g_bi = np.zeros_like(b_i)
        g_P = np.zeros_like(P)
        g_Q = np.zeros_like(Q)

        np.add.at(g_bu, users, err_2)
        np.add.at(g_bi, items, err_2)
        np.add.at(g_P, users, err_2[:, None] * Q[items])
        np.add.at(g_Q, items, err_2[:, None] * P[users])

        g_bu += 2 * reg * b_u
        g_bi += 2 * reg * b_i
        g_P += 2 * reg * P
        g_Q += 2 * reg * Q

        # Adam updates
        for name, param, grad in [('b_u', b_u, g_bu), ('b_i', b_i, g_bi),
                                   ('P', P, g_P), ('Q', Q, g_Q)]:
            adam[name]['m'] = beta1 * adam[name]['m'] + (1 - beta1) * grad
            adam[name]['v'] = beta2 * adam[name]['v'] + (1 - beta2) * grad ** 2

        b_u = b_u - lr * adam['b_u']['m'] / (np.sqrt(adam['b_u']['v']) + eps)
        b_i = b_i - lr * adam['b_i']['m'] / (np.sqrt(adam['b_i']['v']) + eps)
        P = P - lr * adam['P']['m'] / (np.sqrt(adam['P']['v']) + eps)
        Q = Q - lr * adam['Q']['m'] / (np.sqrt(adam['Q']['v']) + eps)

    return mu, P, Q, b_u, b_i


# ---------------------------------------------------------------------------
# 2. Predict and evaluate
# ---------------------------------------------------------------------------
def predict_lfm(data, mu, P, Q, b_u, b_i):
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
# 3. Full pipeline with grid-search tuning
# ---------------------------------------------------------------------------
def run_lfm_tuned(train, valid, test, uid2idx, sid2idx, n_factors=10):
    """Grid search over lr, reg, n_epochs on validation set, evaluate on test."""
    lr_grid = [0.005, 0.01, 0.05]
    reg_grid = [0.02, 1.0, 10.0]
    epoch_grid = [5, 20, 50]

    best_val_mse = np.inf
    best_params = None

    for lr in lr_grid:
        for reg in reg_grid:
            for n_epochs in epoch_grid:
                mu, P, Q, b_u, b_i = train_lfm(
                    train, uid2idx, sid2idx, n_factors, n_epochs, lr, reg
                )
                val_pred = predict_lfm(valid, mu, P, Q, b_u, b_i)
                val_mse = np.mean((val_pred - valid['overall'].values) ** 2)

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_params = (lr, reg, n_epochs, mu, P, Q, b_u, b_i)

    lr, reg, n_epochs, mu, P, Q, b_u, b_i = best_params
    print(f"  Best LFM: lr={lr}, reg={reg}, epochs={n_epochs}, val MSE={best_val_mse:.4f}")

    test_pred = predict_lfm(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results = run_lfm_tuned(train, valid, test, uid2idx, sid2idx, n_factors=10)
    elapsed = time.time() - t

    print(f"\nLFM Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Time: {elapsed:.2f}s")
