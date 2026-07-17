"""
LFM.py

Latent Factor Model with Adam optimiser and grid-search hyperparameter tuning.

Prediction: mu + b_u + b_i + p_u^T q_i

Both user factors P and item factors Q are learned from ratings.
A small grid search over lr, reg, and n_epochs is performed on the
validation set; the best model is then evaluated on the test set.
"""

import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from data_preprocessing import load_amazon_gz, split_data

# ---------------------------------------------------------------------------
# 1. Predict and evaluate
# ---------------------------------------------------------------------------
def predict_lfm(data, mu, P, Q, b_u, b_i):
    users = data['user_idx'].values
    items = data['item_idx'].values
    return mu + b_u[users] + b_i[items] + np.sum(P[users] * Q[items], axis=1)


def evaluate(predictions, true_ratings):
    errors = predictions - true_ratings
    return {
        'MSE': np.mean(errors ** 2),
        'MAE': np.mean(np.abs(errors)),
    }


# ---------------------------------------------------------------------------
# 2. Fit LFM model
# ---------------------------------------------------------------------------
def _fit_lfm(train, valid, uid2idx, sid2idx, n_factors,
             lr, reg, n_epochs=300,
             beta1=0.9, beta2=0.999, eps=1e-8, verbose=False):
    """Train LFM with Adam, evaluating val MSE every epoch. Returns best epoch's params."""
    n_users = len(uid2idx)
    n_items = len(sid2idx)
    mu = train['overall'].mean()

    rng = np.random.RandomState(42)
    P = rng.normal(0, 0.01, (n_users, n_factors)).astype(np.float64)
    Q = rng.normal(0, 0.01, (n_items, n_factors)).astype(np.float64)
    b_u = np.zeros(n_users, dtype=np.float64)
    b_i = np.zeros(n_items, dtype=np.float64)

    adam = {name: {'m': np.zeros_like(p), 'v': np.zeros_like(p)}
            for name, p in [('P', P), ('Q', Q), ('b_u', b_u), ('b_i', b_i)]}

    users = train['user_idx'].values
    items = train['item_idx'].values
    ratings = train['overall'].values.astype(np.float64)
    n_ratings = len(ratings)

    mse_history = []
    _best_vmse = np.inf
    _best_epoch = -1
    _best_params = None
    _patience_counter = 0
    _prev_vmse = np.inf
    _loss_diff_counter = 0

    for epoch in range(n_epochs):
        pred = mu + b_u[users] + b_i[items] + np.sum(P[users] * Q[items], axis=1)
        err = pred - ratings
        err_2 = 2 * err / n_ratings

        g_bu = np.zeros_like(b_u)
        g_bi = np.zeros_like(b_i)
        g_P  = np.zeros_like(P)
        g_Q  = np.zeros_like(Q)

        np.add.at(g_bu, users, err_2)
        np.add.at(g_bi, items, err_2)
        np.add.at(g_P,  users, err_2[:, None] * Q[items])
        np.add.at(g_Q,  items, err_2[:, None] * P[users])

        g_bu += 2 * reg * b_u
        g_bi += 2 * reg * b_i
        g_P  += 2 * reg * P
        g_Q  += 2 * reg * Q

        t = epoch + 1
        for name, grad in [('b_u', g_bu), ('b_i', g_bi), ('P', g_P), ('Q', g_Q)]:
            adam[name]['m'] = beta1 * adam[name]['m'] + (1 - beta1) * grad
            adam[name]['v'] = beta2 * adam[name]['v'] + (1 - beta2) * grad ** 2

        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        b_u = b_u - lr * (adam['b_u']['m'] / bc1) / (np.sqrt(adam['b_u']['v'] / bc2) + eps)
        b_i = b_i - lr * (adam['b_i']['m'] / bc1) / (np.sqrt(adam['b_i']['v'] / bc2) + eps)
        P   = P   - lr * (adam['P']['m']   / bc1) / (np.sqrt(adam['P']['v']   / bc2) + eps)
        Q   = Q   - lr * (adam['Q']['m']   / bc1) / (np.sqrt(adam['Q']['v']   / bc2) + eps)

        train_mse = float(np.mean(err ** 2))
        val_pred = predict_lfm(valid, mu, P, Q, b_u, b_i)
        val_mse = float(np.mean((val_pred - valid['overall'].values) ** 2))
        mse_history.append((epoch + 1, train_mse, val_mse))

        if epoch >= 200:
            if val_mse < _best_vmse:
                _best_vmse = val_mse
                _best_epoch = epoch + 1
                _best_params = (mu, P.copy(), Q.copy(), b_u.copy(), b_i.copy())
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
            print(f"      [LFM lr={lr} reg={reg}] epoch {epoch+1}/{n_epochs}  "
                  f"train MSE {train_mse:.4f}  val MSE {val_mse:.4f}", flush=True)

    return _best_epoch, _best_vmse, _best_params, mse_history


# ---------------------------------------------------------------------------
# 3. Full pipeline with grid-search tuning
# ---------------------------------------------------------------------------
def _run_combo_lfm(args):
    lr, reg, n_epochs, train, valid, uid2idx, sid2idx, n_factors, verbose = args
    best_ep, best_vmse, params, mse_hist = _fit_lfm(
        train, valid, uid2idx, sid2idx, n_factors,
        lr=lr, reg=reg, n_epochs=n_epochs, verbose=verbose,
    )
    return lr, reg, best_ep, best_vmse, params, mse_hist


def run_lfm_tuned(train, valid, test, uid2idx, sid2idx, n_factors=10, verbose=False, n_workers=None):
    """Grid search over lr, reg, n_epochs on validation set, evaluate on test.

    Returns
    -------
    results          : dict with MSE, MAE on test set
    best_info        : dict with best lr, reg, mu (NaN), epochs
    tuning_rows      : list of dicts (one per combo × epoch) for CSV logging
    best_mse_history : list of (epoch, val_mse) for the best combo
    """
    lr_grid  = [0.01]
    reg_grid = [0.001]
    n_epochs = 5000

    combos = [
        (lr, reg, n_epochs, train, valid, uid2idx, sid2idx, n_factors, verbose)
        for lr in lr_grid for reg in reg_grid
    ]
    n_combos = len(combos)

    best_val_mse = np.inf
    best = None
    tuning_rows = []
    best_mse_history = None

    _n_workers = n_workers if n_workers is not None else n_combos
    print(f"Tuning LFM ({n_combos} combos, n_workers={_n_workers})...", flush=True)
    t_tune = _time.time()
    with ProcessPoolExecutor(max_workers=_n_workers) as ex:
        futures = {ex.submit(_run_combo_lfm, c): c for c in combos}
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

    lr, reg, n_ep, (mu, P, Q, b_u, b_i) = best
    print(f"  Best LFM: lr={lr}, reg={reg}, epochs={n_ep}, val MSE={best_val_mse:.4f}")

    test_pred = predict_lfm(test, mu, P, Q, b_u, b_i)
    results = evaluate(test_pred, test['overall'].values)
    results['test_pred'] = test_pred
    best_info = {'lr': lr, 'reg': reg, 'mu': float('nan'), 'epochs': n_ep}
    return results, best_info, tuning_rows, best_mse_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    DATA_PATH = '/Users/gavinshao/Desktop/Master Thesis/Code/Data/reviews_Beauty_5.json.gz'

    data = load_amazon_gz(DATA_PATH)
    train, valid, test, uid2idx, sid2idx = split_data(data, seed=42)

    t = time.time()
    results, _, _ = run_lfm_tuned(train, valid, test, uid2idx, sid2idx, n_factors=10)
    elapsed = time.time() - t

    print(f"\nLFM Results:")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Time: {elapsed:.2f}s")
