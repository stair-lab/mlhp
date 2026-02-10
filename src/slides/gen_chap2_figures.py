"""Generate figures for Chapter 2 (Learning) lecture slides."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(2601)

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def auc_from_scores(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    cmp = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return cmp + 0.5 * ties


def collect_pairs(Y):
    r, c = np.where(~np.isnan(Y))
    upper = r < c
    return r[upper], c[upper], Y[r[upper], c[upper]].astype(float)


def fit_regularized_bt(r_idx, c_idx, y_obs, M, lam=0.0, lr=0.05, epochs=200):
    V = np.zeros(M)
    for _ in range(epochs):
        H = V[r_idx] - V[c_idx]
        p = sigmoid(H)
        err = y_obs - p
        grad = np.zeros(M)
        np.add.at(grad, r_idx, err)
        np.add.at(grad, c_idx, -err)
        grad -= lam * V
        V += lr * grad
    return V


# ── Shared data ────────────────────────────────────────────────────────
M = 30
V_true = rng.normal(0, 1, M)
diff = V_true[:, None] - V_true[None, :]
P = sigmoid(diff)

Y_BT = np.full((M, M), np.nan)
triu_idx = np.triu_indices(M, k=1)
randu = rng.random(size=triu_idx[0].shape[0])
wins = (randu < P[triu_idx])
Y_BT[triu_idx] = wins
Y_BT[(triu_idx[1], triu_idx[0])] = 1.0 - wins

triu_r, triu_c = np.triu_indices(M, k=1)
valid_mask = ~np.isnan(Y_BT[triu_r, triu_c])
triu_r, triu_c = triu_r[valid_mask], triu_c[valid_mask]

n_pairs = triu_r.shape[0]
n_train = int(0.8 * n_pairs)
idx = rng.choice(n_pairs, size=n_train, replace=False)
train_pairs = np.zeros(n_pairs, dtype=bool)
train_pairs[idx] = True

Y_train = np.full_like(Y_BT, np.nan, dtype=float)
Y_test = np.full_like(Y_BT, np.nan, dtype=float)
r_tr, c_tr = triu_r[train_pairs], triu_c[train_pairs]
Y_train[r_tr, c_tr] = Y_BT[r_tr, c_tr]
Y_train[c_tr, r_tr] = 1.0 - Y_BT[r_tr, c_tr]
r_te, c_te = triu_r[~train_pairs], triu_c[~train_pairs]
Y_test[r_te, c_te] = Y_BT[r_te, c_te]
Y_test[c_te, r_te] = 1.0 - Y_BT[r_te, c_te]
np.fill_diagonal(Y_train, np.nan)
np.fill_diagonal(Y_test, np.nan)

r_tr_fit, c_tr_fit, y_tr_fit = collect_pairs(Y_train)
r_te_fit, c_te_fit, y_te_fit = collect_pairs(Y_test)


# ── Figure 1: Train/Test Split ─────────────────────────────────────────
def fig_train_test_split():
    Y_combined = np.full((M, M), np.nan)
    Y_combined[~np.isnan(Y_train)] = Y_train[~np.isnan(Y_train)]
    Y_combined[~np.isnan(Y_test)] = Y_test[~np.isnan(Y_test)] + 2

    colors = ['#4575b4', '#d73027', '#1a9850', '#fdae61']
    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(np.ma.masked_invalid(Y_combined), vmin=0, vmax=3, cmap=cmap)
    ax.set_title("Train/Test Split of Preference Matrix")
    ax.set_xlabel("Items")
    ax.set_ylabel("Items")
    legend_patches = [
        mpatches.Patch(color='#4575b4', label='Train: 0'),
        mpatches.Patch(color='#d73027', label='Train: 1'),
        mpatches.Patch(color='#1a9850', label='Test: 0'),
        mpatches.Patch(color='#fdae61', label='Test: 1'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))
    fig.savefig(OUT / "chap2_train_test_split.png")
    plt.close(fig)
    print("  [1/10] chap2_train_test_split.png")


# ── Figure 2: MLE Training AUC ─────────────────────────────────────────
def fig_mle_training_auc():
    epochs = 100
    lr = 0.01
    train_auc_hist = []
    V_hat = np.zeros(M, dtype=float)

    for t in range(epochs):
        H = V_hat[r_tr_fit] - V_hat[c_tr_fit]
        p = sigmoid(H)
        err = y_tr_fit - p
        grad = np.zeros_like(V_hat)
        np.add.at(grad, r_tr_fit, err)
        np.add.at(grad, c_tr_fit, -err)
        V_hat += lr * grad
        train_auc_hist.append(auc_from_scores(H, y_tr_fit))

    s_te = V_hat[r_te_fit] - V_hat[c_te_fit]
    test_auc = auc_from_scores(s_te, y_te_fit)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, epochs + 1), train_auc_hist, label="Train AUC")
    ax.hlines(test_auc, xmin=0, xmax=epochs, linestyle="--", color="orange", label="Test AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("MLE Training: AUC Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "chap2_mle_training_auc.png")
    plt.close(fig)
    print("  [2/10] chap2_mle_training_auc.png")
    return V_hat


# ── Figure 3: Learned vs True ──────────────────────────────────────────
def fig_learned_vs_true(V_hat):
    V_hat_norm = (V_hat - V_hat.mean()) / (V_hat.std() + 1e-12)
    A = np.vstack([V_hat_norm, np.ones_like(V_hat_norm)]).T
    a, b = np.linalg.lstsq(A, V_true, rcond=None)[0]
    V_hat_aligned = a * V_hat_norm + b

    fig, ax = plt.subplots()
    ax.scatter(V_true, V_hat, s=30, c="red", alpha=0.6, label="Before alignment")
    ax.scatter(V_true, V_hat_aligned, s=30, c="blue", alpha=0.6, label="After alignment")
    lims = [V_true.min() - 0.5, V_true.max() + 0.5]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.set_xlabel("True V")
    ax.set_ylabel("Estimated V")
    corr = np.corrcoef(V_true, V_hat_aligned)[0, 1]
    ax.set_title(f"Learned vs True Utilities (r = {corr:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "chap2_learned_vs_true.png")
    plt.close(fig)
    print("  [3/10] chap2_learned_vs_true.png")


# ── Figure 4: MCMC Trace + Posterior ───────────────────────────────────
def fig_mcmc_trace_posterior():
    def logpost(v, r_idx, c_idx, y_obs, prior_var=1.0):
        s = v[r_idx] - v[c_idx]
        ll = (y_obs * -np.log1p(np.exp(-s)) + (1 - y_obs) * -np.log1p(np.exp(s))).sum()
        lp = -0.5 * np.dot(v, v) / prior_var
        return ll + lp

    steps = 10000
    prop_scale = 0.05
    rng_mcmc = np.random.default_rng(3)
    v_cur = rng_mcmc.normal(scale=0.1, size=M)
    v_cur -= v_cur.mean()
    lp_cur = logpost(v_cur, r_tr_fit, c_tr_fit, y_tr_fit)

    trace = []
    acc = 0
    for t in range(steps):
        v_prop = v_cur.copy()
        j = rng_mcmc.integers(M)
        v_prop[j] += rng_mcmc.normal(scale=prop_scale)
        v_prop -= v_prop.mean()
        lp_prop = logpost(v_prop, r_tr_fit, c_tr_fit, y_tr_fit)
        if np.log(rng_mcmc.random()) < (lp_prop - lp_cur):
            v_cur, lp_cur = v_prop, lp_prop
            acc += 1
        trace.append(v_cur.copy())

    trace = np.array(trace)

    fig, (ax_trace, ax_hist) = plt.subplots(1, 2, width_ratios=[4, 1],
                                             sharey=True, layout='constrained')
    fig.get_layout_engine().set(wspace=0.05)
    ax_trace.plot(trace[:, 0], lw=0.5, color='steelblue')
    ax_trace.set_xlabel("Iteration")
    ax_trace.set_ylabel(r"$v_0$")
    ax_trace.set_title(r"Trace of $v_0$")

    ax_hist.hist(trace[:, 0], bins=50, density=True, orientation='horizontal',
                 color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_hist.set_xlabel("Density")
    ax_hist.tick_params(labelleft=False)
    ax_hist.set_title("Posterior")

    fig.savefig(OUT / "chap2_mcmc_trace_posterior.png")
    plt.close(fig)
    print("  [4/10] chap2_mcmc_trace_posterior.png")


# ── Figure 5: GP Posterior ─────────────────────────────────────────────
def fig_gp_posterior():
    def rbf_kernel(X1, X2, sigma_f=1.0, length_scale=0.5):
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return sigma_f**2 * np.exp(-sq_dist / (2 * length_scale**2))

    def true_reward(x):
        return np.sin(2 * x) + 0.5 * x

    np.random.seed(42)
    n_comp = 30
    X_A = np.random.uniform(-3, 3, n_comp).reshape(-1, 1)
    X_B = np.random.uniform(-3, 3, n_comp).reshape(-1, 1)
    r_A = true_reward(X_A.flatten())
    r_B = true_reward(X_B.flatten())
    y = (np.random.rand(n_comp) < sigmoid(r_A - r_B)).astype(float)

    X_all = np.vstack([X_A, X_B])
    X_unique, inv_idx = np.unique(X_all.flatten(), return_inverse=True)
    X_unique = X_unique.reshape(-1, 1)
    m = len(X_unique)
    idx_A = inv_idx[:n_comp]
    idx_B = inv_idx[n_comp:]

    K = rbf_kernel(X_unique, X_unique) + 1e-4 * np.eye(m)
    K_inv = np.linalg.inv(K)

    r = np.zeros(m)
    for _ in range(20):
        d = r[idx_A] - r[idx_B]
        p = sigmoid(d)
        grad_ll = np.zeros(m)
        W = np.zeros(m)
        for i in range(n_comp):
            residual = y[i] - p[i]
            grad_ll[idx_A[i]] += residual
            grad_ll[idx_B[i]] -= residual
            W[idx_A[i]] += p[i] * (1 - p[i])
            W[idx_B[i]] += p[i] * (1 - p[i])
        grad_total = grad_ll - K_inv @ r
        H = -np.diag(W) - K_inv
        r = r - np.linalg.solve(H, grad_total)

    Sigma_post = np.linalg.inv(K_inv + np.diag(W))
    X_grid = np.linspace(-3, 3, 100).reshape(-1, 1)
    K_star = rbf_kernel(X_grid, X_unique)
    mu_pred = K_star @ K_inv @ r
    K_star_star = rbf_kernel(X_grid, X_grid)
    var_pred = np.diag(K_star_star - K_star @ (K_inv - K_inv @ Sigma_post @ K_inv) @ K_star.T)
    std_pred = np.sqrt(np.maximum(var_pred, 0))

    fig, ax = plt.subplots()
    ax.fill_between(X_grid.flatten(), mu_pred - 2 * std_pred, mu_pred + 2 * std_pred,
                    alpha=0.3, color='blue', label='95% CI')
    ax.plot(X_grid, mu_pred, 'b-', lw=2, label='GP posterior mean')
    ax.plot(X_grid, true_reward(X_grid), 'k--', lw=2, label='True reward')
    ax.scatter(X_A[y == 1], np.full(int(y.sum()), -2.5), c='green', marker='^', s=50, label='Preferred A')
    ax.scatter(X_B[y == 0], np.full(int((1 - y).sum()), -2.5), c='red', marker='v', s=50, label='Preferred B')
    ax.set_xlabel('x')
    ax.set_ylabel('r(x)')
    ax.set_title(f'GP Preference Learning ({n_comp} comparisons)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "chap2_gp_posterior.png")
    plt.close(fig)
    print("  [5/10] chap2_gp_posterior.png")


# ── Figure 6: Validation Curve ─────────────────────────────────────────
def fig_validation_curve():
    rng_reg = np.random.default_rng(42)
    M_s = 10
    V_s = rng_reg.normal(0, 1, M_s)
    n_tr = 30
    all_p = [(i, j) for i in range(M_s) for j in range(i + 1, M_s)]
    tr_idx = rng_reg.choice(len(all_p), n_tr, replace=False)
    tr_p = [all_p[i] for i in tr_idx]
    te_p = [p for p in all_p if p not in tr_p]

    Y_tr_s = np.full((M_s, M_s), np.nan)
    for i, j in tr_p:
        pw = sigmoid(V_s[i] - V_s[j])
        o = 1.0 if rng_reg.random() < pw else 0.0
        Y_tr_s[i, j], Y_tr_s[j, i] = o, 1.0 - o

    Y_te_s = np.full((M_s, M_s), np.nan)
    for i, j in te_p:
        pw = sigmoid(V_s[i] - V_s[j])
        o = 1.0 if rng_reg.random() < pw else 0.0
        Y_te_s[i, j], Y_te_s[j, i] = o, 1.0 - o

    r_tr_s, c_tr_s, y_tr_s = collect_pairs(Y_tr_s)
    r_te_s, c_te_s, y_te_s = collect_pairs(Y_te_s)

    lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
    train_aucs, test_aucs = [], []
    for lam in lambdas:
        Vf = fit_regularized_bt(r_tr_s, c_tr_s, y_tr_s, M_s, lam=lam)
        train_aucs.append(auc_from_scores(Vf[r_tr_s] - Vf[c_tr_s], y_tr_s))
        test_aucs.append(auc_from_scores(Vf[r_te_s] - Vf[c_te_s], y_te_s))

    fig, ax = plt.subplots()
    ax.semilogx(lambdas, train_aucs, 'o-', label='Train AUC', markersize=8)
    ax.semilogx(lambdas, test_aucs, 's-', label='Test AUC', markersize=8)
    ax.set_xlabel(r'Regularization Strength $\lambda$')
    ax.set_ylabel('AUC')
    ax.set_title('Validation Curve: Effect of Regularization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "chap2_validation_curve.png")
    plt.close(fig)
    print("  [6/10] chap2_validation_curve.png")

    return M_s, r_tr_s, c_tr_s, y_tr_s, r_te_s, c_te_s, y_te_s


# ── Figure 7: Early Stopping ──────────────────────────────────────────
def fig_early_stopping(M_s, r_tr_s, c_tr_s, y_tr_s, r_te_s, c_te_s, y_te_s):
    V = np.zeros(M_s)
    tr_hist, val_hist = [], []
    for _ in range(300):
        H = V[r_tr_s] - V[c_tr_s]
        p = sigmoid(H)
        err = y_tr_s - p
        grad = np.zeros(M_s)
        np.add.at(grad, r_tr_s, err)
        np.add.at(grad, c_tr_s, -err)
        V += 0.01 * grad
        tr_hist.append(auc_from_scores(V[r_tr_s] - V[c_tr_s], y_tr_s))
        val_hist.append(auc_from_scores(V[r_te_s] - V[c_te_s], y_te_s))

    best_ep = int(np.argmax(val_hist))

    fig, ax = plt.subplots()
    ax.plot(tr_hist, label='Train AUC', alpha=0.7)
    ax.plot(val_hist, label='Validation AUC', alpha=0.7)
    ax.axvline(best_ep, color='red', linestyle='--', label=f'Best epoch: {best_ep}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('Early Stopping')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "chap2_early_stopping.png")
    plt.close(fig)
    print("  [7/10] chap2_early_stopping.png")


# ── Figure 8: CV Hyperparameter Tuning ────────────────────────────────
def fig_cv_hyperparam(M_s, r_tr_s, c_tr_s, y_tr_s):
    def kfold_cv(r_idx, c_idx, y_obs, M, k=5, lam=0.0):
        rng_cv = np.random.default_rng(123)
        n = len(r_idx)
        fold_ids = np.arange(n) % k
        rng_cv.shuffle(fold_ids)
        scores = []
        for fold in range(k):
            tm = fold_ids != fold
            vm = fold_ids == fold
            Vf = fit_regularized_bt(r_idx[tm], c_idx[tm], y_obs[tm], M, lam=lam)
            s_val = Vf[r_idx[vm]] - Vf[c_idx[vm]]
            scores.append(auc_from_scores(s_val, y_obs[vm]))
        return np.array(scores)

    lambdas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    means, stds = [], []
    for lam in lambdas:
        sc = kfold_cv(r_tr_s, c_tr_s, y_tr_s, M_s, k=5, lam=lam)
        means.append(sc.mean())
        stds.append(sc.std())

    fig, ax = plt.subplots()
    ax.errorbar(lambdas, means, yerr=stds, fmt='o-', capsize=5, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel(r'Regularization Strength $\lambda$')
    ax.set_ylabel('Cross-Validation AUC')
    ax.set_title('Hyperparameter Tuning (5-Fold CV)')
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "chap2_cv_hyperparam.png")
    plt.close(fig)
    print("  [8/10] chap2_cv_hyperparam.png")


# ── Figure 9: GD vs Adam ──────────────────────────────────────────────
def fig_gd_vs_adam(M_s, r_tr_s, c_tr_s, y_tr_s):
    epochs = 150
    # Adam
    V = np.zeros(M_s)
    m_v, v_v = np.zeros(M_s), np.zeros(M_s)
    loss_adam = []
    for t in range(1, epochs + 1):
        H = V[r_tr_s] - V[c_tr_s]
        p = sigmoid(H)
        err = y_tr_s - p
        grad = np.zeros(M_s)
        np.add.at(grad, r_tr_s, err)
        np.add.at(grad, c_tr_s, -err)
        m_v = 0.9 * m_v + 0.1 * grad
        v_v = 0.999 * v_v + 0.001 * grad**2
        mh = m_v / (1 - 0.9**t)
        vh = v_v / (1 - 0.999**t)
        V += 0.1 * mh / (np.sqrt(vh) + 1e-8)
        ll = (y_tr_s * np.log(p + 1e-12) + (1 - y_tr_s) * np.log(1 - p + 1e-12)).sum()
        loss_adam.append(-ll)

    # GD
    V2 = np.zeros(M_s)
    loss_gd = []
    for _ in range(epochs):
        H = V2[r_tr_s] - V2[c_tr_s]
        p = sigmoid(H)
        err = y_tr_s - p
        grad = np.zeros(M_s)
        np.add.at(grad, r_tr_s, err)
        np.add.at(grad, c_tr_s, -err)
        V2 += 0.05 * grad
        ll = (y_tr_s * np.log(p + 1e-12) + (1 - y_tr_s) * np.log(1 - p + 1e-12)).sum()
        loss_gd.append(-ll)

    fig, ax = plt.subplots()
    ax.plot(loss_gd, label='Gradient Descent (lr=0.05)', alpha=0.8)
    ax.plot(loss_adam, label='Adam (lr=0.1)', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Negative Log-Likelihood')
    ax.set_title('Optimization: GD vs Adam')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.savefig(OUT / "chap2_gd_vs_adam.png")
    plt.close(fig)
    print("  [9/10] chap2_gd_vs_adam.png")


# ── Figure 10: LLM Comparison ─────────────────────────────────────────
def fig_llm_comparison():
    rng_llm = np.random.default_rng(2024)
    n_resp = 50
    emb_dim = 8
    embs = rng_llm.normal(0, 1, (n_resp, emb_dim))
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    true_w = rng_llm.normal(0, 1, emb_dim)
    true_w[0], true_w[1] = 2.0, -1.5
    true_u = embs @ true_w

    n_cmp = 200
    pairs, labels = [], []
    for _ in range(n_cmp):
        i, j = rng_llm.choice(n_resp, 2, replace=False)
        pairs.append((i, j))
        pw = sigmoid(true_u[i] - true_u[j])
        if rng_llm.random() < 0.1:
            labels.append(rng_llm.choice([0, 1]))
        else:
            labels.append(1 if rng_llm.random() < pw else 0)
    pairs = np.array(pairs)
    labels = np.array(labels, dtype=float)

    n_train = int(0.8 * n_cmp)
    train_idx = rng_llm.choice(n_cmp, n_train, replace=False)
    r_train, c_train = pairs[train_idx, 0], pairs[train_idx, 1]
    y_train = labels[train_idx]

    # MLE
    V_mle = fit_regularized_bt(r_train, c_train, y_train, n_resp, lam=0.05, lr=0.1, epochs=300)

    # Elo
    V_elo = np.zeros(n_resp)
    for i, j, y in zip(r_train, c_train, y_train):
        p = sigmoid(V_elo[i] - V_elo[j])
        if y == 1.0:
            V_elo[i] += 0.1 * (1 - p)
            V_elo[j] -= 0.1 * (1 - p)
        else:
            V_elo[j] += 0.1 * p
            V_elo[i] -= 0.1 * p
        V_elo -= V_elo.mean()

    # Bayesian (MCMC)
    def log_posterior(V, r, c, y, prior_std=1.0):
        d = V[r] - V[c]
        ll = np.sum(y * d - np.log(1 + np.exp(d)))
        lp = -0.5 * np.sum(V**2) / prior_std**2
        return ll + lp

    np.random.seed(42)
    V_cur = fit_regularized_bt(r_train, c_train, y_train, n_resp, lam=1.0, lr=0.1, epochs=200)
    V_cur -= V_cur.mean()
    lp_cur = log_posterior(V_cur, r_train, c_train, y_train)
    samples = []
    for t in range(2500):
        V_prop = V_cur.copy()
        idx = np.random.randint(n_resp)
        V_prop[idx] += np.random.randn() * 0.15
        V_prop -= V_prop.mean()
        lp_prop = log_posterior(V_prop, r_train, c_train, y_train)
        if np.log(np.random.rand()) < lp_prop - lp_cur:
            V_cur, lp_cur = V_prop, lp_prop
        if t >= 500:
            samples.append(V_cur.copy())
    samples = np.array(samples)
    V_bayes = samples.mean(axis=0)
    V_bayes_std = samples.std(axis=0)

    def align(V_l, V_t):
        Vn = (V_l - V_l.mean()) / (V_l.std() + 1e-8)
        A = np.vstack([Vn, np.ones_like(Vn)]).T
        a, b = np.linalg.lstsq(A, V_t, rcond=None)[0]
        return a * Vn + b

    V_mle_a = align(V_mle, true_u)
    V_elo_a = align(V_elo, true_u)
    V_bayes_a = align(V_bayes, true_u)
    corr_mle = np.corrcoef(true_u, V_mle_a)[0, 1]
    corr_elo = np.corrcoef(true_u, V_elo_a)[0, 1]
    corr_bayes = np.corrcoef(true_u, V_bayes_a)[0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    lims = [true_u.min() - 0.5, true_u.max() + 0.5]

    for ax, Va, name, corr in zip(axes,
                                   [V_mle_a, V_elo_a, V_bayes_a],
                                   ['MLE', 'Elo', 'Bayesian'],
                                   [corr_mle, corr_elo, corr_bayes]):
        if name == 'Bayesian':
            ax.errorbar(true_u, Va, yerr=V_bayes_std * 1.96, fmt='o', alpha=0.5,
                        markersize=4, capsize=2, elinewidth=0.5)
        else:
            ax.scatter(true_u, Va, alpha=0.6, s=30)
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel('True Utility')
        ax.set_ylabel('Learned Utility')
        ax.set_title(f'{name} (r={corr:.3f})')
        ax.grid(True, alpha=0.3)

    fig.suptitle('LLM Preference Learning: Method Comparison', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "chap2_llm_comparison.png")
    plt.close(fig)
    print("  [10/10] chap2_llm_comparison.png")


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Chapter 2 slide figures...")
    fig_train_test_split()
    V_hat = fig_mle_training_auc()
    fig_learned_vs_true(V_hat)
    fig_mcmc_trace_posterior()
    fig_gp_posterior()
    M_s, r_s, c_s, y_s, r_te_s, c_te_s, y_te_s = fig_validation_curve()
    fig_early_stopping(M_s, r_s, c_s, y_s, r_te_s, c_te_s, y_te_s)
    fig_cv_hyperparam(M_s, r_s, c_s, y_s)
    fig_gd_vs_adam(M_s, r_s, c_s, y_s)
    fig_llm_comparison()
    print("Done! All figures saved to", OUT)
