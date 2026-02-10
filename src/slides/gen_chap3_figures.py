"""Generate figures for Chapter 3 (Elicitation) lecture slides."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Style — match gen_chap2_figures.py
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

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ── Figure 1: Fisher Information Parabola ─────────────────────────────
def fig_fisher_parabola():
    p = np.linspace(0, 1, 200)
    fi = p * (1 - p)

    fig, ax = plt.subplots()
    ax.plot(p, fi, 'b-', lw=2.5)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='$p = 0.5$')
    ax.plot(0.5, 0.25, 'ro', markersize=10, zorder=5)
    ax.annotate('max = 0.25', xy=(0.5, 0.25), xytext=(0.65, 0.23),
                fontsize=13, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_xlabel('$p = \\sigma(U + V_j)$')
    ax.set_ylabel('Fisher Information $p(1-p)$')
    ax.set_title('Fisher Information vs Response Probability')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, 0.28)
    fig.savefig(OUT / "chap3_fisher_parabola.png")
    plt.close(fig)
    print("  [1/6] chap3_fisher_parabola.png")


# ── Figure 2: Rasch Active vs Random ─────────────────────────────────
def fig_rasch_active_vs_random():
    rng = np.random.default_rng(42)
    N, M, T = 200, 200, 40
    sigma_U, sigma0 = 1.0, 1.0
    U_true = rng.normal(0, sigma_U, size=N)
    V = np.sort(rng.normal(0, 1.0, size=M))

    def new_state():
        return {"U_hat": 0.0, "tau": 1.0 / sigma0**2, "var": sigma0**2}

    def update(state, vj, y):
        U, tau = state["U_hat"], state["tau"]
        p = sigmoid(U + vj)
        S, I = (y - p), p * (1 - p)
        state["U_hat"] = U + S / (I + tau + 1e-12)
        state["tau"] = tau + I
        state["var"] = 1.0 / (state["tau"] + 1e-12)

    def choose_fisher(state, V, mask):
        cand = np.where(~mask)[0]
        if cand.size == 0:
            return None
        vals = sigmoid(state["U_hat"] + V[cand])
        info = vals * (1.0 - vals)
        return cand[np.argmax(info + 1e-6 * rng.random(size=cand.size))]

    def choose_random(state, V, mask):
        cand = np.where(~mask)[0]
        return rng.choice(cand) if cand.size > 0 else None

    def run_policy(policy_fn):
        states = [new_state() for _ in range(N)]
        asked = np.zeros((N, M), dtype=bool)
        mse_curve, rel_curve = [], []
        for t in range(1, T + 1):
            for i in range(N):
                j = policy_fn(states[i], V, asked[i])
                if j is None:
                    continue
                y = 1 if rng.random() < sigmoid(U_true[i] + V[j]) else 0
                update(states[i], V[j], y)
                asked[i, j] = True
            U_hat = np.array([s["U_hat"] for s in states])
            Var_hat = np.array([s["var"] for s in states])
            mse_curve.append(np.mean((U_hat - U_true)**2))
            rel_curve.append(1.0 - np.mean(Var_hat) / sigma_U**2)
        return np.array(mse_curve), np.array(rel_curve)

    mse_act, rel_act = run_policy(choose_fisher)
    mse_rnd, rel_rnd = run_policy(choose_random)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(1, T + 1)

    axes[0].plot(xs, mse_rnd, 'o-', label="Random", alpha=0.7, markersize=4, markevery=3)
    axes[0].plot(xs, mse_act, 's-', label="Fisher-active", alpha=0.7, markersize=4, markevery=3)
    axes[0].set_xlabel("Queries per user")
    axes[0].set_ylabel("MSE of $\\hat{U}$")
    axes[0].set_title("Estimation Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, rel_rnd, 'o-', label="Random", alpha=0.7, markersize=4, markevery=3)
    axes[1].plot(xs, rel_act, 's-', label="Fisher-active", alpha=0.7, markersize=4, markevery=3)
    axes[1].set_xlabel("Queries per user")
    axes[1].set_ylabel("Reliability")
    axes[1].set_title("Reliability")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "chap3_rasch_active_vs_random.png")
    plt.close(fig)
    print("  [2/6] chap3_rasch_active_vs_random.png")


# ── Figure 3: Factor Model D-Optimal vs Random ───────────────────────
def fig_factor_dopt_vs_random():
    rng = np.random.default_rng(0)
    N, M, K, T = 100, 300, 2, 30
    U_true = rng.normal(0, 1, size=(N, K))
    V = rng.normal(0, 1, size=(M, K))
    Z = rng.normal(0, 0.3, size=(M,))

    def new_state():
        return {"U_hat": np.zeros(K), "Sigma": np.eye(K) * 2.0}

    def update(state, vj, zj, y):
        U, S = state["U_hat"], state["Sigma"]
        p = sigmoid(U @ vj + zj)
        I_j = p * (1 - p) * np.outer(vj, vj)
        Sigma_new = np.linalg.inv(np.linalg.inv(S) + I_j)
        state["U_hat"] = U + Sigma_new @ ((y - p) * vj)
        state["Sigma"] = Sigma_new

    def choose_dopt(state, asked_mask):
        Prec = np.linalg.inv(state["Sigma"])
        cand = np.where(~asked_mask)[0]
        scores = []
        for j in cand:
            w = sigmoid(state["U_hat"] @ V[j] + Z[j])
            w = w * (1 - w)
            scores.append(np.log(np.linalg.det(Prec + w * np.outer(V[j], V[j]))))
        return cand[np.argmax(scores)]

    def run_policy(active=True):
        states = [new_state() for _ in range(N)]
        asked = np.zeros((N, M), dtype=bool)
        mse_curve, rel_curve = [], []
        for t in range(T):
            for i in range(N):
                if active:
                    j = choose_dopt(states[i], asked[i])
                else:
                    j = rng.choice(np.where(~asked[i])[0])
                y = int(rng.random() < sigmoid(U_true[i] @ V[j] + Z[j]))
                update(states[i], V[j], Z[j], y)
                asked[i, j] = True
            U_hat = np.array([s["U_hat"] for s in states])
            err_tr = np.mean([np.trace(s["Sigma"]) for s in states])
            mse_curve.append(np.mean(np.sum((U_hat - U_true)**2, axis=1)))
            rel_curve.append(1 - err_tr / np.trace(np.cov(U_true.T)))
        return np.array(mse_curve), np.array(rel_curve)

    mse_rand, rel_rand = run_policy(active=False)
    mse_act, rel_act = run_policy(active=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(1, T + 1)

    axes[0].plot(xs, mse_rand, 'o-', label="Random", alpha=0.7, markersize=4, markevery=2)
    axes[0].plot(xs, mse_act, 's-', label="D-optimal", alpha=0.7, markersize=4, markevery=2)
    axes[0].set_xlabel("Queries")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Estimation Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, rel_rand, 'o-', label="Random", alpha=0.7, markersize=4, markevery=2)
    axes[1].plot(xs, rel_act, 's-', label="D-optimal", alpha=0.7, markersize=4, markevery=2)
    axes[1].set_xlabel("Queries")
    axes[1].set_ylabel("Reliability")
    axes[1].set_title("Reliability")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "chap3_factor_dopt_vs_random.png")
    plt.close(fig)
    print("  [3/6] chap3_factor_dopt_vs_random.png")


# ── Figure 4: Pairwise D-Optimal vs Random ───────────────────────────
def fig_pairwise_dopt_vs_random():
    rng = np.random.default_rng(0)
    N, M, K, T = 200, 60, 2, 30
    V = rng.normal(0, 1.0, size=(M, K))
    Z = rng.normal(0, 0.3, size=M)
    U_true = rng.normal(0, 1.0, size=(N, K))

    pairs = np.array([(j, k) for j in range(M) for k in range(j + 1, M)])
    X = V[pairs[:, 0]] - V[pairs[:, 1]]
    B = Z[pairs[:, 0]] - Z[pairs[:, 1]]

    def d_opt_gain(Sigma, x, w):
        return np.log1p(w * x @ Sigma @ x)

    def one_step_update(U_hat, Sigma, x, b, y):
        p = sigmoid(U_hat @ x + b)
        w = p * (1 - p)
        Sx = Sigma @ x
        Sigma_new = Sigma - (w / (1.0 + w * (x @ Sx))) * np.outer(Sx, Sx)
        U_new = U_hat + Sigma_new @ ((y - p) * x)
        return U_new, Sigma_new

    def choose_dopt(U_hat, Sigma):
        h = X @ U_hat + B
        p = sigmoid(h)
        w = p * (1 - p)
        gains = np.array([d_opt_gain(Sigma, X[i], w[i]) for i in range(len(X))])
        gains[np.isnan(gains)] = -np.inf
        return int(np.argmax(gains))

    def run_policy(active=True):
        Uh = np.zeros((N, K))
        Sigmas = np.tile(np.eye(K) * 2.0, (N, 1, 1))
        mse_curve, rel_curve = [], []
        for t in range(T):
            for i in range(N):
                idx = choose_dopt(Uh[i], Sigmas[i]) if active else rng.integers(len(pairs))
                x, b = X[idx], B[idx]
                y = 1 if rng.random() < sigmoid(U_true[i] @ x + b) else 0
                Uh[i], Sigmas[i] = one_step_update(Uh[i], Sigmas[i], x, b, y)
            mse_curve.append(np.mean(np.sum((Uh - U_true)**2, axis=1)))
            rel_curve.append(1.0 - np.mean(np.trace(Sigmas, axis1=1, axis2=2)) / K)
        return np.array(mse_curve), np.array(rel_curve)

    mse_d, rel_d = run_policy(active=True)
    mse_r, rel_r = run_policy(active=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(1, T + 1)

    axes[0].plot(xs, mse_r, 'o-', label="Random", alpha=0.7, markersize=4, markevery=2)
    axes[0].plot(xs, mse_d, 's-', label="D-optimal", alpha=0.7, markersize=4, markevery=2)
    axes[0].set_xlabel("Queries per user")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Estimation Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, rel_r, 'o-', label="Random", alpha=0.7, markersize=4, markevery=2)
    axes[1].plot(xs, rel_d, 's-', label="D-optimal", alpha=0.7, markersize=4, markevery=2)
    axes[1].set_xlabel("Queries per user")
    axes[1].set_ylabel("Reliability")
    axes[1].set_title("Reliability")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "chap3_pairwise_dopt_vs_random.png")
    plt.close(fig)
    print("  [4/6] chap3_pairwise_dopt_vs_random.png")


# ── Figure 5: Linear Preference Function D-Optimal ───────────────────
def fig_linear_pref_dopt():
    rng = np.random.default_rng(0)
    d, M, T = 5, 50, 100
    sigma0 = 2.0
    X = rng.normal(0, 1, (M, d))
    W_true = rng.normal(0, 1, d)

    pairs = np.array([(j, k) for j in range(M) for k in range(j + 1, M)])
    Xdiff = X[pairs[:, 0]] - X[pairs[:, 1]]

    def fisher_gain(Sigma, x, w):
        Sx = Sigma @ x
        denom = 1 + w * (x @ Sx)
        Sigma_new = Sigma - (w / denom) * np.outer(Sx, Sx)
        return np.log(np.linalg.det(Sigma) / np.linalg.det(Sigma_new))

    def update(W_hat, Sigma, x, y):
        p = sigmoid(W_hat @ x)
        w = p * (1 - p)
        Sx = Sigma @ x
        denom = 1 + w * (x @ Sx)
        Sigma_new = Sigma - (w / denom) * np.outer(Sx, Sx)
        W_new = W_hat + Sigma_new @ ((y - p) * x)
        return W_new, Sigma_new

    def run_policy(active=True):
        W_hat = np.zeros(d)
        Sigma = np.eye(d) * sigma0**2
        mse_hist, rel_hist = [], []
        for t in range(T):
            if active:
                gains = [fisher_gain(Sigma, x, sigmoid(W_hat @ x) * (1 - sigmoid(W_hat @ x)))
                         for x in Xdiff]
                idx = np.argmax(gains)
            else:
                idx = rng.integers(len(Xdiff))
            x = Xdiff[idx]
            y = rng.random() < sigmoid(W_true @ x)
            W_hat, Sigma = update(W_hat, Sigma, x, y)
            mse_hist.append(np.sum((W_hat - W_true)**2))
            rel_hist.append(1 - np.trace(Sigma) / (d * np.var(W_true)))
        return np.array(mse_hist), np.array(rel_hist)

    mse_act, rel_act = run_policy(True)
    mse_rnd, rel_rnd = run_policy(False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(1, T + 1)

    axes[0].plot(xs, mse_rnd, 'o-', label="Random", alpha=0.7, markersize=3, markevery=8)
    axes[0].plot(xs, mse_act, 's-', label="D-optimal", alpha=0.7, markersize=3, markevery=8)
    axes[0].set_xlabel("Queries")
    axes[0].set_ylabel("MSE of $\\hat{W}$")
    axes[0].set_title("Estimation Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, rel_rnd, 'o-', label="Random", alpha=0.7, markersize=3, markevery=8)
    axes[1].plot(xs, rel_act, 's-', label="D-optimal", alpha=0.7, markersize=3, markevery=8)
    axes[1].set_xlabel("Queries")
    axes[1].set_ylabel("Reliability")
    axes[1].set_title("Reliability")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "chap3_linear_pref_dopt.png")
    plt.close(fig)
    print("  [5/6] chap3_linear_pref_dopt.png")


# ── Figure 6: GP Active Learning Results ──────────────────────────────
def fig_gp_active_results():
    np.random.seed(42)

    def rbf_kernel(X1, X2, sigma_f=1.0, length_scale=0.7):
        sq_dist = np.sum(X1**2, 1, keepdims=True) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return sigma_f**2 * np.exp(-sq_dist / (2 * length_scale**2))

    binary_entropy = lambda p: -p * np.log2(p + 1e-10) - (1 - p) * np.log2(1 - p + 1e-10)

    def true_reward(x):
        return np.sin(2 * x) + 0.3 * x

    class GPActivePref:
        def __init__(self, X_pool, sigma_noise=0.1):
            self.X_pool = X_pool
            self.sigma_noise = sigma_noise
            self.comparisons = []
            self.r = np.zeros(len(X_pool))
            self.Sigma = rbf_kernel(X_pool, X_pool) + 1e-4 * np.eye(len(X_pool))

        def fit(self):
            if len(self.comparisons) == 0:
                return
            m = len(self.X_pool)
            self.K = rbf_kernel(self.X_pool, self.X_pool) + 1e-4 * np.eye(m)
            self.K_inv = np.linalg.inv(self.K)
            self.r = np.zeros(m)
            for _ in range(10):
                grad, W = np.zeros(m), np.zeros(m)
                for i, j, y in self.comparisons:
                    p = sigmoid(self.r[i] - self.r[j])
                    grad[i] += (y - p)
                    grad[j] -= (y - p)
                    W[i] += p * (1 - p)
                    W[j] += p * (1 - p)
                H = -np.diag(W + 1e-6) - self.K_inv
                self.r -= np.linalg.solve(H, grad - self.K_inv @ self.r)
            self.W = W
            self.Sigma = np.linalg.inv(self.K_inv + np.diag(W + 1e-6))

        def acquisition(self, i, j):
            if len(self.comparisons) == 0:
                return 0.5
            var_diff = self.Sigma[i, i] + self.Sigma[j, j] - 2 * self.Sigma[i, j]
            pred_std = np.sqrt(2 * self.sigma_noise**2 + var_diff)
            p_pred = sigmoid((self.r[i] - self.r[j]) / pred_std)
            return binary_entropy(p_pred)

        def select_query(self):
            m = len(self.X_pool)
            best_score, best_pair = -np.inf, (0, 1)
            for i in range(m):
                for j in range(i + 1, m):
                    score = self.acquisition(i, j)
                    if score > best_score:
                        best_score, best_pair = score, (i, j)
            return best_pair

        def add_comparison(self, i, j):
            r_true = true_reward(self.X_pool.flatten())
            p = sigmoid(r_true[i] - r_true[j])
            y = 1 if np.random.rand() < p else 0
            self.comparisons.append((i, j, y))
            self.fit()

    X_pool = np.linspace(-3, 3, 30).reshape(-1, 1)
    n_queries = 25
    r_true = true_reward(X_pool.flatten())

    gp_active = GPActivePref(X_pool)
    mse_active = []
    gp_random = GPActivePref(X_pool)
    mse_random = []

    for t in range(n_queries):
        i, j = gp_active.select_query()
        gp_active.add_comparison(i, j)
        mse_active.append(np.mean((gp_active.r - r_true)**2))

        i, j = np.random.choice(len(X_pool), 2, replace=False)
        gp_random.add_comparison(i, j)
        mse_random.append(np.mean((gp_random.r - r_true)**2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(range(1, n_queries + 1), mse_random, 'o-', label='Random', alpha=0.7, markersize=4)
    axes[0].plot(range(1, n_queries + 1), mse_active, 's-', label='Active (Info Gain)', alpha=0.7, markersize=4)
    axes[0].set_xlabel('Number of comparisons')
    axes[0].set_ylabel('MSE of reward estimate')
    axes[0].set_title('GP Active vs Random')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(X_pool, r_true, 'k--', lw=2, label='True reward')
    axes[1].plot(X_pool, gp_active.r, 'b-', lw=2, label='Active GP')
    axes[1].plot(X_pool, gp_random.r, 'r-', lw=2, alpha=0.7, label='Random GP')
    axes[1].fill_between(X_pool.flatten(),
                         gp_active.r - 2 * np.sqrt(np.diag(gp_active.Sigma)),
                         gp_active.r + 2 * np.sqrt(np.diag(gp_active.Sigma)),
                         alpha=0.2, color='blue', label='$\\pm 2\\sigma$ (Active)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('r(x)')
    axes[1].set_title(f'Learned Rewards ({n_queries} comparisons)')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "chap3_gp_active_results.png")
    plt.close(fig)
    print("  [6/6] chap3_gp_active_results.png")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Chapter 3 slide figures...")
    fig_fisher_parabola()
    fig_rasch_active_vs_random()
    fig_factor_dopt_vs_random()
    fig_pairwise_dopt_vs_random()
    fig_linear_pref_dopt()
    fig_gp_active_results()
    print("Done! All figures saved to", OUT)
