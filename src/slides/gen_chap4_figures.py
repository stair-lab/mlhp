"""Generate figures for Chapter 4 (Decisions) lecture slides."""

import numpy as np
import matplotlib.pyplot as plt
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

rng = np.random.default_rng(42)
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ── Figure 1: GP Kernel + Prior Samples ──────────────────────────────
def fig_gp_kernel_samples():
    """RBF kernel covariance and prior function samples for 3 lengthscales."""

    def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
        diff = X1[:, None] - X2[None, :]
        return variance * np.exp(-0.5 * diff ** 2 / lengthscale ** 2)

    X = np.linspace(-3, 3, 100)
    lengthscales = [0.3, 1.0, 3.0]
    titles = ['Wiggly ($\\ell=0.3$)', 'Medium ($\\ell=1.0$)', 'Smooth ($\\ell=3.0$)']

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    np.random.seed(42)

    for i, (ls, title) in enumerate(zip(lengthscales, titles)):
        # Top: kernel covariance from center
        K_center = rbf_kernel(X, np.array([0.0]), lengthscale=ls).flatten()
        axes[0, i].plot(X, K_center, 'b-', linewidth=2)
        axes[0, i].set_title(title, fontsize=13)
        if i == 0:
            axes[0, i].set_ylabel('$k(x, 0)$')
        axes[0, i].set_xlabel('$x$')
        axes[0, i].grid(True, alpha=0.3)

        # Bottom: GP prior samples
        K = rbf_kernel(X, X, lengthscale=ls) + 1e-6 * np.eye(len(X))
        L = np.linalg.cholesky(K)
        for _ in range(5):
            f_sample = L @ np.random.randn(len(X))
            axes[1, i].plot(X, f_sample, alpha=0.6, linewidth=1)
        if i == 0:
            axes[1, i].set_ylabel('$h(x)$')
        axes[1, i].set_xlabel('$x$')
        axes[1, i].set_ylim(-3, 3)
        axes[1, i].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "chap4_gp_kernel_samples.png")
    plt.close(fig)
    print("  [1/2] chap4_gp_kernel_samples.png")


# ── Figure 2: Winner Concept Divergence ──────────────────────────────
def fig_winner_divergence():
    """Show preference matrix where Condorcet and Borda winners differ."""
    # 4 alternatives: Condorcet winner barely beats everyone,
    # but Borda winner dominates 2 others strongly
    P = np.array([
        [0.50, 0.51, 0.51, 0.51],   # A: Condorcet (barely beats all)
        [0.49, 0.50, 0.90, 0.90],   # B: Borda (strong vs C,D)
        [0.49, 0.10, 0.50, 0.60],   # C
        [0.49, 0.10, 0.40, 0.50],   # D
    ])
    labels = ['A', 'B', 'C', 'D']
    n = len(labels)

    # Compute winners
    borda_scores = [sum(P[i, j] for j in range(n) if j != i) for i in range(n)]
    condorcet = None
    for i in range(n):
        if all(P[i, j] > 0.5 for j in range(n) if j != i):
            condorcet = i
            break

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: preference matrix
    im = axes[0].imshow(P, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(labels)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel('Opponent $j$')
    axes[0].set_ylabel('Option $i$')
    axes[0].set_title('Preference Matrix $P(i \\succ j)$')
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(P[i, j] - 0.5) > 0.15 else 'black'
            axes[0].text(j, i, f'{P[i, j]:.2f}', ha='center', va='center',
                         fontsize=12, color=color, fontweight='bold')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Right: Borda scores
    colors = ['#2ecc71' if i == condorcet else '#e74c3c' if borda_scores[i] == max(borda_scores)
              else '#3498db' for i in range(n)]
    bars = axes[1].bar(range(n), borda_scores, color=colors, alpha=0.85, edgecolor='black')
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Borda Score')
    axes[1].set_title('Borda Scores')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label=f'Condorcet: {labels[condorcet]}'),
        Patch(facecolor='#e74c3c', edgecolor='black', label=f'Borda: {labels[np.argmax(borda_scores)]}'),
    ]
    axes[1].legend(handles=legend_elements, fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT / "chap4_winner_divergence.png")
    plt.close(fig)
    print("  [2/2] chap4_winner_divergence.png")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Chapter 4 slide figures...")
    fig_gp_kernel_samples()
    fig_winner_divergence()
    print("Done! All figures saved to", OUT)
