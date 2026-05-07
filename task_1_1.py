"""
Task 1.1 - Distributed Optimization via Gradient Tracking
=========================================================
Implements the Gradient Tracking algorithm to solve:
    min_z  sum_{i=1}^{N}  ell_i(z)
where each ell_i(z) is a quadratic function.

Graph topologies supported: cycle, path, star
Weights computed via Metropolis-Hastings method.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

# ─────────────────────────────────────────────
# 1.  Graph construction & Metropolis-Hastings
# ─────────────────────────────────────────────

def metropolis_hastings_weights(adj: np.ndarray) -> np.ndarray:
    """
    Compute doubly-stochastic weight matrix via Metropolis-Hastings.
    W_ij = 1 / (1 + max(deg_i, deg_j))  for i != j in E
    W_ii = 1 - sum_{j != i} W_ij
    """
    N = adj.shape[0]
    degrees = adj.sum(axis=1)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and adj[i, j] == 1:
                W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        W[i, i] = 1.0 - W[i, :].sum()
    return W


def build_graph(topology: str, N: int):
    """Return (adjacency matrix, NetworkX graph) for the requested topology."""
    if topology == "cycle":
        G = nx.cycle_graph(N)
    elif topology == "path":
        G = nx.path_graph(N)
    elif topology == "star":
        G = nx.star_graph(N - 1)   # star_graph(k) has k+1 nodes
    else:
        raise ValueError(f"Unknown topology: {topology}")
    adj = nx.to_numpy_array(G, dtype=float)
    return adj, G


# ─────────────────────────────────────────────
# 2.  Local quadratic costs
# ─────────────────────────────────────────────

def make_quadratic_costs(N: int, d: int, seed: int = 42):
    """
    Generate N strongly-convex quadratic functions:
        ell_i(z) = 0.5 * z^T Q_i z + c_i^T z
    Returns lists of (Q_i, c_i).
    """
    rng = np.random.default_rng(seed)
    Qs, cs = [], []
    for _ in range(N):
        A = rng.standard_normal((d, d))
        Q = A.T @ A + np.eye(d) * 0.1          # ensure positive definite
        c = rng.standard_normal(d)
        Qs.append(Q)
        cs.append(c)
    return Qs, cs


def local_cost(z, Q, c):
    return 0.5 * z @ Q @ z + c @ z

def local_grad(z, Q, c):
    return Q @ z + c

def global_cost(z, Qs, cs):
    return sum(local_cost(z, Q, c) for Q, c in zip(Qs, cs))

def global_grad(z, Qs, cs):
    return sum(local_grad(z, Q, c) for Q, c in zip(Qs, cs))

def optimal_solution(Qs, cs):
    """Analytic optimum: (sum Q_i) z* = -(sum c_i)"""
    Q_tot = sum(Qs)
    c_tot = sum(cs)
    return np.linalg.solve(Q_tot, -c_tot)


# ─────────────────────────────────────────────
# 3.  Gradient Tracking algorithm
# ─────────────────────────────────────────────

def gradient_tracking(W, Qs, cs, d: int, alpha: float = 0.01,
                       max_iter: int = 1000, seed: int = 0):
    """
    Gradient Tracking (GT / DIGing).

    State per agent:
        x_i^k  ∈ R^d  – local estimate of z*
        y_i^k  ∈ R^d  – gradient tracker (tracks global gradient)

    Updates:
        x_i^{k+1} = sum_j W_ij x_j^k  -  alpha * y_i^k
        y_i^{k+1} = sum_j W_ij y_j^k  +  nabla ell_i(x_i^{k+1}) - nabla ell_i(x_i^k)
    """
    N = len(Qs)
    rng = np.random.default_rng(seed)

    # initialise
    X = rng.standard_normal((N, d))             # x_i^0
    Y = np.array([local_grad(X[i], Qs[i], cs[i]) for i in range(N)])  # y_i^0

    z_star = optimal_solution(Qs, cs)

    cost_hist, grad_norm_hist = [], []

    for k in range(max_iter):
        # --- consensus step ---
        X_new = W @ X - alpha * Y              # shape (N, d)

        # --- gradient tracker update ---
        grad_new = np.array([local_grad(X_new[i], Qs[i], cs[i]) for i in range(N)])
        grad_old = np.array([local_grad(X[i], Qs[i], cs[i]) for i in range(N)])
        Y_new = W @ Y + grad_new - grad_old

        X, Y = X_new, Y_new

        # --- logging (use average estimate as current candidate) ---
        z_avg = X.mean(axis=0)
        cost_hist.append(global_cost(z_avg, Qs, cs))
        grad_norm_hist.append(np.linalg.norm(global_grad(z_avg, Qs, cs)))

    return X, cost_hist, grad_norm_hist, z_star


# ─────────────────────────────────────────────
# 4.  Plotting helpers
# ─────────────────────────────────────────────

def plot_results(results: dict, title_suffix: str = ""):
    """
    results = { topology_name : (cost_hist, grad_norm_hist) }
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Gradient Tracking – distributed quadratic min {title_suffix}",
                 fontsize=13, fontweight="bold")

    for name, (cost_h, gnorm_h) in results.items():
        axes[0].semilogy(cost_h, label=name)
        axes[1].semilogy(gnorm_h, label=name)

    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Global cost  f(z̄ᵏ)")
    axes[0].set_title("Cost function evolution"); axes[0].legend(); axes[0].grid(True)

    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("‖∇f(z̄ᵏ)‖")
    axes[1].set_title("Gradient norm evolution"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    return fig


def plot_graph(adj: np.ndarray, topology: str, W: np.ndarray):
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="steelblue",
            node_size=600, font_color="white", font_weight="bold")
    ax.set_title(f"Graph topology: {topology}")
    return fig


# ─────────────────────────────────────────────
# 5.  Main simulation
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── hyper-parameters ──────────────────────
    N      = 10      # number of agents
    d      = 2       # dimension of optimisation variable
    alpha  = 0.005   # step size
    ITERS  = 800     # gradient tracking iterations
    TOPOLOGIES = ["cycle", "path", "star"]

    # ── generate shared cost functions ────────
    Qs, cs = make_quadratic_costs(N, d, seed=7)
    z_star = optimal_solution(Qs, cs)
    print(f"Analytic optimum z* = {z_star}")
    print(f"Optimal cost f(z*) = {global_cost(z_star, Qs, cs):.6f}\n")

    results = {}

    for topo in TOPOLOGIES:
        adj, G = build_graph(topo, N)
        W = metropolis_hastings_weights(adj)

        # verify doubly stochastic
        assert np.allclose(W.sum(axis=0), 1), "W not column-stochastic"
        assert np.allclose(W.sum(axis=1), 1), "W not row-stochastic"

        X_final, cost_h, gnorm_h = gradient_tracking(
            W, Qs, cs, d, alpha=alpha, max_iter=ITERS)[:3]

        z_avg_final = X_final.mean(axis=0)
        err = np.linalg.norm(z_avg_final - z_star)
        print(f"[{topo:6s}]  Final cost = {cost_h[-1]:.6f}  |  "
              f"‖z̄ - z*‖ = {err:.2e}  |  "
              f"grad norm = {gnorm_h[-1]:.2e}")

        results[topo] = (cost_h, gnorm_h)

    # ── plots ─────────────────────────────────
    fig_res = plot_results(results, title_suffix=f"(N={N}, d={d}, α={alpha})")
    fig_res.savefig("/mnt/user-data/outputs/task1_1_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved → task1_1_convergence.png")