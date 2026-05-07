"""
Task 1.3 - Distributed Classification via Gradient Tracking
============================================================
Extends Tasks 1.1 and 1.2:
  • Splits dataset across N agents using the group-specific rule
  • Runs Gradient Tracking (distributed logistic regression)
  • Compares distributed vs centralised solution

Set GROUP_NUMBER and N below before running.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── import shared utilities from task 1.1 and 1.2 ──────────────────────────
# (copy the necessary functions here so each file is self-contained)

# ─────────────────────────────────────────────
# CONFIGURE YOUR GROUP HERE
# ─────────────────────────────────────────────
GROUP_NUMBER = 1   # <-- change to your actual group number
N_AGENTS     = 10  # number of agents

# ─────────────────────────────────────────────
# 1.  Feature mappings  (same as Task 1.2)
# ─────────────────────────────────────────────

def phi_cubic(X):
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0]**3])

def phi_superellipse(X):
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0]**4, X[:, 1]**4])

def phi_parabola(X):
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0]**2])

def phi_hyperbola(X):
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0]*X[:, 1]])

def get_mappings(group_number):
    if group_number % 2 == 1:
        return [("Cubic", phi_cubic), ("Superellipse", phi_superellipse)]
    else:
        return [("Parabola", phi_parabola), ("Hyperbola", phi_hyperbola)]


# ─────────────────────────────────────────────
# 2.  Dataset generation  (same as Task 1.2)
# ─────────────────────────────────────────────

def generate_dataset(M, w_true, b_true, phi_fn, noise=0.02, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (M, 2))
    Phi = phi_fn(X)
    scores = Phi @ w_true + b_true
    labels = np.where(scores >= 0, 1, -1)
    flip = rng.random(M) < noise
    labels[flip] *= -1
    return X, labels


# ─────────────────────────────────────────────
# 3.  Data splitting rule (project spec)
# ─────────────────────────────────────────────

def split_dataset(X, labels, N, group_number, seed=0):
    """
    P = 40 + (G mod 3) * 10  (baseline %)
    Odd  groups → sort remaining by x1 (horizontal feature-biased)
    Even groups → sort remaining by x2 (vertical feature-biased)

    Returns list of (X_i, labels_i) for i = 0..N-1
    """
    rng = np.random.default_rng(seed)
    M = len(labels)
    G = group_number
    P = 40 + (G % 3) * 10
    n_base = int(round(M * P / 100))

    # --- baseline: random P% split equally among N agents ---
    idx_all = rng.permutation(M)
    base_idx = idx_all[:n_base]
    rest_idx = idx_all[n_base:]

    per_agent_base = n_base // N
    agent_data = []
    for i in range(N):
        start = i * per_agent_base
        end   = (i + 1) * per_agent_base if i < N - 1 else n_base
        agent_data.append([base_idx[start:end]])

    # --- biased split of remaining (100-P)% ---
    sort_col = 0 if G % 2 == 1 else 1   # odd→x1, even→x2
    sorted_rest = rest_idx[np.argsort(X[rest_idx, sort_col])]
    n_rest = len(sorted_rest)
    per_agent_rest = n_rest // N

    for i in range(N):
        start = i * per_agent_rest
        end   = (i + 1) * per_agent_rest if i < N - 1 else n_rest
        agent_data[i].append(sorted_rest[start:end])

    # combine
    splits = []
    for i in range(N):
        idx_i = np.concatenate(agent_data[i])
        splits.append((X[idx_i], labels[idx_i]))

    return splits


# ─────────────────────────────────────────────
# 4.  Local logistic loss & gradient
# ─────────────────────────────────────────────

def local_logistic_loss(wb, Phi_i, labels_i):
    w, b = wb[:-1], wb[-1]
    margins = labels_i * (Phi_i @ w + b)
    return np.sum(np.log1p(np.exp(-margins)))

def local_logistic_grad(wb, Phi_i, labels_i):
    w, b = wb[:-1], wb[-1]
    margins = labels_i * (Phi_i @ w + b)
    sigma = 1.0 / (1.0 + np.exp(margins))
    weighted = labels_i * (-sigma)
    return np.concatenate([Phi_i.T @ weighted, [weighted.sum()]])

def global_logistic_loss(wb, Phi_list, labels_list):
    return sum(local_logistic_loss(wb, P, l) for P, l in zip(Phi_list, labels_list))

def global_logistic_grad(wb, Phi_list, labels_list):
    return sum(local_logistic_grad(wb, P, l) for P, l in zip(Phi_list, labels_list))


# ─────────────────────────────────────────────
# 5.  Graph & Metropolis-Hastings  (same as 1.1)
# ─────────────────────────────────────────────

def metropolis_hastings_weights(adj):
    N = adj.shape[0]
    degrees = adj.sum(axis=1)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and adj[i, j] == 1:
                W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        W[i, i] = 1.0 - W[i, :].sum()
    return W

def build_graph(topology, N):
    if topology == "cycle":   G = nx.cycle_graph(N)
    elif topology == "path":  G = nx.path_graph(N)
    elif topology == "star":  G = nx.star_graph(N - 1)
    else: raise ValueError(topology)
    return nx.to_numpy_array(G, dtype=float)


# ─────────────────────────────────────────────
# 6.  Gradient Tracking for classification
# ─────────────────────────────────────────────

def gradient_tracking_classification(W, Phi_list, labels_list,
                                      q_plus_1: int,
                                      alpha: float = 1e-3,
                                      max_iter: int = 2000,
                                      seed: int = 0):
    """
    Distributed logistic regression via Gradient Tracking.
    Each agent i holds Phi_list[i], labels_list[i].
    """
    N = len(Phi_list)
    rng = np.random.default_rng(seed)

    # init: all agents start at same small perturbation
    WB = rng.standard_normal((N, q_plus_1)) * 0.01   # local estimates
    Y  = np.array([local_logistic_grad(WB[i], Phi_list[i], labels_list[i])
                   for i in range(N)])                # gradient trackers

    cost_hist, gnorm_hist = [], []

    for k in range(max_iter):
        WB_new = W @ WB - alpha * Y

        grad_new = np.array([local_logistic_grad(WB_new[i], Phi_list[i], labels_list[i])
                             for i in range(N)])
        grad_old = np.array([local_logistic_grad(WB[i], Phi_list[i], labels_list[i])
                             for i in range(N)])
        Y = W @ Y + grad_new - grad_old
        WB = WB_new

        wb_avg = WB.mean(axis=0)
        cost_hist.append(global_logistic_loss(wb_avg, Phi_list, labels_list))
        gnorm_hist.append(np.linalg.norm(global_logistic_grad(wb_avg, Phi_list, labels_list)))

    return WB, cost_hist, gnorm_hist


# ─────────────────────────────────────────────
# 7.  Centralised gradient descent  (for comparison)
# ─────────────────────────────────────────────

def centralised_gd(Phi_full, labels_full, q_plus_1,
                   alpha=5e-4, max_iter=3000):
    wb = np.zeros(q_plus_1)
    cost_h, gnorm_h = [], []
    for _ in range(max_iter):
        g = local_logistic_grad(wb, Phi_full, labels_full)
        cost_h.append(local_logistic_loss(wb, Phi_full, labels_full))
        gnorm_h.append(np.linalg.norm(g))
        wb -= alpha * g
    return wb, cost_h, gnorm_h


# ─────────────────────────────────────────────
# 8.  Evaluation & plots
# ─────────────────────────────────────────────

def misclassification(wb, Phi, labels):
    w, b = wb[:-1], wb[-1]
    preds = np.sign(Phi @ w + b);  preds[preds == 0] = 1
    return np.mean(preds != labels) * 100


def plot_comparison(cost_c, gnorm_c, cost_d, gnorm_d, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Centralised vs Distributed – {title}", fontsize=12, fontweight="bold")

    axes[0].semilogy(cost_c,  label="Centralised", color="steelblue")
    axes[0].semilogy(cost_d,  label="Distributed", color="tomato", linestyle="--")
    axes[0].set_title("Cost evolution"); axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Global loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].semilogy(gnorm_c, label="Centralised", color="steelblue")
    axes[1].semilogy(gnorm_d, label="Distributed", color="tomato", linestyle="--")
    axes[1].set_title("Gradient norm"); axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("‖∇L‖"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    return fig


def plot_decision_boundary_distributed(X, labels, wb_dist, wb_cent,
                                        phi_fn, title=""):
    x1r = np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 300)
    x2r = np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 300)
    xx1, xx2 = np.meshgrid(x1r, x2r)
    grid = np.column_stack([xx1.ravel(), xx2.ravel()])
    Phi_g = phi_fn(grid)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, wb, name in zip(axes, [wb_dist, wb_cent], ["Distributed", "Centralised"]):
        w, b = wb[:-1], wb[-1]
        scores = (Phi_g @ w + b).reshape(xx1.shape)
        ax.contourf(xx1, xx2, scores, levels=[-1e9, 0, 1e9],
                    colors=["#f7c6c7", "#c6daf7"], alpha=0.5)
        ax.contour(xx1, xx2, scores, levels=[0], colors="black", linewidths=1.5)
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="bwr",
                   edgecolors="k", s=20, alpha=0.7)
        miss = misclassification(wb, phi_fn(X), labels)
        ax.set_title(f"{name}  (miss={miss:.1f}%)")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    M        = 500
    N        = N_AGENTS
    alpha_d  = 5e-4    # distributed step
    alpha_c  = 5e-4    # centralised step
    ITERS_D  = 2000
    ITERS_C  = 3000
    TOPOLOGY = "cycle"   # change to "path" or "star" for other topologies

    mappings = get_mappings(GROUP_NUMBER)
    G_num    = GROUP_NUMBER
    P        = 40 + (G_num % 3) * 10
    print(f"Group {G_num} | P={P}% baseline | topology={TOPOLOGY} | N={N}\n")

    # build weight matrix
    adj = build_graph(TOPOLOGY, N)
    W   = metropolis_hastings_weights(adj)

    for map_name, phi_fn in mappings:
        print(f"═══ Mapping: {map_name} ═══")

        # ground-truth parameters
        rng = np.random.default_rng(99)
        q   = phi_fn(np.zeros((1, 2))).shape[1]
        w_t = rng.standard_normal(q);  w_t /= np.linalg.norm(w_t)
        b_t = rng.uniform(-0.5, 0.5)

        X, labels = generate_dataset(M, w_t, b_t, phi_fn, noise=0.02, seed=42)
        Phi_full  = phi_fn(X)
        q_plus_1  = q + 1

        # ── data split ──
        splits     = split_dataset(X, labels, N, G_num, seed=0)
        Phi_list   = [phi_fn(Xi)  for Xi, _ in splits]
        label_list = [li          for  _, li in splits]

        print(f"  Agent data sizes: {[len(li) for li in label_list]}")

        # ── centralised ──
        wb_c, cost_c, gnorm_c = centralised_gd(
            Phi_full, labels, q_plus_1, alpha=alpha_c, max_iter=ITERS_C)
        miss_c = misclassification(wb_c, Phi_full, labels)
        print(f"  [Centralised]  final loss={cost_c[-1]:.4f}  miss={miss_c:.2f}%")

        # ── distributed ──
        WB_final, cost_d, gnorm_d = gradient_tracking_classification(
            W, Phi_list, label_list, q_plus_1,
            alpha=alpha_d, max_iter=ITERS_D)
        wb_d = WB_final.mean(axis=0)
        miss_d = misclassification(wb_d, Phi_full, labels)
        print(f"  [Distributed]  final loss={cost_d[-1]:.4f}  miss={miss_d:.2f}%\n")

        tag = map_name.lower()

        # comparison curves
        fig_cmp = plot_comparison(cost_c, gnorm_c, cost_d, gnorm_d,
                                  title=f"{map_name} (N={N}, {TOPOLOGY})")
        fig_cmp.savefig(f"/mnt/user-data/outputs/task1_3_comparison_{tag}.png",
                        dpi=150, bbox_inches="tight")

        # decision boundaries
        fig_db = plot_decision_boundary_distributed(
            X, labels, wb_d, wb_c, phi_fn,
            title=f"Decision Boundaries – {map_name}")
        fig_db.savefig(f"/mnt/user-data/outputs/task1_3_boundaries_{tag}.png",
                       dpi=150, bbox_inches="tight")

    plt.show()
    print("All plots saved.")