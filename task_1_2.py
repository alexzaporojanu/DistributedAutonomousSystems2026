"""
Task 1.2 - Centralized Classification via Logistic Regression
=============================================================
Supports both Odd and Even group feature mappings.

Odd  Groups: Cubic      ϕ(x) = [x1, x2, x1³]
             Superellipse ϕ(x) = [x1, x2, x1⁴, x2⁴]

Even Groups: Parabola   ϕ(x) = [x1, x2, x1²]
             Hyperbola   ϕ(x) = [x1, x2, x1·x2]

Set GROUP_NUMBER below before running.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# CONFIGURE YOUR GROUP HERE
# ─────────────────────────────────────────────
GROUP_NUMBER = 1   # <-- change to your actual group number

# ─────────────────────────────────────────────
# 1.  Feature mappings
# ─────────────────────────────────────────────

def phi_cubic(X):
    """Odd: ϕ(x) = [x1, x2, x1³]"""
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0]**3])

def phi_superellipse(X):
    """Odd: ϕ(x) = [x1, x2, x1⁴, x2⁴]"""
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0]**4, X[:, 1]**4])

def phi_parabola(X):
    """Even: ϕ(x) = [x1, x2, x1²]"""
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0]**2])

def phi_hyperbola(X):
    """Even: ϕ(x) = [x1, x2, x1·x2]"""
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0]*X[:, 1]])


def get_mappings(group_number: int):
    if group_number % 2 == 1:
        return [("Cubic",       phi_cubic),
                ("Superellipse", phi_superellipse)]
    else:
        return [("Parabola",    phi_parabola),
                ("Hyperbola",   phi_hyperbola)]


# ─────────────────────────────────────────────
# 2.  Dataset generation
# ─────────────────────────────────────────────

def generate_dataset(M: int, w_true: np.ndarray, b_true: float,
                     phi_fn, noise: float = 0.05, seed: int = 42):
    """
    Generate M 2-D points and label them using the true separating function
    w^T ϕ(x) + b = 0 with optional label-flip noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (M, 2))
    Phi = phi_fn(X)
    scores = Phi @ w_true + b_true
    labels = np.where(scores >= 0, 1, -1)

    # optional label noise
    flip = rng.random(M) < noise
    labels[flip] *= -1

    return X, labels


# ─────────────────────────────────────────────
# 3.  Logistic regression loss & gradient
# ─────────────────────────────────────────────

def logistic_loss(wb, Phi, labels):
    """
    L(w,b) = sum_m log(1 + exp(-p_m (w^T ϕ(D_m) + b)))
    wb      : (q+1,) vector  [w; b]
    Phi     : (M, q) mapped features
    labels  : (M,) in {-1, +1}
    """
    w, b = wb[:-1], wb[-1]
    margins = labels * (Phi @ w + b)
    # numerically stable: log(1 + exp(-m)) = log(1 + exp(-m))
    return np.sum(np.log1p(np.exp(-margins)))


def logistic_grad(wb, Phi, labels):
    """Gradient of logistic loss w.r.t. [w; b]."""
    w, b = wb[:-1], wb[-1]
    margins = labels * (Phi @ w + b)
    sigma = 1.0 / (1.0 + np.exp(margins))      # sigmoid(-margin)
    dL_dmargin = -sigma                          # d loss / d margin_m
    # d margin_m / d w = p_m ϕ(D_m),  d margin_m / d b = p_m
    weighted = labels * dL_dmargin               # (M,)
    grad_w = Phi.T @ weighted
    grad_b = weighted.sum()
    return np.concatenate([grad_w, [grad_b]])


# ─────────────────────────────────────────────
# 4.  Gradient descent solver (centralised)
# ─────────────────────────────────────────────

def gradient_descent(Phi, labels, alpha: float = 1e-3,
                     max_iter: int = 2000, tol: float = 1e-8,
                     seed: int = 0):
    q = Phi.shape[1]
    rng = np.random.default_rng(seed)
    wb = rng.standard_normal(q + 1) * 0.01

    cost_hist, grad_norm_hist = [], []

    for k in range(max_iter):
        g = logistic_grad(wb, Phi, labels)
        cost_hist.append(logistic_loss(wb, Phi, labels))
        grad_norm_hist.append(np.linalg.norm(g))

        wb -= alpha * g

        if grad_norm_hist[-1] < tol:
            print(f"  Converged at iteration {k+1}")
            break

    return wb, cost_hist, grad_norm_hist


# ─────────────────────────────────────────────
# 5.  Evaluation & visualisation
# ─────────────────────────────────────────────

def misclassification_rate(wb, Phi, labels):
    w, b = wb[:-1], wb[-1]
    preds = np.sign(Phi @ w + b)
    preds[preds == 0] = 1
    return np.mean(preds != labels) * 100


def plot_decision_boundary(X, labels, wb, phi_fn, title: str = ""):
    """Plot dataset and learned separating curve in original 2-D space."""
    x1_rng = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)
    x2_rng = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
    xx1, xx2 = np.meshgrid(x1_rng, x2_rng)
    grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    Phi_grid = phi_fn(grid)
    w, b = wb[:-1], wb[-1]
    scores = (Phi_grid @ w + b).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx1, xx2, scores, levels=[-1e9, 0, 1e9],
                colors=["#f7c6c7", "#c6daf7"], alpha=0.6)
    ax.contour(xx1, xx2, scores, levels=[0], colors="black", linewidths=1.5)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels,
                         cmap="bwr", edgecolors="k", s=20, alpha=0.8)
    p1 = mpatches.Patch(color="#c6daf7", label="Predicted +1")
    p2 = mpatches.Patch(color="#f7c6c7", label="Predicted -1")
    ax.legend(handles=[p1, p2], loc="upper right", fontsize=8)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_training_curves(cost_h, gnorm_h, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Centralised GD – {title}", fontsize=12, fontweight="bold")
    axes[0].semilogy(cost_h);        axes[0].set_title("Loss evolution")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("L(w,b)")
    axes[0].grid(True)
    axes[1].semilogy(gnorm_h);        axes[1].set_title("Gradient norm")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("‖∇L‖")
    axes[1].grid(True)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    M        = 500    # total dataset size
    alpha    = 5e-4   # step size
    max_iter = 3000

    mappings = get_mappings(GROUP_NUMBER)
    print(f"Group {GROUP_NUMBER} ({'Odd' if GROUP_NUMBER % 2 else 'Even'}) "
          f"→ using: {[n for n, _ in mappings]}\n")

    for map_name, phi_fn in mappings:
        print(f"═══ Feature mapping: {map_name} ═══")

        # true parameters (arbitrary ground truth)
        rng = np.random.default_rng(99)
        q = phi_fn(np.zeros((1, 2))).shape[1]
        w_true = rng.standard_normal(q);  w_true /= np.linalg.norm(w_true)
        b_true = rng.uniform(-0.5, 0.5)

        X, labels = generate_dataset(M, w_true, b_true, phi_fn, noise=0.02, seed=42)
        Phi = phi_fn(X)

        print(f"  Dataset: M={M}, q={q}, "
              f"pos={np.sum(labels==1)}, neg={np.sum(labels==-1)}")

        wb_opt, cost_h, gnorm_h = gradient_descent(
            Phi, labels, alpha=alpha, max_iter=max_iter)

        miss = misclassification_rate(wb_opt, Phi, labels)
        print(f"  Final loss      = {cost_h[-1]:.4f}")
        print(f"  Final grad norm = {gnorm_h[-1]:.2e}")
        print(f"  Misclassified   = {miss:.2f}%\n")

        tag = map_name.lower()

        fig_c = plot_training_curves(cost_h, gnorm_h, title=f"{map_name} (M={M})")
        fig_c.savefig(f"/mnt/user-data/outputs/task1_2_curves_{tag}.png",
                      dpi=150, bbox_inches="tight")

        fig_b = plot_decision_boundary(
            X, labels, wb_opt, phi_fn,
            title=f"Decision boundary – {map_name}  (miss={miss:.1f}%)")
        fig_b.savefig(f"/mnt/user-data/outputs/task1_2_boundary_{tag}.png",
                      dpi=150, bbox_inches="tight")

    plt.show()
    print("All plots saved to /mnt/user-data/outputs/")