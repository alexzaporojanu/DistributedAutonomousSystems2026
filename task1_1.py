import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


N = 5
IN = np.eye(N)
G = nx.path_graph(n=N-1)
G = nx.star_graph(n=N-1)
#G = nx.cycle_graph(n=N)

Adj = nx.adjacency_matrix(G).toarray()

print(G)
weightedAdj = np.zeros((N, N))
for i in range(N):
    N_i = np.nonzero(Adj[i])[0]
    deg_i = len(N_i)

    for j in N_i:
        N_j = np.nonzero(Adj[j])[0]
        deg_j = len(N_j)

        weightedAdj[i, j] = 1 / (1 + max([deg_i, deg_j]))

weightedAdj = weightedAdj + IN - np.diag(np.sum(weightedAdj, axis=0))


print(np.sum(weightedAdj, axis=0))
print(np.sum(weightedAdj, axis=1))


def cost_function(z, Q, r):
    val = 0.5 * Q * z * z + r * z
    grad = Q * z + r
    return val, grad

Q = np.zeros((N))
r = np.zeros((N))
for ii in range(N):
    Q[ii] = np.random.rand() + 1  # uniform in 1,2
    r[ii] = 10 * (np.random.rand() - 0.5)  # uniform in -5,5


print(Q)
print(r)
maxK = 1000

z = np.zeros((maxK, N))
s = np.zeros((maxK, N))
for ii in range(N):
    _, grad_ell_ii = cost_function(z[0, ii], Q[ii], r[ii])
    s[0, ii] = grad_ell_ii


cost = np.zeros((maxK))
gradient = np.zeros((maxK))
consensus = np.zeros((maxK))


stepsize = 2 * 1e-1
for k in range(maxK - 1):
    print(k)

    for ii in range(N):
        N_ii = np.nonzero(Adj[ii])[0]

        z[k + 1, ii] = weightedAdj[ii, ii] * z[k, ii]
        for jj in N_ii:
            z[k + 1, ii] += weightedAdj[ii, jj] * z[k, jj]

        z[k + 1, ii] += -stepsize * s[k, ii]

        _, grad_ell_ii_new = cost_function(z[k + 1, ii], Q[ii], r[ii])
        ell_ii, grad_ell_ii_old = cost_function(z[k, ii], Q[ii], r[ii])

        s[k + 1, ii] = weightedAdj[ii, ii] * s[k, ii]
        for jj in N_ii:
            s[k + 1, ii] += weightedAdj[ii, jj] * s[k, jj]

        s[k + 1, ii] += grad_ell_ii_new - grad_ell_ii_old

        cost[k] += ell_ii
        gradient[k] += grad_ell_ii_old

    consensus[k] = np.linalg.norm(z[k] - np.mean(z[k]))

fig, axes = plt.subplots(ncols=3, figsize=(10, 7))
ax = axes[0]
ax.plot(cost[:-1])
ax.set_title("Cost")
ax.grid()


ax = axes[1]
# ax.plot(np.abs(gradient[:-1]))
ax.semilogy(np.abs(gradient[:-1]))
ax.grid()
ax.set_title("Gradient")

ax = axes[2]
# ax.plot(consensus[:-1])
ax.semilogy(consensus[:-1])
ax.grid()
ax.set_title("Consensus")

plt.show()
