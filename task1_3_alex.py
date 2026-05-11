import numpy as np
import matplotlib.pyplot as plt
from task1_1 import weightedAdj as W
from task1_2 import phi_parabola, logistic_grad, generate_dataset, logistic_loss, misclassification_rate

G=6
P=40 + (G % 3) * 10
def split_dataset_even_groups(X, y, N_agents,P, G, seed=42):
    """
    Divide il dataset in N subsets per i gruppi PARI (Even Groups).
    - Mantiene P% dei dati come Baseline Random (divisa equamente).
    - Il restante (100-P)% viene ordinato per la feature x2 e diviso in blocchi contigui.
    """
    rng = np.random.default_rng(seed)
    M = len(X)
    # 1. Calcolo percentuale P
    num_baseline = int(M * (P / 100))
    print(f"Group {G}: P = {P}%. Baseline points: {num_baseline}, Remaining points: {M - num_baseline}")
    
    # 2. Creiamo un array di indici da 0 a M-1 e lo mescoliamo
    indices = np.arange(M)
    rng.shuffle(indices)
    
    # Separiamo gli indici per la baseline e per il resto
    idx_baseline = indices[:num_baseline]
    idx_remaining = indices[num_baseline:]
    # Dividiamo gli indici della baseline equamente tra gli N agenti
    baseline_splits = np.array_split(idx_baseline, N_agents)
    
    # 3. Vertical Feature-Biased Split sul (100-P)% rimanente
    # Estraiamo i valori della feature x2 (seconda colonna, indice 1) per i dati rimanenti
    x2_remaining = X[idx_remaining, 1]
    
    # np.argsort restituisce l'ordine degli indici per avere l'array crescente
    sorted_order = np.argsort(x2_remaining)
    idx_remaining_sorted = idx_remaining[sorted_order]
    
    # Dividiamo gli indici ordinati in N blocchi contigui
    sorted_splits = np.array_split(idx_remaining_sorted, N_agents)
    
    # 4. Uniamo le due parti per ciascun agente
    agent_data = {}
    for i in range(N_agents):
        # Concateniamo la sua porzione di baseline e la sua porzione del blocco ordinato
        agent_final_indices = np.concatenate((baseline_splits[i], sorted_splits[i]))
        
        # Salviamo X e y per questo agente
        agent_data[i] = {
            'X': X[agent_final_indices],
            'y': y[agent_final_indices]
        }
        
    return agent_data

# ==========================================
# ESEMPIO DI UTILIZZO E TEST
# ==========================================
if __name__ == "__main__":
    G = 6       # Group number
    N = 5       # Number of agents
    M = 500     # Dataset size
    
    rng = np.random.default_rng(99)
    w_true = rng.standard_normal(3) # 3 for Parabola
    b_true = rng.uniform(-0.5, 0.5)
    X, y = generate_dataset(M, w_true, b_true, phi_parabola)
    
    # FIX: Added P to the arguments
    agents = split_dataset_even_groups(X, y, N, P, G)

    q = 3                   
    d = q + 1               
    max_iter = 2000
    stepsize = 0.05

    z = np.zeros((N, d))         
    s = np.zeros((N, d))         
    grad_old = np.zeros((N, d))  

    for i in range(N):
        agents[i]['Phi'] = phi_parabola(agents[i]['X'])

    for i in range(N):
        grad_old[i] = logistic_grad(z[i], agents[i]['Phi'], agents[i]['y'])
        s[i] = grad_old[i].copy()

    cost_history = []
    consensus_history = []
    grad_norm_history = [] # Added to track gradient norm

    print("Starting Distributed Gradient Tracking...")
    for k in range(max_iter):
        
        z_new = W @ z - stepsize * s
        
        grad_new = np.zeros((N, d))
        total_cost = 0
        
        for i in range(N):
            grad_new[i] = logistic_grad(z_new[i], agents[i]['Phi'], agents[i]['y'])
            # Track sum of local costs
            total_cost += logistic_loss(z_new[i], agents[i]['Phi'], agents[i]['y'])
            
        s_new = W @ s + grad_new - grad_old
        
        z = z_new
        s = s_new
        grad_old = grad_new
        
        # --- TRACK METRICS ---
        cost_history.append(total_cost)
        # Norm of the stacked gradients matrix
        grad_norm_history.append(np.linalg.norm(grad_new)) 
        # Deviation from the mean of all agents
        consensus_error = np.linalg.norm(z - np.mean(z, axis=0))
        consensus_history.append(consensus_error)

    # --- EVALUATION ---
    final_wb = np.mean(z, axis=0)
    Phi_all = phi_parabola(X) # Map the whole dataset to evaluate
    miss_rate = misclassification_rate(final_wb, Phi_all, y)
    
    print("--- Distributed Results ---")
    print(f"Final Total Loss:      {cost_history[-1]:.4f}")
    print(f"Final Gradient Norm:   {grad_norm_history[-1]:.2e}")
    print(f"Consensus Error:       {consensus_history[-1]:.2e}")
    print(f"Misclassification Rate:{miss_rate:.2f}%")

    # --- PLOTTING ---
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle(f"Task 1.3 - Distributed Logistic Regression (Parabola)")

    axes[0].plot(cost_history)
    axes[0].set_title("Total Cost")
    axes[0].set_xlabel("Iterations")
    axes[0].grid(True)

    axes[1].semilogy(grad_norm_history)
    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iterations")
    axes[1].grid(True)

    axes[2].semilogy(consensus_history)
    axes[2].set_title("Consensus Error")
    axes[2].set_xlabel("Iterations")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()