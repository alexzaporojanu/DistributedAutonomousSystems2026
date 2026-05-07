import numpy as np
import matplotlib.pyplot as plt
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
    N = 10      # Number of agents
    M = 500     # Dataset size
    
    # Generiamo un dataset fittizio (sostituisci col tuo generate_dataset)
    rng = np.random.default_rng(42)
    X_dummy = rng.uniform(-3, 3, (M, 2))
    y_dummy = np.ones(M) # labels fittizie per ora
    
    # Chiamiamo la funzione di split
    agents = split_dataset_even_groups(X_dummy, y_dummy, N, G)
    
    # Stampiamo un recap per controllare che i conti tornino
    for i in range(N):
        print(f"Agent {i} has received {len(agents[i]['X'])} data points.")
        
   # --- VISUALIZZAZIONE VELOCE ---
    plt.figure(figsize=(8, 6))
    
    # Genera una palette di colori automatica in base al numero di agenti N
    # 'tab10' ha 10 colori molto distinti. Se N > 10, usa 'tab20' o 'viridis'
    cmap = plt.get_cmap('tab10') 
    
    for i in range(N):
        # Usiamo cmap(i % 10) per evitare errori anche se N è altissimo
        plt.scatter(agents[i]['X'][:, 0], agents[i]['X'][:, 1], 
                    color=cmap(i % 10), label=f'Agent {i+1}', s=15, alpha=0.7)
        
    plt.title("Dataset Split: Baseline (Random) + Vertical Split (x2)")
    plt.xlabel("x1")
    plt.ylabel("x2 (Sorting Feature)")
    plt.legend()
    plt.grid(True)
    plt.show()
