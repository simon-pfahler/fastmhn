import matplotlib.pyplot as plt
import random
import os
import pickle

from Evaluate_Clustering_KL import evaluate_clustering
from optimize_theta import optimize_theta
from Spectral_Clustering import spectral_clustering
from plot_graphs_grid import plot_clustered_graph_grid

# === Parameter ===
d = 10               # Anzahl der Events
N = 300              # Anzahl der Datenpunkte
reg = 1e-2           # Regularisierungsparameter
nr_iterations = 200  # Optimierungsschritte
k_range = 5          # Bereich um k0, der getestet wird
max_size = 6         # Max. Clustergröße für Cluster mit e1/e2

# === Speicherort für Theta-Daten ===
theta_file = f"theta_data_d{d}_N{N}.pkl"

# ===  Frage, ob Theta neu generiert werden soll ===
if os.path.exists(theta_file):
    use_existing = input(f"❓ Existierende Theta-Matrix gefunden ({theta_file}). Neu generieren? (y/n): ").lower()
else:
    use_existing = 'y'

if use_existing == 'y':
    print("🔧 Optimiere Theta-Matrix neu...")
    theta_np, Q_full, p_full = optimize_theta(d=d, N=N, nr_iterations=nr_iterations, reg=reg)
    with open(theta_file, 'wb') as f:
        pickle.dump((theta_np, Q_full, p_full), f)
    print(f"💾 Theta-Daten gespeichert in {theta_file}")
else:
    print("📂 Lade gespeicherte Theta-Daten...")
    with open(theta_file, 'rb') as f:
        theta_np, Q_full, p_full = pickle.load(f)

# ===  Wähle zwei feste Events (e1, e2) ===
e1, e2 = random.sample(range(d), 2)
print(f"🎯 Fixierte Events: e1 = {e1}, e2 = {e2}")

# ===  Ermittle Startwert k₀ aus Spectral Clustering ===
clusters, k0 = spectral_clustering(theta_np.copy(), e1, e2, max_size=max_size, k=None)
print(f"📌 Startwert k₀ aus Gap-Heuristik: {k0}")

# ===  Teste verschiedene k-Werte um k0 ===
k_values = list(range(max(2, k0 - k_range), k0 + k_range + 1))
best_kl = float("inf")
best_k = None
best_clusters = None
kl_history = []
cluster_labelings = []

for k in k_values:
    print(f"\n🔍 Teste k = {k}")
    clusters, k_used = spectral_clustering(theta_np.copy(), e1, e2, max_size=max_size, k=k)

    for cluster in clusters:
        if e1 in cluster and e2 in cluster:
            if len(cluster) > max_size and 0 not in cluster:
                print(f"⚠️ Cluster mit e1 und e2 überschreitet max_size={max_size}, wird ignoriert.")
                break
            try:
                result = evaluate_clustering(theta_np, Q_full, p_full, e1, e2, k=k)
                kl = result[0]
                kl_history.append((k, kl))
                cluster_labelings.append((k, result[1]))  # Speichere Cluster
                if kl < best_kl:
                    best_kl = kl
                    best_k = k
                    best_clusters = result[1]
            except Exception as e:
                print(f"❌ Fehler bei k={k}: {e}")
            break

# ===  Ausgabe ===
print("\n✅ Optimierung abgeschlossen!")
print(f"Bestes k: {best_k}, KL-Divergenz: {best_kl:.5f}")

# === Schritt 6: Plot der KL-Divergenzen ===
if kl_history:
    k_list, kl_list = zip(*kl_history)
    plt.figure(figsize=(8, 5))
    plt.plot(k_list, kl_list, marker="o")
    plt.xlabel("Anzahl der Cluster (k)")
    plt.ylabel("KL-Divergenz")
    plt.title(f"Optimales k = {best_k}, KL = {best_kl:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("❌ Keine gültigen KL-Werte gefunden – überprüfe Clustering.")

# ===  Zeichne Clustergraphen für alle getesteten k ===
print("\n📊 Visualisiere alle Clustergraphen...")
plot_clustered_graph_grid(theta_np.copy(), cluster_labelings)
