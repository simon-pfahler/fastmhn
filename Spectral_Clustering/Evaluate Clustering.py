import numpy as np
import torch
from fastmhn.utility import create_indep_model
from fastmhn.approx import approx_gradient
from fastmhn.explicit import create_full_Q, calculate_pTheta
from scipy.special import rel_entr
from Spectral_Clustering import spectral_clustering  # deine Clusterfunktion
import random

# Deaktiviere automatische Gradientenberechnung (wir setzen sie manuell)
torch.set_grad_enabled(False)

def kl_divergence(p, q):
    """
    Berechne die KL-Divergenz D_KL(p || q)

    Mathematisch:
        D_KL(p || q) = Σ p(x) * log(p(x)/q(x))

    `p` und `q` sind Wahrscheinlichkeitsverteilungen (z.B. Vektoren der Länge 2^d)
    """
    return np.sum(rel_entr(p, q))


def evaluate_clustering(d=20, N=300, nr_iterations=200, reg=1e-2, k=None):
    """
    Führt den gesamten Ablauf zur Evaluation des Clusterings durch:
    1. Daten erzeugen
    2. θ (Theta-Matrix) lernen mit fastmhn
    3. Wahrscheinlichkeitsverteilung p_θ berechnen
    4. Clustering mit Spectral Clustering
    5. Tensorproduktverteilung p_clustered berechnen
    6. KL-Divergenz D_KL(p || p_clustered) berechnen
    """

    # === 1. Erzeuge zufällige Binärdaten (NxD-Matrix mit 0/1 Einträgen) ===
    data = np.random.randint(2, size=(N, d), dtype=np.int32)

    # === 2. Lerne eine Theta-Matrix mit dem "independent model" als Startwert ===
    theta = torch.tensor(create_indep_model(data), requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.1, betas=(0.7, 0.9), eps=1e-8)

    for t in range(nr_iterations):
        optimizer.zero_grad()

        # Berechne den Gradienten der approximativen Likelihood
        g = -torch.from_numpy(approx_gradient(theta.detach().numpy(), data))

        # Reguliere Off-Diagonalwerte mit L1-Regularisierung
        g += reg * torch.sign(theta * (1 - torch.eye(theta.shape[0])))

        # Setze den Gradienten und führe einen Optimierungsschritt aus
        theta.grad = g
        optimizer.step()

    # Konvertiere Theta zu einem Numpy-Array
    theta_np = theta.detach().numpy()

    # === 3. Berechne p_θ = (I - Q)^(-1) * p₀  (Stationäre Verteilung) ===
    Q_full = create_full_Q(theta_np)    # Erzeuge Q-Matrix aus θ
    p_full = calculate_pTheta(theta_np)     # Berechne daraus die Wahrscheinlichkeitsverteilung p_θ

    # === 4. Führe Spectral Clustering auf der θ-Matrix durch ===
    e1, e2 = random.sample(range(d), 2)   # Zwei zufällige Events, die im selben Cluster sein sollen
    clusters = spectral_clustering(theta_np, e1=e1, e2=e2,max_size=6, k=None)

    # === 5. Berechne p_clustered als Tensorprodukt der einzelnen Clusterverteilungen ===
    # p_clustered ≈ p_θ1 ⊗ p_θ2 ⊗ ... ⊗ p_θk
    p_clustered = np.array([1.0])  # Startwert fürs Tensorprodukt

    for cluster in clusters:
        if len(cluster) == 0:
            continue

        # Extrahiere Teilmatrix θ_cluster
        theta_sub = theta_np[np.ix_(cluster, cluster)]

        # Berechne Q und p_theta für den Cluster
        Q_sub = create_full_Q(theta_sub)
        p_sub = calculate_pTheta(theta_sub)


        # Tensorprodukt der bisherigen Approximation mit diesem Cluster
        p_clustered = np.kron(p_clustered, p_sub)

    # Normiere die approximative Verteilung zur Sicherheit
    p_clustered /= p_clustered.sum()

    # === 6. Berechne die KL-Divergenz D_KL(p_full || p_clustered) ===
    kl = kl_divergence(p_full, p_clustered)

    print(f"e1: {e1}, e2: {e2}, KL-Divergenz: {kl:.5f}")
    return kl, clusters, p_full, p_clustered, e1, e2


# === Hauptprogrammstart ===
if __name__ == "__main__":
    evaluate_clustering()
