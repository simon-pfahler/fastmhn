import numpy as np
from fastmhn.explicit import create_full_Q, calculate_pTheta
from scipy.special import rel_entr
from Spectral_Clustering import spectral_clustering  # deine Clusterfunktion
import random

def kl_divergence(p, q):
    """
    Berechne die KL-Divergenz D_KL(p || q)

    Mathematisch:
        D_KL(p || q) = Σ p(x) * log(p(x)/q(x))

    `p` und `q` sind Wahrscheinlichkeitsverteilungen (z.B. Vektoren der Länge 2^d)
    """
    return np.sum(rel_entr(p, q))

def evaluate_clustering(theta_np, Q_full, p_full, e1, e2, k=None, max_size=6):
    """
    Führt Spectral Clustering auf gegebener Theta-Matrix aus und berechnet KL-Divergenz.

    Parameter:
        - theta_np: optimierte Theta-Matrix
        - Q_full: volle Q-Matrix aus theta_np
        - p_full: exakte Wahrscheinlichkeitsverteilung p_θ
        - e1, e2: zwei Events, die im selben Cluster sein sollen
        - k: Anzahl der Cluster (optional)
        - max_size: maximale Clustergröße

    Rückgabe:
        - KL-Divergenz
        - Cluster
        - p_full, p_clustered
        - e1, e2

    """
    d = theta_np.shape[0]

    clusters, _ = spectral_clustering(theta_np, e1=e1, e2=e2, max_size=max_size, k=k)

    p_clustered = np.array([1.0])
    for cluster in clusters:
        if len(cluster) == 0:
            continue

        theta_sub = theta_np[np.ix_(cluster, cluster)]
        # kein create_full_Q() mehr!
        p_sub = calculate_pTheta(theta_sub)
        p_clustered = np.kron(p_clustered, p_sub)

    p_clustered /= p_clustered.sum()
    kl = kl_divergence(p_full, p_clustered)

    print(f"e1: {e1}, e2: {e2}, KL-Divergenz: {kl:.5f}")
    return kl, clusters, p_full, p_clustered, e1, e2