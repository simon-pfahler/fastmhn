import networkx as nx
import numpy as np
import pandas as pd
from Kmeans_function import Kmeans

def spectral_clustering(theta_matrix, e1, e2, max_size=None, k=None, verbose=False):

    """
    Performs spectral clustering based on a `theta` matrix, with the
    restriction that `e1` and `e2` have to be in the same cluster, and that
    their cluster contains at most `max_size` events.

    `theta`: dxd theta matrix
    `e1`: first event of the important event pair
    `e2`: second event of the important event pair
    `max_size`: maximum allowed size for the cluster containing `e1` and `e2`
    `verbose`: set to `True` to get more output
    """

    # Setze alle Diagonaleinträge auf 0
    np.fill_diagonal(theta_matrix, 0)
    # erstelle leere Liste für Indizes der isolierten events
    iso_event_idx_list = []

    # Speichere alle Indizes der isolierten events in iso_event_idx_list
    # und symmetrisiere die theta-matrix
    for i in range(theta_matrix.shape[0]):
        if np.all(theta_matrix[i, :] == 0) and np.all(theta_matrix[:, i] == 0):
            iso_event_idx_list.append(i)
        else:
            for j in range(i + 1, theta_matrix.shape[1]):
                if theta_matrix[i, j] != 0 or theta_matrix[j, i] != 0:
                    max_val = max(abs(theta_matrix[i, j]), abs(theta_matrix[j, i]))
                    theta_matrix[i, j] = max_val
                    theta_matrix[j, i] = max_val

    #### Überpüfe ob Startknoten über einen Pfad verbunden sind ####
    # erstelle ungerichteten Graphen aus neuer theta-matrix
    G = nx.from_numpy_array(np.array(theta_matrix))
    if nx.has_path(G, e1, e2):
        pass
    else:
        print("Warnung: Es existiert kein Pfad von {e1} zu {e2}.")
    # erstelle array mit indizes von 0 bis länge theta-matrix
    all_indices = set(range(theta_matrix.shape[0]))
    # filtere alle indizes der isolierten events heraus sodass array der nicht isolierten events überig bleibt
    non_iso_indices = sorted(all_indices - set(iso_event_idx_list))
    iso_event_idx = np.array(iso_event_idx_list, dtype=int)  # wandle liste in array um

    if np.isin(e1, iso_event_idx) and not np.isin(e2, iso_event_idx):
        print("Start event e1 ist isoliert")
        e1_A = None  # Index e1_A ist der Index des events e1 in Adjazenzmatrix A
        e2_A = non_iso_indices.index(e2)  # Index e2_A ist der Index des events e1 in Adjazenzmatrix A
    elif np.isin(e2, iso_event_idx) and not np.isin(e1, iso_event_idx):
        print("Start event e2 ist isoliert")
        e1_A = non_iso_indices.index(e1)  # Index e1_A ist der Index des events e1 in Adjazenzmatrix A
        e2_A = None  # Index e2_A ist der Index des events e2 in Adjazenzmatrix A
    elif np.isin([e1, e2], iso_event_idx).any():
        print("Start event e1 und e2 sind isoliert")
        e1_A = None
        e2_A = None
    else:
        e1_A = non_iso_indices.index(e1)
        e2_A = non_iso_indices.index(e2)

    # Erstelle Adajazenzmatrix aus theta_matrix durch
    # Entfernen aller Zeilen und Spalten aus theta_matrix, die in iso_event_idx_list stehen:
    A_temp = np.delete(theta_matrix, iso_event_idx, axis=0)  # Lösche alle Zeilen mit Indizes in iso_event_idx
    A = np.delete(A_temp, iso_event_idx, axis=1)

    # Berechne gewichtete Degree-Matrix
    D = np.diag(np.sum(A, axis=1))

    # Erzeuge unnormierte Laplace-Matrix L=D-A
    L = D - A
    # Berechne die normalisierte Laplace-Matrix L_sym L_sym = D^(-1/2) * L * D^(-1/2)
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(
        D)))  # np.diag(D)=> Vektor der Diagonalen, np.sqrt(np.diag(D)) liefert elementweise Wurzeln, 1/.. => Inverses , np.diag(.. => wandelt Vektor wieder in Matrix um
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # Ermittle Eigenwerte und Eigenvektoren von L_sym
    eigenwerte, eigenvektoren = np.linalg.eigh(L_sym)
    if k == None:
        # Berechne Differenzen zwischen benachbarten Eigenwerten
        gaps = np.diff(eigenwerte)

        # Wähle k basierend auf dem größten Sprung
        k = np.argmax(gaps) + 1

    # Wähle k die Eigenvektoren zu den k kleinsten Eigenwerten und normiere Eigenvektoren
    T = eigenvektoren[:, :k]
    T_norm = np.zeros_like(T)
    for i in range(T.shape[0]):
        if np.linalg.norm(T[i]) == 0:
            print("Error: Nullzeile in Adjazenzmatrix")
        else:
            T_norm[i] = T[i] / np.linalg.norm(T[i])

    # Erstelle Cluster mit Kmeans für alle nicht isolierten events
    clusters_reduced, cluster_labels_reduced, cent = Kmeans(T_norm, k, e1_A, e2_A, max_iterations=100, tolerance=1e-5)
    # Erweiter Cluster um ein Cluster in das alle isolierten events kommen
    clusters = [[] for _ in range(k + 1)]

    # erstelle leeres vektor array der Länge der theta-matrix
    cluster_labels_all = np.zeros(theta_matrix.shape[0], dtype=int)
    # erhöhe alle Clusterlabels um 1 sodass label 0 für isolierte events frei bleibt
    cluster_labels_reduced = cluster_labels_reduced + 1

    for i_reduced, i_original in enumerate(non_iso_indices):  # enumerate gibt index und zugehörigen Eintrag
        cluster_labels_all[i_original] = cluster_labels_reduced[
            i_reduced]  # Setzt das Clusterlabel der nicht isolierten events an die Stelle mit dem ursprünglichen Knotenindex

    for i in range(len(cluster_labels_all)):
        clusters[cluster_labels_all[i]].append(i)

    return clusters ,k