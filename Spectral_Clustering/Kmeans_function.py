import numpy as np
from Init_centroids_function import init_cent
# data: nxd-matrix wobei n-te Zeile ein Knoten
# k: Anzahl Cluster
# max_iterations: Max Anzahl Iterationen bevor Abbruch
# tolerance: Änderung der Cluster-Zentren (Centroids) sollte kleiner als tolerance sein
def Kmeans(data, k, must_link_1=None, must_link_2=None,max_iterations=100, tolerance=1e-4):
    # int_cent wählt k gleichmäßig verteilte Cluster-Zentren
    cent = init_cent(data, k)
    cent_temp = np.zeros_like(cent)
    cent_konv = np.ones(k)
    konv = 1

    dist = np.zeros((data.shape[0], k))   # (n, k)
    min_dist = np.zeros(data.shape[0])    # (n,)

    iteration = 0
    while konv >= tolerance and iteration < max_iterations:
        # Abstände der Knotenpunkten zu den Cluster-Zenteren berechnen und in dist speichern
        for i in range(data.shape[0]):      # n
            for j in range(k):             # k
                dist[i, j] = np.linalg.norm(cent[j, :] - data[i, :]) # Zeile entspricht Knoten , Spalte entpricht Cluster

        # Clusters & Labels zurücksetzen
        cluster_labels = np.zeros(data.shape[0])  # (n,)
        clusters = [[] for _ in range(k)]

        # Zuordnung der Knoten zu den Cluster-Zentren mit min. euklidischen Abstand
        for i in range(dist.shape[0]):          # n
            cluster_labels[i] = np.argmin(dist[i, :]) # speichert Clusterindex also Spaltenindex an der i-ten Stelle
            min_dist[i] = np.min(dist[i, :])
            clusters[int(cluster_labels[i])].append(data[i])

            #  Must-Link Constraint: stelle sicher, dass must_link_1 und must_link_2 im selben Cluster sind
            if must_link_1 is not None and must_link_2 is not None:
                target_cluster = int(cluster_labels[must_link_1])

                if int(cluster_labels[must_link_2]) != target_cluster:
                    old_cluster = int(cluster_labels[must_link_2])

                    # Sicher entfernen mit array_equal statt remove()
                    clusters[old_cluster] = [
                        point for point in clusters[old_cluster]
                        if not np.array_equal(point, data[must_link_2])
                    ]

                    clusters[target_cluster].append(data[must_link_2])
                    cluster_labels[must_link_2] = target_cluster
        # Neue Centroids
        for i in range(k):
            if len(clusters[i]) > 0:
                cent_temp[i, :] = np.mean(clusters[i], axis=0)
            else:
                cent_temp[i, :] = cent[i, :]
            cent_konv[i] = np.linalg.norm(cent[i, :] - cent_temp[i, :])

        knov = np.sum(cent_konv)
        cent = cent_temp.copy()   # Kopie
        konv = knov

        iteration += 1


    return clusters, cluster_labels, cent







