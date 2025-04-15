import numpy as np

def init_cent(data, k):
    n = data.shape[0]
    centroids = []

    # Wähle den ersten Centroid zufällig
    first_index = np.random.choice(n)
    centroids.append(data[first_index])

    # Weitere Centroids wählen
    for _ in range(1, k):
        # Berechne minimale Abstände zum nächsten schon gewählten Centroid
        distances = np.array([
            min(np.linalg.norm(x - c) ** 2 for c in centroids)
            for x in data
        ])

        # Wahrscheinlichkeiten proportional zu den quadrierten Abständen
        probabilities = distances / distances.sum()

        # Wähle neuen Centroid mit dieser Wahrscheinlichkeit
        next_index = np.random.choice(n, p=probabilities)
        centroids.append(data[next_index])

    return np.array(centroids)