import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import ceil, sqrt

def plot_clustered_graph_grid(adj_matrix, cluster_labelings, node_size=400):
    """
    Plotte mehrere Clustergraphen nebeneinander (für verschiedene k-Werte).

    `adj_matrix`: Adjazenzmatrix
    `cluster_labelings`: Liste von (k, clusters)
    """

    num_graphs = len(cluster_labelings)
    cols = ceil(sqrt(num_graphs))
    rows = ceil(num_graphs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = axes.flatten()

    for ax, (k_val, clusters) in zip(axes, cluster_labelings):
        labels = np.zeros(adj_matrix.shape[0], dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for node in cluster:
                labels[node] = cluster_idx

        valid_nodes = np.where(labels > 0)[0]
        G = nx.from_numpy_array(adj_matrix)
        pos = nx.spring_layout(G.subgraph(valid_nodes), seed=42)

        cluster_colors = [labels[n] for n in valid_nodes]

        nx.draw_networkx_nodes(G, pos, nodelist=valid_nodes,
                               node_color=cluster_colors,
                               node_size=node_size, ax=ax, cmap=plt.cm.tab10)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), ax=ax, alpha=0.3)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color='black', font_size=8)

        ax.set_title(f"Clustergraph für k = {k_val}")
        ax.axis('off')

    for i in range(len(cluster_labelings), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()