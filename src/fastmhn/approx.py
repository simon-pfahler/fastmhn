import numpy as np
from joblib import Parallel, delayed

from .clustering import hierarchical_clustering
from .exact import gradient_and_score
from .utility import create_pD


def approx_gradient(
    theta,
    data,
    clustering_algorithm=hierarchical_clustering,
    max_cluster_size=None,
    verbose=False,
):
    """
    Calculates approximate gradients using the clustering provided by
    `fastmhn.clustering.get_cluster`.

    `theta`: dxd theta matrix
    `data`: Nxd matrix containing the dataset
    `clustering_algorithm`: Clustering algorithm to use, default is
        `hierarchical_clustering` from `fastmhn.clustering`
    `max_cluster_size`: maximal allowed size for clusters
    `verbose`: set to `True` to get more output
    """
    d = theta.shape[0]
    if max_cluster_size is None:
        max_cluster_size = d

    gradient = np.zeros((d, d))
    tasks = [(i, j) for i in range(d) for j in range(i, d)]

    # helper function for parallelization
    def _compute_gradient_block(i, j):
        columns = clustering_algorithm(
            theta, e1=i, e2=j, max_size=max_cluster_size
        )[0]
        g, _ = gradient_and_score(
            theta[np.ix_(columns, columns)], data[:, columns]
        )
        if i == j:
            return (i, g[0, 0], columns)
        else:
            return (i, j, g[0, 1], g[1, 0], columns)

    results = Parallel(n_jobs=-1)(
        delayed(_compute_gradient_block)(i, j) for i, j in tasks
    )

    for res in results:
        if len(res) == 3:
            i, gii, columns = res
            gradient[i, i] = gii
            if verbose:
                print(f"{i}, {i}: {gii} (Cluster {columns})")
        else:
            i, j, gij, gji, columns = res
            gradient[i, j] = gij
            gradient[j, i] = gji
            if verbose:
                print(f"{i}, {j}: {gij} (Cluster {columns})")
                print(f"{j}, {i}: {gji} (Cluster {columns})")

    return gradient


def approx_score(
    theta,
    data,
    clustering_algorithm=hierarchical_clustering,
    max_cluster_size=None,
    verbose=False,
):
    """
    Calculates approximate score using the clustering provided by
    `fastmhn.clustering.get_cluster`.

    `theta`: dxd theta matrix
    `data`: Nxd matrix containing the dataset
    `clustering_algorithm`: Clustering algorithm to use, default is
        `hierarchical_clustering` from `fastmhn.clustering`
    `max_cluster_size`: maximal allowed size for clusters
    `verbose`: set to `True` to get more output
    """
    d = theta.shape[0]
    if max_cluster_size is None:
        max_cluster_size = d

    score = 0
    clustering = clustering_algorithm(theta, max_size=max_cluster_size)
    # go through clusters and add their sub-scores
    for columns in clustering:
        _, s = gradient_and_score(
            theta[np.ix_(columns, columns)], data[:, columns]
        )
        score += s
    return score
