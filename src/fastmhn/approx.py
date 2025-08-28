import numpy as np

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
    if max_cluster_size == None:
        max_cluster_size = d

    gradient = np.zeros((d, d))
    if verbose:
        print("Indices: gradient")
    for i in range(d):
        # base rate gradients
        columns = clustering_algorithm(
            theta, e1=i, e2=i, max_size=max_cluster_size
        )[0]
        g, _ = gradient_and_score(
            theta[np.ix_(columns, columns)], data[:, columns]
        )
        key = tuple(sorted(columns))
        gradient[i, i] = g[0, 0]

        if verbose:
            print(f"{i}, {i}: {gradient[i,i]} ({len(columns)} events)")
        for j in range(i + 1, d):
            # influence gradients
            columns = clustering_algorithm(
                theta, e1=i, e2=j, max_size=max_cluster_size
            )[0]
            g, _ = gradient_and_score(
                theta[np.ix_(columns, columns)], data[:, columns]
            )
            key = tuple(sorted(columns))
            gradient[i, j] = g[0, 1]
            gradient[j, i] = g[1, 0]
            if verbose:
                print(f"{i}, {j}: {gradient[i,j]} ({len(columns)} events)")
                print(f"{j}, {i}: {gradient[j,i]} ({len(columns)} events)")
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
    if max_cluster_size == None:
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
