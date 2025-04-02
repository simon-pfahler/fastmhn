import numpy as np

from .clustering import get_clusters
from .exact import gradient_and_score
from .utility import create_pD


def approx_gradient(theta, data, max_cluster_size=None):
    """
    Calculates approximate gradients using the clustering provided by
    `fastmhn.clustering.get_cluster`.

    `theta`: dxd theta matrix
    `data`: Nxd matrix containing the dataset
    `max_cluster_size`: maximal allowed size for clusters
    """
    d = theta.shape[0]
    if max_cluster_size == None:
        max_cluster_size = d

    gradient = np.zeros((d, d))
    for i in range(d):
        # base rate gradients
        columns = get_clusters(theta, i, i, max_cluster_size)[0]
        g, _ = gradient_and_score(
            theta[np.ix_(columns, columns)], create_pD(data[:, columns])
        )
        key = tuple(sorted(columns))
        gradient[i, i] = g[0, 0]
        for j in range(i + 1, d):
            # influence gradients
            columns = get_clusters(theta, i, j, max_cluster_size)[0]
            g, _ = gradient_and_score(
                theta[np.ix_(columns, columns)], create_pD(data[:, columns])
            )
            key = tuple(sorted(columns))
            gradient[i, j] = g[0, 1]
            gradient[j, i] = g[1, 0]
    return gradient
