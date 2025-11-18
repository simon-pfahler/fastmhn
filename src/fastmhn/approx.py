import warnings

import numpy as np
from joblib import Parallel, delayed

from .clustering import hierarchical_clustering
from .exact import gradient_and_score
from .utility import create_pD


def __get_approx_gradient_and_score_contributions(
    theta,
    data,
    weights=None,
    clustering_algorithm=hierarchical_clustering,
    max_cluster_size=None,
):
    d = theta.shape[0]
    if max_cluster_size is None:
        max_cluster_size = d

    if weights is None:
        weights = np.ones(data.shape[0])

    # calculate gradient and score contributions for each patient individually
    def process_patient(nr_patient):
        # exact calculation if possible
        if np.sum(data[nr_patient]) <= max_cluster_size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g, s = gradient_and_score(
                    theta, data[nr_patient : nr_patient + 1]
                )
            g *= weights[nr_patient]
            s *= weights[nr_patient]
            return g, s

        # approximate calculation
        clustering = clustering_algorithm(
            theta,
            max_size=max_cluster_size,
            active_events=[i for i, x in enumerate(data[nr_patient]) if x == 1],
        )

        g_total = np.zeros_like(theta)
        s_total = 0
        for cluster in clustering:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g, s = gradient_and_score(
                    theta[np.ix_(cluster, cluster)],
                    data[nr_patient : nr_patient + 1, cluster],
                )
            s_total += s
            for cluster_i, i in enumerate(cluster):
                for cluster_j, j in enumerate(cluster):
                    g_total[i, j] += g[cluster_i, cluster_j]

        g_total *= weights[nr_patient]
        s_total *= weights[nr_patient]
        return g_total, s_total

    results = Parallel(n_jobs=-1)(
        delayed(process_patient)(i) for i in range(data.shape[0])
    )

    return tuple(zip(*results))


def approx_gradient_and_score(
    theta,
    data,
    weights=None,
    clustering_algorithm=hierarchical_clustering,
    max_cluster_size=None,
    verbose=False,
):
    """
    Calculates approximate gradients and scores using a provided clustering
    algorithm.

    `theta`: dxd theta matrix
    `data`: Nxd matrix containing the dataset
    `weights`: array of length N, used set the influence of individual samples
        on the score and gradient, default is `None`, which uses a weight of 1
        for all samples
    `clustering_algorithm`: Clustering algorithm to use, default is
        `hierarchical_clustering` from `fastmhn.clustering`
    `max_cluster_size`: maximal allowed size for clusters
    `verbose`: set to `True` to get more output
    """
    d = theta.shape[0]
    if max_cluster_size is None:
        max_cluster_size = d

    if weights is None:
        weights = np.ones(data.shape[0])

    if verbose:
        avg_MB = np.mean(np.sum(data, axis=1))
        max_MB = np.max(np.sum(data, axis=1))
        nr_samples_approx = np.sum(np.sum(data, axis=1) > max_cluster_size)
        print(
            f"Dataset information for gradient and score calculation:\n"
            f"\t{data.shape[0]} Patients\n"
            f"\tAverage mutational burden: {avg_MB}\n"
            f"\tMaximum mutational burden: {max_MB}\n"
            f"\tNumber of samples with MB > {max_cluster_size}: "
            f"{nr_samples_approx}"
        )

    gradients, scores = __get_approx_gradient_and_score_contributions(
        theta, data, weights, clustering_algorithm, max_cluster_size
    )
    gradient = np.sum(gradients, axis=0) / np.sum(weights)
    score = np.sum(scores) / np.sum(weights)

    return gradient, score
