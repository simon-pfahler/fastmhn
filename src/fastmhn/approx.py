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
    """
    Internal function to compute gradient and score contributions for each sample.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    data : numpy.ndarray
        Nxd matrix containing the dataset
    weights : numpy.ndarray, optional
        Array of length N containing sample weights
    clustering_algorithm : callable, optional
        Clustering algorithm to use. Default is hierarchical_clustering.
    max_cluster_size : int, optional
        Maximum allowed size for clusters. Default is None (uses d).

    Returns
    -------
    tuple of list
        (gradients, scores) where each is a list of contributions from each sample
    """
    d = theta.shape[0]
    if max_cluster_size is None:
        max_cluster_size = d

    if weights is None:
        weights = np.ones(data.shape[0])

    # calculate gradient and score contributions for each patient individually
    def process_patient(nr_patient):
        # exact calculation if possible
        if (
            np.sum(data[nr_patient]) <= max_cluster_size
            and data.shape[1] <= 256
        ):
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
            g_total[np.ix_(cluster, cluster)] += g

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
    Calculates approximate gradients and scores using a provided clustering algorithm.

    For each sample in the dataset:
    - If the number of active events <= max_cluster_size and d <= 256,
      uses exact calculation
    - Otherwise, uses hierarchical clustering to decompose the problem
      into smaller sub-problems that can be solved exactly

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    data : numpy.ndarray
        Nxd matrix containing the dataset
    weights : numpy.ndarray, optional
        Array of length N, used to set the influence of individual samples
        on the score and gradient. Default is None, which uses a weight of 1
        for all samples.
    clustering_algorithm : callable, optional
        Clustering algorithm to use. Default is hierarchical_clustering
        from fastmhn.clustering.
    max_cluster_size : int, optional
        Maximal allowed size for clusters. Default is None, which uses d.
    verbose : bool, optional
        If True, prints dataset information. Default is False.

    Returns
    -------
    tuple of numpy.ndarray and float
        (gradient, score) where gradient is a dxd matrix and score is a float
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
