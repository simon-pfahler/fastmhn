import numpy as np

from .approx import approx_gradient_and_score
from .clustering import hierarchical_clustering
from .utility import adam, cmhn_from_omhn, create_indep_model


def learn_mhn(
    data,
    weights=None,
    reg=1e-2,
    gradient_and_score_params={},
    theta_init=None,
    adam_params={},
):
    """
    Learn an MHN given some data.

    `data`: Nxd matrix containing the dataset
    `weights`: array of length N, used set the influence of individual samples
        on the score and gradient, default is `None`, which uses a weight of 1
        for all samples
    `reg`: Regularization strength
    `gradient_and_score_params`: Parameters passed to gradient_and_score call
    `theta_init`: Initial theta, default is the independence model
    `clustering_algorithm`: Clustering algorithm to use, default is
        `hierarchical_clustering` from `fastmhn.clustering`
    `adam_params`: Parameters passed to adam
    """

    d = data.shape[1]
    N = data.shape[0]

    # >>> handle absent events
    if np.any(np.sum(data, axis=0) == 0):
        indices = np.where(np.sum(data, axis=0) == 0)[0]
        subdata = np.delete(data, indices, axis=1)

        if theta_init is not None:
            subtheta_init = np.delete(
                np.delete(theta_init, indices, axis=0), indices, axis=1
            )
        else:
            subtheta_init = None
        subtheta = learn_mhn(
            subdata,
            weights,
            reg,
            gradient_and_score_params,
            subtheta_init,
            adam_params,
        )

        theta = np.zeros((d, d))

        rows_and_cols = [i for i in range(d) if i not in indices]

        theta[np.ix_(rows_and_cols, rows_and_cols)] = subtheta

        for i in indices:
            theta[i, i] = np.log(np.log(2) / N)
        # MISSING: set the diagonal entries to sensible values

        return theta
    # <<< handle absent events

    # >>> initialization
    if theta_init is None:
        theta_init = create_indep_model(data, weights=weights)

    gradient_and_score_params.setdefault(
        "clustering_algorithm", hierarchical_clustering
    )
    gradient_and_score_params.setdefault(
        "max_cluster_size", np.max(np.sum(data, axis=1))
    )

    adam_params.setdefault("alpha", 0.1)
    adam_params.setdefault("beta1", 0.7)
    adam_params.setdefault("beta2", 0.9)
    adam_params.setdefault("eps", 1e-8)
    adam_params.setdefault("verbose", True)
    # <<< initialization

    grad_and_score_func = lambda theta: approx_gradient_and_score(
        theta, data, weights=weights, **gradient_and_score_params
    )

    regularization_mask = np.ones_like(theta_init, dtype=bool)
    regularization_mask ^= np.eye(d, dtype=bool)

    reg_grad_func = lambda theta: -reg * regularization_mask * np.sign(theta)

    theta = adam(theta_init, grad_and_score_func, reg_grad_func, **adam_params)

    return theta


def learn_omhn(
    data,
    weights=None,
    reg=1e-2,
    gradient_and_score_params={},
    theta_init=None,
    adam_params={},
):
    """
    Learn an MHN given some data.

    `data`: Nxd matrix containing the dataset
    `weights`: array of length N, used set the influence of individual samples
        on the score and gradient, default is `None`, which uses a weight of 1
        for all samples
    `reg`: Regularization strength
    `gradient_and_score_params`: Parameters passed to gradient_and_score call
    `theta_init`: Initial theta, default is the independence model
    `clustering_algorithm`: Clustering algorithm to use, default is
        `hierarchical_clustering` from `fastmhn.clustering`
    `adam_params`: Parameters passed to adam
    """

    d = data.shape[1]
    N = data.shape[0]

    # >>> handle absent events
    if np.any(np.sum(data, axis=0) == 0):
        indices = np.where(np.sum(data, axis=0) == 0)[0]
        subdata = np.delete(data, indices, axis=1)

        if theta_init is not None:
            subtheta_init = np.delete(
                np.delete(theta_init, indices, axis=0), indices, axis=1
            )
        else:
            subtheta_init = None
        subtheta = learn_omhn(
            subdata,
            weights,
            reg,
            gradient_and_score_params,
            subtheta_init,
            adam_params,
        )

        theta = np.zeros((d + 1, d))

        rows = [i for i in range(d + 1) if i not in indices]
        cols = [i for i in range(d) if i not in indices]

        theta[np.ix_(rows, cols)] = subtheta

        for i in indices:
            theta[i, i] = np.log(np.log(2) / N)
        # MISSING: set the diagonal entries to sensible values

        return theta
    # <<< handle absent events

    # >>> initialization
    if theta_init is None:
        theta_init = np.zeros((d + 1, d))
        theta_init[:d] = create_indep_model(data, weights=weights)

    gradient_and_score_params.setdefault(
        "clustering_algorithm", hierarchical_clustering
    )
    gradient_and_score_params.setdefault(
        "max_cluster_size", np.max(np.sum(data, axis=1))
    )

    adam_params.setdefault("alpha", 0.1)
    adam_params.setdefault("beta1", 0.7)
    adam_params.setdefault("beta2", 0.9)
    adam_params.setdefault("eps", 1e-8)
    adam_params.setdefault("verbose", True)
    # <<< initialization

    def grad_and_score_func(theta):
        ctheta = cmhn_from_omhn(theta)
        g = np.zeros_like(theta)
        g[:d], s = approx_gradient_and_score(
            ctheta, data, weights=weights, **gradient_and_score_params
        )

        g[d] = -np.einsum("ij->j", g[:d] * (1 - np.eye(g.shape[1])))

        return g, s

    regularization_mask = np.ones_like(theta_init, dtype=bool)
    regularization_mask[:d] ^= np.eye(d, dtype=bool)

    reg_grad_func = lambda theta: -reg * regularization_mask * np.sign(theta)

    theta = adam(theta_init, grad_and_score_func, reg_grad_func, **adam_params)

    return theta
