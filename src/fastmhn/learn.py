import numpy as np

from .approx import approx_gradient_and_score
from .clustering import hierarchical_clustering
from .utility import adam, cmhn_from_omhn, create_indep_model


def learn_mhn(
    data,
    reg=1e-2,
    gradient_and_score_params={},
    theta_init=None,
    adam_params={},
):
    """
    Learn an MHN given some data.

    `data`: Nxd matrix containing the dataset
    `reg`: Regularization strength
    `gradient_and_score_params`: Parameters passed to gradient_and_score call
    `theta_init`: Initial theta, default is the independence model
    `clustering_algorithm`: Clustering algorithm to use, default is
        `hierarchical_clustering` from `fastmhn.clustering`
    `adam_params`: Parameters passed to adam
    `verbose`: Print intermediate information (default: `False`)
    """

    d = data.shape[1]

    # >>> initialization
    if theta_init is None:
        theta_init = create_indep_model(data)

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
        theta, data, **gradient_and_score_params
    )

    regularization_mask = np.ones_like(theta_init, dtype=bool)
    regularization_mask ^= np.eye(d, dtype=bool)

    reg_grad_func = lambda theta: -reg * regularization_mask * np.sign(theta)

    theta = adam(theta_init, grad_and_score_func, reg_grad_func, **adam_params)

    return theta


def learn_omhn(
    data,
    reg=1e-2,
    gradient_and_score_params={},
    theta_init=None,
    adam_params={},
):
    """
    Learn an MHN given some data.

    `data`: Nxd matrix containing the dataset
    `reg`: Regularization strength
    `gradient_and_score_params`: Parameters passed to gradient_and_score call
    `theta_init`: Initial theta, default is the independence model
    `clustering_algorithm`: Clustering algorithm to use, default is
        `hierarchical_clustering` from `fastmhn.clustering`
    `adam_params`: Parameters passed to adam
    `verbose`: Print intermediate information (default: `False`)
    """

    d = data.shape[1]

    # >>> initialization
    if theta_init is None:
        theta_init = np.zeros((d + 1, d))
        theta_init[:d] = create_indep_model(data)

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
            ctheta, data, **gradient_and_score_params
        )

        g[d] = -np.einsum("ij->j", g[:d] * (1 - np.eye(g.shape[1])))

        return g, s

    regularization_mask = np.ones_like(theta_init, dtype=bool)
    regularization_mask[:d] ^= np.eye(d, dtype=bool)

    reg_grad_func = lambda theta: -reg * regularization_mask * np.sign(theta)

    theta = adam(theta_init, grad_and_score_func, reg_grad_func, **adam_params)

    return theta
