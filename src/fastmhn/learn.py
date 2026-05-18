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
    Learn an MHN (Mutational Hierarchical Network) model from data.

    This function fits an MHN model to the given dataset using the Adam optimizer.
    It handles absent events (events that never occur in the data) by learning
    a sub-model on the present events and then extending it.

    Parameters
    ----------
    data : numpy.ndarray
        Nxd matrix containing the dataset (binary: 0 or 1)
    weights : numpy.ndarray, optional
        Array of length N, used to set the influence of individual samples
        on the score and gradient. Default is None, which uses a weight of 1
        for all samples.
    reg : float, optional
        L1 regularization strength. Default is 1e-2.
    gradient_and_score_params : dict, optional
        Parameters passed to the gradient_and_score function.
        Default is an empty dict.
    theta_init : numpy.ndarray, optional
        Initial theta matrix. Default is None, which uses the independence model.
    adam_params : dict, optional
        Parameters passed to the Adam optimizer.
        Default is an empty dict (uses default Adam parameters).

    Returns
    -------
    numpy.ndarray
        dxd learned theta matrix for the MHN model
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
    Learn an oMHN (observation Mutational Hierarchical Network) model from data.

    oMHN extends MHN by adding observation rates that model the probability
    of observing each event. This function fits an oMHN model to the given
    dataset using the Adam optimizer.

    Parameters
    ----------
    data : numpy.ndarray
        Nxd matrix containing the dataset (binary: 0 or 1)
    weights : numpy.ndarray, optional
        Array of length N, used to set the influence of individual samples
        on the score and gradient. Default is None, which uses a weight of 1
        for all samples.
    reg : float, optional
        L1 regularization strength. Default is 1e-2.
    gradient_and_score_params : dict, optional
        Parameters passed to the gradient_and_score function.
        Default is an empty dict.
    theta_init : numpy.ndarray, optional
        Initial theta matrix of shape (d+1)xd. Default is None, which initializes
        with zeros for observation rates and independence model for the rest.
    adam_params : dict, optional
        Parameters passed to the Adam optimizer.
        Default is an empty dict (uses default Adam parameters).

    Returns
    -------
    numpy.ndarray
        (d+1)xd learned theta matrix for the oMHN model
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
