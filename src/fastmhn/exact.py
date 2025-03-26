from mhn.training import likelihood_cmhn
from mhn.training.state_containers import StateContainer


def gradient_and_score(theta, data):
    """
    Performs an exact gradient and score calculation for a `theta` matrix and a
    dataset.

    `theta`: dxd theta matrix
    `data`: Nxd matrix containing the dataset
    """
    return likelihood_cmhn.gradient_and_score(theta, StateContainer(data))
