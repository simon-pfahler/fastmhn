import numpy as np
from mhn.training import likelihood_cmhn
from mhn.training.state_containers import StateContainer


def gradient_and_score(theta, data):
    return likelihood_cmhn.gradient_and_score(
        theta, StateContainer(data.astype(np.int32))
    )
