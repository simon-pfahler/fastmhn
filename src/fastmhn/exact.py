from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mhn.training import likelihood_cmhn
from mhn.training.state_containers import StateContainer


def gradient_and_score(
    theta: NDArray[np.float64], data: NDArray[np.int32]
) -> Tuple[NDArray[np.float64], float]:
    """
    Calculates the gradient and score using the mhn package's likelihood_cmhn.

    This is an alternative implementation that delegates to the mhn library.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    data : numpy.ndarray
        Nxd matrix containing the dataset (binary: 0 or 1)

    Returns
    -------
    tuple of numpy.ndarray and float
        (gradient, score) where gradient is a dxd matrix and score is a float
    """
    return likelihood_cmhn.gradient_and_score(
        theta, StateContainer(data.astype(np.int32))
    )
