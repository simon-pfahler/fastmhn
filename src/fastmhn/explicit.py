from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .utility import create_pD, jacobi

# Precompute bit masks for efficiency
_bit_masks_cache: dict[int, Tuple[NDArray[np.bool_], NDArray[np.bool_]]] = {}


def _get_bit_masks(d: int) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Get or create bit masks for a given dimension d.

    Parameters
    ----------
    d : int
        Dimension (number of events)

    Returns
    -------
    tuple of numpy.ndarray
        (masks_0, masks_1) where masks_0[j] is a boolean array indicating
        positions where bit j is 0, and masks_1[j] indicates positions where bit j is 1
    """
    if d not in _bit_masks_cache:
        n_states = 2**d
        # For each bit position j (0 to d-1), create masks for bit (d-1-j)
        masks_0: list[NDArray[np.bool_]] = []
        masks_1: list[NDArray[np.bool_]] = []
        for j in range(d):
            bit_pos = d - 1 - j
            mask0 = (np.arange(n_states) & (1 << bit_pos)) == 0
            mask1 = (np.arange(n_states) & (1 << bit_pos)) != 0
            masks_0.append(mask0)
            masks_1.append(mask1)
        _bit_masks_cache[d] = (np.array(masks_0), np.array(masks_1))
    return _bit_masks_cache[d]


def calculate_pTheta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculates the time-marginalized probability distribution pTheta for a given theta matrix.

    This solves (I - Q) pTheta = p0, where p0 is the initial state (all zeros).

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix

    Returns
    -------
    numpy.ndarray
        Array of length 2^d containing the probability distribution pTheta
    """
    d = theta.shape[0]

    p0 = np.zeros(2**d, dtype=np.float64)
    p0[0] = 1

    op_diag = lambda x: apply_eye_minus_Q_diag(theta, x)
    op_offdiag = lambda x: apply_eye_minus_Q_offdiag(theta, x)

    pTheta = jacobi(op_diag, op_offdiag, p0)

    return pTheta


def score(theta: NDArray[np.float64], pD: NDArray[np.float64]) -> float:
    """
    Calculates the log-likelihood score for a given theta matrix and data distribution.

    score = sum_x pD[x] * ln(pTheta[x])

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    pD : numpy.ndarray
        Array of length 2^d containing the data distribution

    Returns
    -------
    float
        The log-likelihood score
    """
    d = theta.shape[0]

    pTheta = calculate_pTheta(theta)

    return float(np.dot(pD, np.log(pTheta)))


def gradient_and_score(
    theta: NDArray[np.float64], data: NDArray[np.int32]
) -> Tuple[NDArray[np.float64], float]:
    """
    Calculates the gradient and score for a given theta matrix and data distribution.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    data : numpy.ndarray
        Nxd matrix containing the dataset

    Returns
    -------
    tuple of numpy.ndarray and float
        (gradient, score) where gradient is a dxd matrix and score is a float
    """
    d = theta.shape[0]

    pD = create_pD(data)

    pTheta = calculate_pTheta(theta)

    score = float(np.dot(pD, np.log(pTheta)))

    op_diag = lambda x: apply_eye_minus_Q_diag(theta, x)
    op_offdiag = lambda x: apply_eye_minus_Q_offdiag(theta, x, transpose=True)
    q = jacobi(op_diag, op_offdiag, pD / pTheta)

    gradient = np.zeros((d, d), dtype=np.float64)

    # Get precomputed masks for gradient computation
    masks_0, masks_1 = _get_bit_masks(d)

    for i in range(d):
        h = apply_Qdiff_ii(theta, pTheta, i)
        r = q * h
        for j in range(d):
            if i == j:
                gradient[i, i] = np.sum(r)
                continue
            mask = masks_1[j]  # (2**d >> (j + 1)) != 0
            gradient[i, j] = np.sum(r[mask])

    return gradient, score


def apply_eye_minus_Q(
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    transpose: bool = False,
) -> NDArray[np.float64]:
    """
    Calculates (I-Q) @ x for a given theta matrix and vector x.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    x : numpy.ndarray
        Vector of length 2^d
    transpose : bool, optional
        If True, calculates (I-Q)^T @ x instead. Default is False.

    Returns
    -------
    numpy.ndarray
        Result vector of length 2^d
    """
    d = theta.shape[0]
    n_states = 2**d
    bigTheta = np.exp(theta)

    # Get precomputed masks
    masks_0, masks_1 = _get_bit_masks(d)

    b = x.copy()
    for i in range(d):
        v = x.copy()
        for j in range(d):
            mask0 = masks_0[j]
            mask1 = masks_1[j]
            if i == j:
                if transpose:
                    v[mask0] = (
                        -bigTheta[i, i] * v[mask0] + bigTheta[i, i] * v[mask1]
                    )
                    v[mask1] = 0
                else:
                    v[mask0] *= -bigTheta[i, i]
                    v[mask1] = -v[mask0]
            else:
                v[mask1] *= bigTheta[i, j]
        b -= v
    return b


def apply_eye_minus_Q_diag(
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    transpose: bool = False,
) -> NDArray[np.float64]:
    """
    Calculates diag(I-Q) @ x for a given theta matrix and vector x.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    x : numpy.ndarray
        Vector of length 2^d
    transpose : bool, optional
        Does not affect the result (only added for interface consistency with
        apply_eye_minus_Q and apply_eye_minus_Q_offdiag). Default is False.

    Returns
    -------
    numpy.ndarray
        Result vector of length 2^d
    """
    d = theta.shape[0]
    bigTheta = np.exp(theta)

    # Get precomputed masks
    masks_0, masks_1 = _get_bit_masks(d)

    b = x.copy()
    for i in range(d):
        v = x.copy()
        for j in range(d):
            mask0 = masks_0[j]
            mask1 = masks_1[j]
            if i == j:
                v[mask0] *= -bigTheta[i, i]
                v[mask1] = 0
            else:
                v[mask1] *= bigTheta[i, j]
        b -= v
    return b


def apply_eye_minus_Q_offdiag(
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    transpose: bool = False,
) -> NDArray[np.float64]:
    """
    Calculates offdiag(I-Q) @ x for a given theta matrix and vector x.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    x : numpy.ndarray
        Vector of length 2^d
    transpose : bool, optional
        If True, calculates offdiag(I-Q)^T @ x. Default is False.

    Returns
    -------
    numpy.ndarray
        Result vector of length 2^d
    """
    d = theta.shape[0]
    bigTheta = np.exp(theta)

    # Get precomputed masks
    masks_0, masks_1 = _get_bit_masks(d)

    b = np.zeros_like(x)
    for i in range(d):
        v = x.copy()
        for j in range(d):
            mask0 = masks_0[j]
            mask1 = masks_1[j]
            if i == j:
                if transpose:
                    v[mask0] = bigTheta[i, i] * v[mask1]
                    v[mask1] = 0
                else:
                    v[mask1] = bigTheta[i, i] * v[mask0]
                    v[mask0] = 0
            else:
                v[mask1] *= bigTheta[i, j]
        b -= v
    return b


def apply_Qdiff_ii(
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    i: int,
) -> NDArray[np.float64]:
    """
    Calculates dQ/d(theta_ii) @ x for a given theta matrix and vector x.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix
    x : numpy.ndarray
        Vector of length 2^d
    i : int
        Index of theta matrix element to take derivative with respect to

    Returns
    -------
    numpy.ndarray
        Result vector of length 2^d
    """
    d = theta.shape[0]
    bigTheta = np.exp(theta)

    # Get precomputed masks
    masks_0, masks_1 = _get_bit_masks(d)

    v = x.copy()
    for k in range(d):
        mask0 = masks_0[k]
        mask1 = masks_1[k]
        if k == i:
            v[mask0] *= -bigTheta[i, i]
            v[mask1] = -v[mask0]
        else:
            v[mask1] *= bigTheta[i, k]
    return v


def create_full_Q(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Creates the full Q matrix for a given theta matrix.

    The Q matrix is the generator of the continuous-time Markov process.
    For MHN models, Q has a specific block structure based on theta.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix

    Returns
    -------
    numpy.ndarray
        2^d x 2^d Q matrix
    """
    d = theta.shape[0]
    bigTheta = np.exp(theta)
    Q = np.zeros((2**d, 2**d), dtype=np.float64)
    for i in range(d):
        term = np.ones(1)
        for j in range(d):
            if i == j:
                term = np.kron(
                    term, np.array([[-bigTheta[i, i], 0], [bigTheta[i, i], 0]])
                )
            else:
                term = np.kron(term, np.array([[1, 0], [0, bigTheta[i, j]]]))
        Q += term
    return Q
