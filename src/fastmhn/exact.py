import numpy as np
from mhn.training import likelihood_cmhn
from mhn.training.state_containers import StateContainer

from .utility import backward_substitution, forward_substitution


def calculate_pTheta(theta):
    """
    Calculates the time-marginalized probability distribution pTheta for a
    given theta matrix.

    `theta`: dxd theta matrix
    """
    d = theta.shape[0]

    op = lambda x: apply_eye_minus_Q(theta, x)
    p0 = np.zeros(2**d)
    p0[0] = 1
    pTheta = forward_substitution(op, p0)

    return pTheta


def score(theta, pD):
    """
    Calculates the score for a given theta matrix and data distribution.

    `theta`: dxd theta matrix
    `pD`: 2**d probability distribution of the data
    """
    d = theta.shape[0]

    pTheta = calculate_pTheta(theta)

    return np.dot(pD, np.log(pTheta))


def gradient_and_score(theta, pD):
    """
    Calculates the gradient and score for a given theta matrix and data
    distribution.

    `theta`: dxd theta matrix
    `pD`: 2**d probability distribution of the data
    """
    d = theta.shape[0]

    pTheta = calculate_pTheta(theta)

    score = np.dot(pD, np.log(pTheta))

    op_transpose = lambda x: apply_eye_minus_Q(theta, x, transpose=True)
    q = backward_substitution(op_transpose, pD / pTheta)

    gradient = np.zeros((d, d))
    for i in range(d):
        h = apply_Qdiff_ii(theta, pTheta, i)
        r = q * h
        for j in range(d):
            if i == j:
                gradient[i, i] = np.sum(r)
                continue
            mask = (np.arange(2**d) & (2**d >> (j + 1))) != 0
            gradient[i, j] = np.sum(r[mask])

    return gradient, score


def apply_eye_minus_Q(theta, x, transpose=False):
    """
    Calculates (I-Q) @ x for a given theta matrix and vector x

    `theta`: dxd theta matrix
    `x`: 2**d vector
    """
    d = theta.shape[0]
    bigTheta = np.exp(theta)
    b = x.copy()
    for i in range(d):
        v = x.copy()
        for j in range(d):
            mask0 = (np.arange(2**d) & (2**d >> (j + 1))) == 0
            mask1 = (np.arange(2**d) & (2**d >> (j + 1))) != 0
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


def apply_Qdiff_ii(theta, x, i):
    """
    Calculates dQ/d(theta_ii) @ x for a given theta matrix and vector x

    `theta`: dxd theta matrix
    `x`: 2**d vector
    `i`: index of theta matrix to take derivative with respect to
    """
    d = theta.shape[0]
    bigTheta = np.exp(theta)
    v = x.copy()
    for k in range(d):
        mask0 = (np.arange(2**d) & (2**d >> (k + 1))) == 0
        mask1 = (np.arange(2**d) & (2**d >> (k + 1))) != 0
        if k == i:
            v[mask0] *= -bigTheta[i, i]
            v[mask1] = -v[mask0]
        else:
            v[mask1] *= bigTheta[i, k]
    return v


def create_full_Q(theta):
    """
    Creates the full Q matrix for a given theta matrix

    `theta`: dxd theta matrix
    """
    d = theta.shape[0]
    bigTheta = np.exp(theta)
    Q = np.zeros((2**d, 2**d))
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
