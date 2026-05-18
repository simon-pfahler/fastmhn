import numpy as np

from .utility import create_pD, jacobi

# Precompute bit masks for efficiency
_bit_masks_cache = {}


def _get_bit_masks(d):
    """Get or create bit masks for a given dimension d."""
    if d not in _bit_masks_cache:
        n_states = 2**d
        # For each bit position j (0 to d-1), create masks for bit (d-1-j)
        masks_0 = []
        masks_1 = []
        for j in range(d):
            bit_pos = d - 1 - j
            mask0 = (np.arange(n_states) & (1 << bit_pos)) == 0
            mask1 = (np.arange(n_states) & (1 << bit_pos)) != 0
            masks_0.append(mask0)
            masks_1.append(mask1)
        _bit_masks_cache[d] = (np.array(masks_0), np.array(masks_1))
    return _bit_masks_cache[d]


def calculate_pTheta(theta):
    """
    Calculates the time-marginalized probability distribution pTheta for a
    given theta matrix.

    `theta`: dxd theta matrix
    """
    d = theta.shape[0]

    p0 = np.zeros(2**d)
    p0[0] = 1

    op_diag = lambda x: apply_eye_minus_Q_diag(theta, x)
    op_offdiag = lambda x: apply_eye_minus_Q_offdiag(theta, x)

    pTheta = jacobi(op_diag, op_offdiag, p0)

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


def gradient_and_score(theta, data):
    """
    Calculates the gradient and score for a given theta matrix and data
    distribution.

    `theta`: dxd theta matrix
    `pD`: 2**d probability distribution of the data
    """
    d = theta.shape[0]

    pD = create_pD(data)

    pTheta = calculate_pTheta(theta)

    score = np.dot(pD, np.log(pTheta))

    op_diag = lambda x: apply_eye_minus_Q_diag(theta, x)
    op_offdiag = lambda x: apply_eye_minus_Q_offdiag(theta, x, transpose=True)
    q = jacobi(op_diag, op_offdiag, pD / pTheta)

    gradient = np.zeros((d, d))

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


def apply_eye_minus_Q(theta, x, transpose=False):
    """
    Calculates (I-Q) @ x for a given theta matrix and vector x

    `theta`: dxd theta matrix
    `x`: 2**d vector
    `transpose`: set to true if (I-Q)^T @ x should be calculated
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


def apply_eye_minus_Q_diag(theta, x, transpose=False):
    """
    Calculates diag(I-Q) @ x for a given theta matrix and vector x

    `theta`: dxd theta matrix
    `x`: 2**d vector
    `transpose`: does not do anything, only added so the interface is the same
        as for `apply_eye_minus_Q` and `apply_eye_minus_Q_offdiag`
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


def apply_eye_minus_Q_offdiag(theta, x, transpose=False):
    """
    Calculates offdiag(I-Q) @ x for a given theta matrix and vector x

    `theta`: dxd theta matrix
    `x`: 2**d vector
    `transpose`: set to true if offdiad(I-Q)^T @ x should be calculated
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


def apply_Qdiff_ii(theta, x, i):
    """
    Calculates dQ/d(theta_ii) @ x for a given theta matrix and vector x

    `theta`: dxd theta matrix
    `x`: 2**d vector
    `i`: index of theta matrix to take derivative with respect to
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
