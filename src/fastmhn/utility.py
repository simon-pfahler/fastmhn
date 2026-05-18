from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_subdata(
    data: NDArray[np.int32], columns: list[int]
) -> NDArray[np.int32]:
    """
    Reduces a dataset to contain fewer events.

    Parameters
    ----------
    data : numpy.ndarray
        Nxd matrix containing the dataset
    columns : list of int
        Indices of columns (events) to keep in the dataset

    Returns
    -------
    numpy.ndarray
        Nxk matrix containing only the selected columns, where k = len(columns)
    """
    return data[:, columns]


def create_indep_model(
    data: NDArray[np.int32], weights: Optional[NDArray[np.float64]] = None
) -> NDArray[np.float64]:
    """
    Creates the independence model for a given dataset.

    The independence model has theta[i,j] = 0 for i != j, with diagonal
    entries set to log(f[i] / (1 - f[i])), where f[i] is the frequency
    of event i.

    Parameters
    ----------
    data : numpy.ndarray
        Nxd matrix containing the dataset (binary: 0 or 1)
    weights : numpy.ndarray, optional
        Array of length N, used to set the influence of individual samples
        on the model. Default is None, which uses a weight of 1 for all samples.

    Returns
    -------
    numpy.ndarray
        dxd theta matrix for the independence model
    """
    if weights is None:
        weights = np.ones(data.shape[0], dtype=np.float64)

    d = data.shape[1]
    theta = np.zeros((d, d), dtype=np.float64)
    f = np.sum(data * np.expand_dims(weights, -1), axis=0) / np.sum(weights)
    for i in range(d):
        theta[i, i] = np.log(f[i] / (1 - f[i]))
    return theta


def generate_theta(
    d: int,
    base_rate_loc: float = -1,
    base_rate_scale: float = 1,
    influence_loc: float = 0,
    influence_scale: float = 1,
    sparsity: float = 0.8,
) -> NDArray[np.float64]:
    """
    Generates artificial theta matrices with similar features to biological ones.

    The diagonal entries (base rates) are drawn from a normal distribution.
    Off-diagonal entries (influences) are drawn from a Laplace distribution
    and are sparsified (set to zero with probability `sparsity`).

    Parameters
    ----------
    d : int
        Number of events (dimension of theta matrix)
    base_rate_loc : float, optional
        Mean of the normal distribution for base rates (diagonal entries).
        Should be around -1 for few (<20) events and smaller for more events.
        Default is -1.
    base_rate_scale : float, optional
        Standard deviation of the normal distribution for base rates.
        Default is 1.
    influence_loc : float, optional
        Mean of the Laplace distribution for influences (off-diagonal entries).
        Default is 0.
    influence_scale : float, optional
        Scale parameter of the Laplace distribution for influences.
        Default is 1.
    sparsity : float, optional
        Fraction of off-diagonal entries to set to zero (0.0 to 1.0).
        Default is 0.8.

    Returns
    -------
    numpy.ndarray
        dxd theta matrix
    """
    theta = np.random.laplace(
        loc=influence_loc, scale=influence_scale, size=(d, d)
    )
    theta = (np.random.random((d, d)) > sparsity) * theta
    np.fill_diagonal(
        theta,
        np.random.normal(loc=base_rate_loc, scale=base_rate_scale, size=d),
    )
    return theta


def generate_data(thetaGT: NDArray[np.float64], size: int) -> NDArray[np.int32]:
    """
    Generates artificial data from a ground truth theta matrix.

    This method uses the Gillespie algorithm to sample from the Markov process
    parameterized by the given theta matrix. Each sample starts from the
    all-zero state and adds events until no more can be added.

    Parameters
    ----------
    thetaGT : numpy.ndarray
        dxd ground truth theta matrix
    size : int
        Number of samples to generate

    Returns
    -------
    numpy.ndarray
        Nxd matrix of binary data (0 or 1), where N = size
    """
    d = thetaGT.shape[0]

    data = np.zeros((size, d), dtype=np.int32)

    # Precompute exp(thetaGT) once for efficiency
    exp_theta = np.exp(thetaGT)
    diag_exp = exp_theta.diagonal()

    zero_column = True
    run_nr = 0
    while run_nr < 10 and zero_column == True:

        # generate the samples
        for i in range(size):
            sample = np.zeros(d, dtype=np.int32)

            # Gillespie algorithm: add events until the sample gets observed
            transitionRates = diag_exp.copy()
            while True:

                # get next event
                rateSum = np.sum(transitionRates) + 1
                rand = np.random.rand()
                sumRejected = 0
                newEvent = 0
                while sumRejected + transitionRates[newEvent] < rand * rateSum:
                    sumRejected += transitionRates[newEvent]
                    newEvent += 1
                    if newEvent == d:
                        break
                if newEvent == d:
                    break
                sample[newEvent] = 1

                # update transition rates - use precomputed exp_theta
                transitionRates[newEvent] = 0
                transitionRates *= exp_theta[:, newEvent]

            data[i] = sample

        # check if there is a zero column
        if all(np.sum(data, axis=1) != 0):
            zero_column = False
        run_nr += 1

    return data


def create_pD(data: NDArray[np.int32]) -> NDArray[np.float64]:
    """
    Creates the data distribution given a dataset.

    pD is a probability distribution over all 2^d possible states, where
    each state corresponds to a unique combination of events being present (1)
    or absent (0).

    Parameters
    ----------
    data : numpy.ndarray
        Nxd dataset of binary values (0 or 1)

    Returns
    -------
    numpy.ndarray
        Array of length 2^d containing the probability distribution.
        pD[i] is the probability of state i (interpreted as binary number).
    """

    d = data.shape[1]
    N = data.shape[0]
    # Handle empty dataset
    if N == 0:
        # Return uniform distribution over all 2^d states, or [1.0] for d=0
        pD = np.ones(2**d)
        pD /= 2**d
        return pD
    # Vectorized: convert each row to integer using bitwise operations
    powers = 1 << np.arange(d - 1, -1, -1)
    indices = np.dot(data, powers)
    pD = np.bincount(indices, minlength=2**d)
    return pD / N


def forward_substitution(
    lower_triangular_operator: Callable[[NDArray], NDArray],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Solves the linear equation Lx = b for a lower triangular matrix L,
    given as an operator.

    Parameters
    ----------
    lower_triangular_operator : callable
        Function that takes a vector x and returns Lx, where L is a
        lower triangular matrix
    rhs : numpy.ndarray
        Right-hand side vector b of the linear equation

    Returns
    -------
    numpy.ndarray
        Solution vector x
    """
    res = np.zeros_like(rhs)
    single = np.zeros_like(rhs)
    single[0] = 1
    res[0] = rhs[0] / lower_triangular_operator(single)[0]
    for i in range(1, res.shape[0]):
        res[i] = rhs[i] - lower_triangular_operator(res)[i]
        single = np.zeros_like(rhs)
        single[i] = 1
        res[i] /= lower_triangular_operator(single)[i]
    return res


def backward_substitution(
    upper_triangular_operator: Callable[[NDArray], NDArray],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Solves the linear equation Ux = b for an upper triangular matrix U,
    given as an operator.

    Parameters
    ----------
    upper_triangular_operator : callable
        Function that takes a vector x and returns Ux, where U is an
        upper triangular matrix
    rhs : numpy.ndarray
        Right-hand side vector b of the linear equation

    Returns
    -------
    numpy.ndarray
        Solution vector x
    """
    res = np.zeros_like(rhs)
    single = np.zeros_like(rhs)
    single[-1] = 1
    res[-1] = rhs[-1] / upper_triangular_operator(single)[-1]
    for i in range(2, len(rhs) + 1):
        res[-i] = rhs[-i] - upper_triangular_operator(res)[-i]
        single = np.zeros_like(rhs)
        single[-i] = 1
        res[-i] /= upper_triangular_operator(single)[-i]
    return res


def jacobi(
    op_diag: Callable[[NDArray], NDArray],
    op_offdiag: Callable[[NDArray], NDArray],
    rhs: NDArray[np.float64],
    iterations: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Approximates the solution of the linear equation Ax = b using the Jacobi method.

    The matrix A is decomposed as A = D + O, where D is the diagonal part
    and O is the off-diagonal part. The operators op_diag and op_offdiag
    return Dx and Ox respectively.

    Parameters
    ----------
    op_diag : callable
        Function that takes a vector x and returns diag(A)x
    op_offdiag : callable
        Function that takes a vector x and returns offdiag(A)x
    rhs : numpy.ndarray
        Right-hand side vector b of the linear equation
    iterations : int, optional
        Number of iterations to perform. Default is len(rhs).bit_length(),
        which gives the exact solution for triangular matrices.

    Returns
    -------
    numpy.ndarray
        Approximate solution vector x
    """

    if iterations is None:
        iterations = len(rhs).bit_length()

    res = np.zeros_like(rhs)

    dg = op_diag(np.ones_like(res))

    for _ in range(iterations):
        res = rhs - op_offdiag(res)
        res /= dg

    return res


def get_score_offset(
    data: NDArray[np.int32], weights: Optional[NDArray[np.float64]] = None
) -> float:
    """
    Calculates the offset between the log-likelihood score and the KL divergence.

    The KL divergence is obtained from the score and this offset via:
        D_KL = offset - score

    The offset is calculated via:
        offset = sum_x p_{D,x} * ln(p_{D,x})

    Parameters
    ----------
    data : numpy.ndarray
        Nxd dataset to calculate offset for
    weights : numpy.ndarray, optional
        Array of length N, used to set the influence of individual samples
        on the offset. Default is None, which uses a weight of 1 for all samples.

    Returns
    -------
    float
        The score offset value
    """

    if weights is None:
        weights = np.ones(data.shape[0], dtype=np.float64)

    N = np.sum(weights)
    # Use np.unique with return_inverse for vectorized computation
    unique_samples, inverse_indices = np.unique(
        data, axis=0, return_inverse=True
    )

    # Sum weights for each unique sample
    counts = np.zeros(len(unique_samples), dtype=np.float64)
    for i, idx in enumerate(inverse_indices):
        counts[idx] += weights[i]

    # Vectorized computation of offset
    p = counts / N
    offset = float(np.sum(np.where(p > 0, p * np.log(p), 0)))

    return offset


def cmhn_from_omhn(theta_omhn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Converts an oMHN (observation MHN) theta matrix to the equivalent cMHN (classical MHN).

    The relationship between oMHN and cMHN parameters is:
        theta_cmhn[i,j] = theta_omhn[i,j] - theta_omhn[d,j]  for i != j
        theta_cmhn[i,i] = theta_omhn[i,i]

    Parameters
    ----------
    theta_omhn : numpy.ndarray
        (d+1)xd theta matrix for the observation MHN

    Returns
    -------
    numpy.ndarray
        dxd theta matrix for the classical MHN
    """
    d = theta_omhn.shape[1]
    ctheta: NDArray[np.float64] = theta_omhn[:d] - theta_omhn[d]
    np.fill_diagonal(ctheta, np.diag(theta_omhn[:d]))
    return ctheta


def adamW(
    params_init: NDArray[np.float64],
    grad_and_score_func: Callable[
        [NDArray[np.float64]], Tuple[NDArray[np.float64], float]
    ],
    reg_grad_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    alpha: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    N_max: int = 1000,
    score_threshold: float = 1e-5,
    param_change_threshold: float = 1e-3,
    verbose: bool = False,
) -> NDArray[np.float64]:
    """
    Optimizes parameters to maximize a score function using AdamW algorithm.

    AdamW is a variant of Adam that properly decouples weight decay from
    the gradient update. See https://arxiv.org/pdf/1711.05101.

    Parameters
    ----------
    params_init : numpy.ndarray
        Initial values for the parameters
    grad_and_score_func : callable
        Function that takes parameters and returns (gradient, score) tuple
    reg_grad_func : callable
        Function that takes parameters and returns the gradient of the
        regularization term
    alpha : float, optional
        Step size (learning rate). Default is 1e-3.
    beta1 : float, optional
        Decay coefficient for first moment. Default is 0.9.
    beta2 : float, optional
        Decay coefficient for second moment. Default is 0.999.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-8.
    N_max : int, optional
        Maximum number of iterations. Default is 1000.
    score_threshold : float, optional
        Relative score difference for convergence. Default is 1e-5.
    param_change_threshold : float, optional
        Maximum parameter change for convergence. Default is 1e-3.
    verbose : bool, optional
        Whether to print intermediate information. Default is False.

    Returns
    -------
    numpy.ndarray
        Optimized parameters
    """

    # Initialization
    params = np.copy(params_init)
    m = np.zeros_like(params)
    v = np.zeros_like(params)

    s_prev = -np.inf
    t = 1
    while t <= N_max:
        g, s = grad_and_score_func(params)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)

        step = alpha * (mhat / (np.sqrt(vhat) + eps) + reg_grad_func(params))

        params += step

        if verbose:
            print(f"{t} - {s}")

        # convergence checks
        if t > 1:
            if np.abs(s_prev - s) / np.abs(s_prev) < score_threshold:
                if verbose:
                    print("Optimization stopped due to score")
                break
            if np.max(np.abs(step)) < param_change_threshold:
                if verbose:
                    print("Optimization stopped due to parameter change")
                break
        s_prev = s
        t += 1

    if t == N_max:
        if verbose:
            print("Optimization stopped due to maximum number of iterations")

    return params


def adam(
    params_init: NDArray[np.float64],
    grad_and_score_func: Callable[
        [NDArray[np.float64]], Tuple[NDArray[np.float64], float]
    ],
    reg_grad_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    alpha: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    N_max: int = 1000,
    score_threshold: float = 1e-5,
    param_change_threshold: float = 1e-3,
    verbose: bool = False,
) -> NDArray[np.float64]:
    """
    Optimizes parameters to maximize a score function using the Adam algorithm.

    Adam (Adaptive Moment Estimation) combines the benefits of AdaGrad and RMSProp.
    See https://arxiv.org/abs/1412.6980.

    Parameters
    ----------
    params_init : numpy.ndarray
        Initial values for the parameters
    grad_and_score_func : callable
        Function that takes parameters and returns (gradient, score) tuple
    reg_grad_func : callable
        Function that takes parameters and returns the gradient of the
        regularization term
    alpha : float, optional
        Step size (learning rate). Default is 1e-3.
    beta1 : float, optional
        Decay coefficient for first moment. Default is 0.9.
    beta2 : float, optional
        Decay coefficient for second moment. Default is 0.999.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-8.
    N_max : int, optional
        Maximum number of iterations. Default is 1000.
    score_threshold : float, optional
        Relative score difference for convergence. Default is 1e-5.
    param_change_threshold : float, optional
        Maximum parameter change for convergence. Default is 1e-3.
    verbose : bool, optional
        Whether to print intermediate information. Default is False.

    Returns
    -------
    numpy.ndarray
        Optimized parameters
    """

    # Initialization
    params = np.copy(params_init)
    m = np.zeros_like(params)
    v = np.zeros_like(params)

    s_prev = -np.inf
    t = 1
    while t <= N_max:
        g, s = grad_and_score_func(params)

        g += reg_grad_func(params)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)

        step = alpha * mhat / (np.sqrt(vhat) + eps)

        params += step

        if verbose:
            print(f"{t} - {s}")

        # convergence checks
        if t > 1:
            if np.abs(s_prev - s) / np.abs(s_prev) < score_threshold:
                if verbose:
                    print("Optimization stopped due to score")
                break
            if np.max(np.abs(step)) < param_change_threshold:
                if verbose:
                    print("Optimization stopped due to parameter change")
                break
        s_prev = s
        t += 1

    if t == N_max:
        if verbose:
            print("Optimization stopped due to maximum number of iterations")

    return params
