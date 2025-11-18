import numpy as np


def get_subdata(data, columns):
    """
    Reduces a dataset to contain less events

    `data`: Nxd matrix containing the dataset
    `columns`: columns (events) to keep in the dataset
    """
    return data[:, columns]


def create_indep_model(data):
    """
    Creates the independence model for a given dataset

    `data`: Nxd matrix containing the dataset
    """
    d = data.shape[1]
    theta = np.zeros((d, d))
    f = np.mean(data, axis=0)
    for i in range(d):
        theta[i, i] = np.log(f[i] / (1 - f[i]))
    return theta


def generate_theta(
    d,
    base_rate_loc=-1,
    base_rate_scale=1,
    influence_loc=0,
    influence_scale=1,
    sparsity=0.8,
):
    """
    Generates artificial theta matrices that have similar features to
    biological ones

    `d`: number of events
    `base_rate_loc`: average value of the base rates, should be around -1 for
        few (<20) events and be smaller for more events, default is -1
    `base_rate_scale`: scale parameter of the base rates, default is 1
    `influence_loc`: average value of the influences, default is 0
    `influence_scale`: scale parameter of the influences, default is 1
    `sparsity`: percentage of influences which are set to zero, should be
        larger the larger theta is, default is 0.8
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


def generate_data(thetaGT, size):
    """
    Generates artificial data from a ground truth theta matrix

    This method uses the Gillespie algorithm to sample from the Markov process
    parameterized by the given theta matrix

    `thetaGT`: ground truth theta matrix
    `size`: number of samples to generate
    """
    d = thetaGT.shape[0]

    data = np.zeros((size, d), dtype=np.int32)

    zero_column = True
    run_nr = 0
    while run_nr < 10 and zero_column == True:

        # generate the samples
        for i in range(size):
            sample = np.zeros(d, dtype=np.int32)

            # Gillespie algorithm: add events until the sample gets observed
            transitionRates = np.exp(np.diag(thetaGT))
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

                # update transition rates
                transitionRates[newEvent] = 0
                transitionRates *= np.exp(thetaGT[:, newEvent])

            data[i] = sample

        # check if there is a zero column
        if all(np.sum(data, axis=1) != 0):
            zero_column = False
        run_nr += 1

    return data


def create_pD(data):
    """
    Creates the data distribution given a dataset

    `data`: Nxd dataset to calculate pD for
    """

    d = data.shape[1]
    pD = np.zeros(2**d)
    for sample in data:
        index = int("".join(map(str, sample)), 2)
        pD[index] += 1
    return pD / data.shape[0]


def forward_substitution(lower_triangular_operator, rhs):
    """
    Calculates the solution of the linear equation Lx=b for a lower triangular
    matrix L, given as an operator.

    `lower_triangular_operator`: function that takes a vector `x` and returns
        `Lx`
    `rhs`: right-hand side of the linear equation
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


def jacobi(op_diag, op_offdiag, rhs, iterations=None):
    """
    Approximates the solution of the linear equation Ax=b for a matrix A, given
    as a sum of a diagonal and an off-diagonal operator.

    `op_diag`: function that takes a vector `x` and returns `diag(A)x`
    `op_offdiag`: function that takes a vector `x` and returns `offdiag(A)x`
    `rhs`: right-hand side of the linear equation
    `iterations`: number of iterations to perform, defaults to `len(rhs)`,
        which gives the exact solution the case of a triangular matrix
    """

    if iterations == None:
        iterations = len(rhs).bit_length()

    res = np.ones_like(rhs) / len(rhs)

    dg = op_diag(np.ones_like(res))

    for i in range(iterations):
        res = rhs + op_offdiag(res)
        res /= dg

    return res


def get_score_offset(data, weights=None):
    """
    Calculates the offset between the log-likelihood score and the KL divergence.
    The KL divergence is obtained from the score and this offset via
        D_{KL} = offset - score.

    The offset is calculated via
        offset = sum_{x}p_{D,x}\ln(p_{D,x})

    `data`: Nxd dataset to calculate offset for
    `weights`: array of length N, used set the influence of individual samples
        on the score and gradient, default is `None`, which uses a weight of 1
        for all samples
    """

    if weights is None:
        weights = np.ones(data.shape[0])

    N = np.sum(weights)
    unique_samples = np.unique(data, axis=0)

    offset = 0
    for sample in unique_samples:
        count = np.sum(weights[np.all(data == sample, axis=1)])
        offset += count / N * np.log(count / N)

    return offset


def backward_substitution(upper_triangular_operator, rhs):
    """
    Calculates the solution of the linear equation Ux=b for an upper triangular
    matrix U, given as an operator.

    `upper_triangular_operator`: function that takes a vector `x` and returns
        `Ux`
    `rhs`: right-hand side of the linear equation
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


def cmhn_from_omhn(theta_omhn):
    """
    Create the equivalent cMHN from an oMHN.

    `theta_omhn`: observation MHN theta matrix
    """
    d = theta_omhn.shape[1]
    ctheta = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                ctheta[i, j] = theta_omhn[i, j]
            else:
                ctheta[i, j] = theta_omhn[i, j] - theta_omhn[d, j]

    return ctheta


def adamW(
    params_init,
    grad_and_score_func,
    reg_grad_func,
    alpha=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    N_max=1000,
    score_threshold=1e-5,
    param_change_threshold=1e-3,
    verbose=False,
):
    """
    Optimize the parameters to maximize a score function using AdamW.
    See [https://arxiv.org/pdf/1711.05101](https://arxiv.org/pdf/1711.05101).

    `params_init`: inital values for the parameters
    `grad_and_score_func`: function to calculate the gradient and score, given
        parameters
    `reg_grad_func`: function to calculate the gradient of the regularization,
        given parameters
    `alpha`: step size (default: `1e-3`)
    `beta1`: decay coefficient for first moment (default: `0.9`)
    `beta2`: decay coefficient for second moment (default: `0.999`)
    `eps`: epsilon value to avoid division by zero (default: `1e-8`)
    `N_max`: maximum number of iterations (default: `1000`)
    `score_threshold`: relative score difference needed for convergence
        (default: `1e-5`)
    `param_change_threshold`: convergence is reached when all parameter change
        less than this threshold (default: `1e-3`)
    `verbose`: Print some intermediate information (default: `False`)
    """

    # Initialization
    params = np.copy(params_init)
    m = np.zeros_like(params)
    v = np.zeros_like(params)

    s_prev = -np.inf
    for t in range(1, N_max + 1):
        g, s = grad_and_score_func(params)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)

        step = alpha * (mhat / (np.sqrt(vhat) + eps) - reg_grad_func(params))

        params += step

        if verbose:
            print(f"{t} - {s}")

        # convergence checks
        if np.abs(s_prev - s) / np.abs(s_prev) < score_threshold:
            if verbose:
                print(f"Converged due to score")
            break
        if np.max(np.abs(step)) < param_change_threshold:
            if verbose:
                print(f"Converged due to parameter change")
            break
        s_prev = s

    return params
