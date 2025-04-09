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
