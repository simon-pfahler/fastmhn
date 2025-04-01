import numpy as np


def write_matrix(filename, m, s=None, s_true=None):
    """
    Writes a numpy matrix to disk, along with some additional information

    `filename`: Path of the file to write the matrix to
    `m`: numpy matrix to write to disk
    `s`: score to print along with the matrix
    `s_true`: true score to print along with the matrix
    """
    with open(filename, "w") as f:
        for i in range(m.shape[0]):
            f.write(" ".join(map(str, m[i])) + "\n")
        f.write(f"Score {s}\n")
        f.write(f"True score {s_true}\n")


def get_data(filename):
    """
    Reads a dataset from disk

    `filename`: Path of the file to read data from
    """
    data = None
    if filename[-3:] == "csv":
        data = np.genfromtxt(
            filename, delimiter=",", skip_header=1, dtype=np.int32
        )
    elif filename[-3:] == "dat":
        data = np.genfromtxt(
            filename, delimiter=" ", skip_header=1, dtype=np.int32
        )
    return data


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
