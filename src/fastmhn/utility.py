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
