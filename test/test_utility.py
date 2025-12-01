import numpy as np

import fastmhn

rng = np.random.default_rng(42)
np.random.seed(43)

# >>> setup
d = 3
N = 10000
# <<< setup


def test_generate_data():
    theta = np.diag([np.log(10), np.log(10), np.log(10)])
    data = fastmhn.utility.generate_data(theta, N)
    active_events = np.sum(data, axis=1)
    nr_samples = dict(zip(*np.unique(active_events, return_counts=True)))
    probability = 1 / 31
    assert (
        np.abs(nr_samples[0] - N * probability) < 20 * np.sqrt(N) * probability
    ), f"Wrong number of samples without any events generated ({nr_samples[0]})!"
    probability = 30 / 31 * 1 / 21
    assert (
        np.abs(nr_samples[1] - N * probability) < 20 * np.sqrt(N) * probability
    ), f"Wrong number of samples with one event generated ({nr_samples[1]})!"
    probability = 30 / 31 * 20 / 21 * 1 / 11
    assert (
        np.abs(nr_samples[2] - N * probability) < 20 * np.sqrt(N) * probability
    ), f"Wrong number of samples with two event generated ({nr_samples[2]})!"
    probability = 30 / 31 * 20 / 21 * 10 / 11
    assert (
        np.abs(nr_samples[3] - N * probability) < 20 * np.sqrt(N) * probability
    ), f"Wrong number of samples with three event generated ({nr_samples[3]})!"


def test_create_indep_model():
    theta = np.diag(rng.normal(size=(d)))
    data = fastmhn.utility.generate_data(theta, N)
    theta_ind = fastmhn.utility.create_indep_model(data)
    for i in range(d):
        assert np.abs(theta_ind[i, i] - theta[i, i]) < 10 / np.sqrt(
            N
        ), f"Independence model leads to wrong entry for event {i}: absolute error is {np.abs(theta_ind[i,i]-theta[i,i])}!"


def test_create_pD():
    data = rng.integers(2, size=(N, d), dtype=np.int32)
    pD = fastmhn.utility.create_pD(data)
    assert (
        np.abs(np.sum(pD) - 1) < 1e-12
    ), f"pD does not sum to 1 (but rather {np.sum(pD)})!"


def test_forward_substitution():
    lower_triangular_matrix = np.tril(rng.normal(size=(d, d)))
    lower_triangular_operator = lambda x: lower_triangular_matrix @ x
    rhs = rng.normal(size=(d))
    res = fastmhn.utility.forward_substitution(lower_triangular_operator, rhs)
    res_np = np.linalg.solve(lower_triangular_matrix, rhs)
    assert (
        np.linalg.norm(res - res_np) < 1e-12
    ), f"Forward substitution leads to wrong result!"


def test_backward_substitution():
    upper_triangular_matrix = np.triu(rng.normal(size=(d, d)))
    upper_triangular_operator = lambda x: upper_triangular_matrix @ x
    rhs = rng.normal(size=(d))
    res = fastmhn.utility.backward_substitution(upper_triangular_operator, rhs)
    res_np = np.linalg.solve(upper_triangular_matrix, rhs)
    assert (
        np.linalg.norm(res - res_np) < 1e-12
    ), f"Backward substitution leads to wrong result!"


def test_jacobi():
    diagonal_matrix = np.diag(rng.normal(size=(d)))
    diagonal_operator = lambda x: diagonal_matrix @ x
    rhs = rng.normal(size=(d))
    res = fastmhn.utility.jacobi(
        diagonal_operator, lambda x: 0, rhs, iterations=1
    )
    res_np = np.linalg.solve(diagonal_matrix, rhs)
    assert (
        np.linalg.norm(res - res_np) < 1e-12
    ), f"Jacobi leads to wrong result for diagonal matrix!"

    lower_triangular_matrix = np.tril(rng.normal(size=(d, d)))
    diagonal_matrix = np.diag(np.diag(lower_triangular_matrix))
    diagonal_operator = lambda x: diagonal_matrix @ x
    strictly_lower_triangular_matrix = lower_triangular_matrix - diagonal_matrix
    strictly_lower_triangular_operator = (
        lambda x: strictly_lower_triangular_matrix @ x
    )
    rhs = rng.normal(size=(d))
    res = fastmhn.utility.jacobi(
        diagonal_operator,
        strictly_lower_triangular_operator,
        rhs,
        iterations=1 << d,
    )
    res_np = np.linalg.solve(lower_triangular_matrix, rhs)
    assert (
        np.linalg.norm(res - res_np) < 1e-12
    ), f"Jacobi leads to wrong result for triangular matrix!"


def test_get_score_offset():
    data = np.zeros((N, d), dtype=np.int32)
    data[:, 1] = 1
    offset = fastmhn.utility.get_score_offset(data)
    assert (
        np.abs(offset) < 1e-12
    ), f"Score offset for trivial dataset is wrong ({offset})!"


def test_cmhn_from_omhn():
    omhn = rng.normal(size=(d + 1, d))
    cmhn = fastmhn.utility.cmhn_from_omhn(omhn)
    assert np.all(
        np.diag(omhn[:d]) == np.diag(cmhn)
    ), f"Diagonal entries of oMHN and equivalent cMHN do not match!"


def test_adamW():
    def grad_and_score_func(params):
        s = -((params[0] - 0.5) ** 2) - (2 * params[1] - 3) ** 2
        g = np.array([-2 * (params[0] - 0.5), -4 * (2 * params[1] - 3)])
        return g, s

    def reg_grad_func(params):
        return 0

    params_init = np.ones(2)
    params_opt = fastmhn.utility.adamW(
        params_init,
        grad_and_score_func=grad_and_score_func,
        reg_grad_func=reg_grad_func,
        N_max=10000,
        param_change_threshold=1e-12,
        score_threshold=1e-12,
    )
    assert (
        np.linalg.norm(params_opt - np.array([0.5, 1.5])) < 1e-8
    ), f"Optimum found by AdamW is wrong!"
