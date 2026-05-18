import numpy as np

import fastmhn


def test_generate_data(N_large):
    """Test that generated data has correct distribution."""
    theta = np.diag([np.log(10), np.log(10), np.log(10)])
    data = fastmhn.utility.generate_data(theta, N_large)
    active_events = np.sum(data, axis=1)
    nr_samples = dict(zip(*np.unique(active_events, return_counts=True)))
    probability = 1 / 31
    assert (
        np.abs(nr_samples[0] - N_large * probability)
        < 20 * np.sqrt(N_large) * probability
    ), f"Wrong number of samples without any events generated ({nr_samples[0]})!"
    probability = 30 / 31 * 1 / 21
    assert (
        np.abs(nr_samples[1] - N_large * probability)
        < 20 * np.sqrt(N_large) * probability
    ), f"Wrong number of samples with one event generated ({nr_samples[1]})!"
    probability = 30 / 31 * 20 / 21 * 1 / 11
    assert (
        np.abs(nr_samples[2] - N_large * probability)
        < 20 * np.sqrt(N_large) * probability
    ), f"Wrong number of samples with two event generated ({nr_samples[2]})!"
    probability = 30 / 31 * 20 / 21 * 10 / 11
    assert (
        np.abs(nr_samples[3] - N_large * probability)
        < 20 * np.sqrt(N_large) * probability
    ), f"Wrong number of samples with three event generated ({nr_samples[3]})!"


def test_create_indep_model(rng, d, N):
    """Test independence model matches ground truth for uniform data."""
    theta = np.diag(rng.normal(size=d))
    data = fastmhn.utility.generate_data(theta, N)
    theta_ind = fastmhn.utility.create_indep_model(data)
    for i in range(d):
        assert np.abs(theta_ind[i, i] - theta[i, i]) < 10 / np.sqrt(N), (
            f"Independence model leads to wrong entry for event {i}: "
            f"absolute error is {np.abs(theta_ind[i,i]-theta[i,i])}!"
        )


def test_create_pD(rng, d, N):
    """Test pD creation correctness and normalization."""
    data = rng.integers(2, size=(N, d), dtype=np.int32)
    pD = fastmhn.utility.create_pD(data)

    # Test sum to 1
    assert (
        np.abs(np.sum(pD) - 1) < 1e-12
    ), f"pD does not sum to 1 (but rather {np.sum(pD)})!"

    # Test correctness by comparing to slow implementation
    pD_slow = np.zeros(2**d)
    for sample in data:
        index = int("".join(map(str, sample)), 2)
        pD_slow[index] += 1
    pD_slow /= N
    assert np.allclose(
        pD, pD_slow, atol=1e-12
    ), "create_pD values don't match slow implementation"


def test_forward_substitution(rng, d):
    """Test forward substitution against np.linalg.solve."""
    lower_triangular_matrix = np.tril(rng.normal(size=(d, d)))
    lower_triangular_operator = lambda x: lower_triangular_matrix @ x
    rhs = rng.normal(size=d)
    res = fastmhn.utility.forward_substitution(lower_triangular_operator, rhs)
    res_np = np.linalg.solve(lower_triangular_matrix, rhs)
    assert (
        np.linalg.norm(res - res_np) < 1e-12
    ), f"Forward substitution leads to wrong result!"


def test_backward_substitution(rng, d):
    """Test backward substitution against np.linalg.solve."""
    upper_triangular_matrix = np.triu(rng.normal(size=(d, d)))
    upper_triangular_operator = lambda x: upper_triangular_matrix @ x
    rhs = rng.normal(size=d)
    res = fastmhn.utility.backward_substitution(upper_triangular_operator, rhs)
    res_np = np.linalg.solve(upper_triangular_matrix, rhs)
    assert (
        np.linalg.norm(res - res_np) < 1e-12
    ), f"Backward substitution leads to wrong result!"


def test_jacobi(rng, d):
    """Test Jacobi solver on diagonal and triangular matrices."""
    diagonal_matrix = np.diag(rng.normal(size=d))
    diagonal_operator = lambda x: diagonal_matrix @ x
    rhs = rng.normal(size=d)
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
    rhs = rng.normal(size=d)
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


def test_get_score_offset(rng, d, N):
    """Test score offset calculation for trivial and non-trivial cases."""
    # Trivial case: all samples identical
    data = np.zeros((N, d), dtype=np.int32)
    data[:, 1] = 1
    offset = fastmhn.utility.get_score_offset(data)
    assert (
        np.abs(offset) < 1e-12
    ), f"Score offset for trivial dataset is wrong ({offset})!"

    # Non-trivial case: known distribution
    # Create data with 2 types of samples
    data2 = np.zeros((100, 2), dtype=np.int32)
    data2[:50, 0] = 1  # 50 samples with only event 0
    data2[50:, 1] = 1  # 50 samples with only event 1
    offset2 = fastmhn.utility.get_score_offset(data2)
    expected = 0.5 * np.log(0.5) + 0.5 * np.log(0.5)
    assert (
        np.abs(offset2 - expected) < 1e-10
    ), f"Score offset for known distribution is wrong: {offset2} != {expected}"

    # Test with weights
    weights = np.ones(100)
    weights[:50] = 2.0  # First 50 samples have weight 2
    offset3 = fastmhn.utility.get_score_offset(data2, weights=weights)
    # Total weight = 100 + 50 = 150
    # p0 = 100/150, p1 = 50/150
    p0 = 100 / 150
    p1 = 50 / 150
    expected_weighted = p0 * np.log(p0) + p1 * np.log(p1)
    assert (
        np.abs(offset3 - expected_weighted) < 1e-10
    ), f"Score offset with weights is wrong: {offset3} != {expected_weighted}"


def test_cmhn_from_omhn(rng, d):
    """Test cMHN conversion from oMHN."""
    omhn = rng.normal(size=(d + 1, d))
    cmhn = fastmhn.utility.cmhn_from_omhn(omhn)

    # Test diagonals match
    assert np.all(
        np.diag(omhn[:d]) == np.diag(cmhn)
    ), f"Diagonal entries of oMHN and equivalent cMHN do not match!"

    for i in range(d):
        for j in range(d):
            if i == j:
                # Diagonal: cmhn[i,i] = omhn[i,i]
                assert (
                    cmhn[i, j] == omhn[i, j]
                ), f"Diagonal entry cmhn[{i},{j}] != omhn[{i},{j}]"
            else:
                # Off-diagonal: cmhn[i,j] = omhn[i,j] - omhn[d,j]
                expected = omhn[i, j] - omhn[d, j]
                assert (
                    cmhn[i, j] == expected
                ), f"Off-diagonal entry cmhn[{i},{j}] = {cmhn[i,j]}, expected {expected}"


def test_adamW():
    """Test AdamW optimizer on simple quadratic function."""

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


def test_create_pD_empty_data():
    """Test pD creation with empty dataset (d=0)."""
    data = np.zeros((0, 0), dtype=np.int32)
    pD = fastmhn.utility.create_pD(data)
    assert (
        pD.shape[0] == 1
    ), f"pD should have 1 element for d=0, got {pD.shape[0]}"
    # Empty dataset should return uniform distribution
    assert (
        np.abs(pD[0] - 1.0) < 1e-12
    ), f"Empty dataset should have pD[0]=1, got {pD[0]}"


def test_create_pD_single_event():
    """Test pD creation with single event (d=1)."""
    data = np.array([[0], [1], [0], [1], [0]], dtype=np.int32)
    pD = fastmhn.utility.create_pD(data)
    assert pD.shape[0] == 2, f"pD should have 2 elements for d=1"
    assert np.abs(pD[0] - 0.6) < 1e-12, f"pD[0] should be 0.6, got {pD[0]}"
    assert np.abs(pD[1] - 0.4) < 1e-12, f"pD[1] should be 0.4, got {pD[1]}"


def test_create_pD_all_zeros(rng, d):
    """Test pD creation with all-zero data."""
    data = np.zeros((10, d), dtype=np.int32)
    pD = fastmhn.utility.create_pD(data)
    assert np.abs(pD[0] - 1) < 1e-12, f"pD[0] should be 1 for all-zero data"
    assert (
        np.sum(pD[1:]) < 1e-12
    ), f"Other pD entries should be 0 for all-zero data"


def test_create_pD_all_ones(rng, d):
    """Test pD creation with all-ones data."""
    data = np.ones((10, d), dtype=np.int32)
    pD = fastmhn.utility.create_pD(data)
    all_ones_index = 2**d - 1
    assert (
        np.abs(pD[all_ones_index] - 1) < 1e-12
    ), f"pD[{all_ones_index}] should be 1 for all-ones data"
    assert (
        np.sum(pD[:all_ones_index]) < 1e-12
    ), f"Other pD entries should be 0 for all-ones data"


def test_get_score_offset_single_sample():
    """Test score offset with single sample."""
    data = np.array([[1, 0, 1]], dtype=np.int32)
    offset = fastmhn.utility.get_score_offset(data)
    # Single sample: p=1 for that sample, offset = 1 * log(1) = 0
    assert (
        np.abs(offset) < 1e-12
    ), f"Offset for single sample should be 0, got {offset}"


def test_get_score_offset_uniform_weights():
    """Test score offset with uniform weights."""
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    weights = np.ones(4)
    offset = fastmhn.utility.get_score_offset(data, weights=weights)
    # Each sample has p=0.25, offset = 4 * 0.25 * log(0.25) = log(0.25)
    expected = np.log(0.25)
    assert (
        np.abs(offset - expected) < 1e-12
    ), f"Offset {offset} != expected {expected}"


def test_cmhn_from_omhn_single_event():
    """Test cMHN from oMHN with d=1."""
    omhn = np.array([[1.0], [0.5]])  # d=1, so omhn is (2, 1)
    cmhn = fastmhn.utility.cmhn_from_omhn(omhn)
    assert cmhn.shape == (1, 1), f"Expected (1,1), got {cmhn.shape}"
    assert cmhn[0, 0] == omhn[0, 0], "Diagonal should match"


def test_generate_theta_d1():
    """Test theta generation with d=1."""
    theta = fastmhn.utility.generate_theta(1)
    assert theta.shape == (1, 1), f"Expected (1,1), got {theta.shape}"


def test_create_indep_model_uniform_data(d):
    """Test independence model with uniform data."""
    data = np.zeros((150, d), dtype=np.int32)
    # Set each event to be present in exactly 50 samples
    data[:50, 0] = 1
    data[50:100, 1] = 1
    data[100:, 2] = 1
    theta = fastmhn.utility.create_indep_model(data)
    # For uniform data on each event (f[i] = 1/3), theta[i,i] = log(1/2)
    for i in range(d):
        assert (
            np.abs(theta[i, i] - np.log(0.5)) < 1e-12
        ), f"Independence model is wrong for uniform data"
