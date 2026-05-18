import numpy as np

import fastmhn

rng = np.random.default_rng(42)
np.random.seed(43)

# >>> setup
d = 3
# <<< setup


def test_learn_mhn_init():
    """Test MHN learning initialization returns diagonal theta for uniform data."""
    np.random.seed(43)
    N = 100000
    data = rng.integers(2, size=(N, d), dtype=np.int32)
    adam_params = {"N_max": 0, "verbose": False}

    theta = fastmhn.learn.learn_mhn(data, adam_params=adam_params)

    assert (
        np.linalg.norm(theta - np.diag(np.diag(theta))) < 1e-12
    ), f"Initialization of learn_mhn does not return diagonal theta!"

    for i in range(d):
        assert np.abs(1 / (1 + np.exp(theta[i, i])) - 0.5) < 1 / np.sqrt(
            N
        ), f"Base rates in initialization of learn_mhn do not match uniform data!"


def test_learn_mhn_regularization():
    """Test that regularization suppresses off-diagonal entries in MHN learning."""
    np.random.seed(43)
    thetaGT = fastmhn.utility.generate_theta(d)
    N = 100
    data = fastmhn.utility.generate_data(thetaGT, N)
    adam_params = {"N_max": 10, "verbose": False}

    theta_unreg = fastmhn.learn.learn_mhn(data, reg=0, adam_params=adam_params)
    theta_reg = fastmhn.learn.learn_mhn(data, reg=5e-2, adam_params=adam_params)

    assert (
        np.linalg.norm(theta_reg - np.diag(np.diag(theta_reg)))
        / np.linalg.norm(theta_unreg - np.diag(np.diag(theta_unreg)))
        < 0.1
    ), f"Regularization does not suppress off-diagonal entries enough!"


def test_learn_mhn_absent_events():
    """Test MHN learning handles absent events correctly."""
    np.random.seed(43)
    N = 100
    data = np.zeros((N, d), dtype=np.int32)
    data[: N // 2, 0] = 1
    data[0, 1] = 1
    adam_params = {"N_max": 0, "verbose": False}

    theta = fastmhn.learn.learn_mhn(data, adam_params=adam_params)

    assert (
        theta[2, 2] < theta[1, 1]
    ), f"Base rate of absent event is not smallest!"


def test_learn_omhn_init():
    """Test oMHN learning initialization."""
    np.random.seed(43)
    N = 100000
    data = rng.integers(2, size=(N, d), dtype=np.int32)
    adam_params = {"N_max": 0, "verbose": False}

    theta = fastmhn.learn.learn_omhn(data, adam_params=adam_params)

    assert np.all(
        theta[d] == 0
    ), f"Initialization of learn_omhn observation rates is wrong!"

    assert (
        np.linalg.norm(theta[:d] - np.diag(np.diag(theta[:d]))) < 1e-12
    ), f"Initialization of learn_omhn does not return diagonal theta!"

    for i in range(d):
        assert np.abs(1 / (1 + np.exp(theta[i, i])) - 0.5) < 1 / np.sqrt(
            N
        ), f"Base rates in initialization of learn_omhn do not match uniform data!"


def test_learn_omhn_regularization():
    """Test that regularization suppresses off-diagonal entries in oMHN learning."""
    np.random.seed(43)
    thetaGT = fastmhn.utility.generate_theta(d)
    N = 100
    data = fastmhn.utility.generate_data(thetaGT, N)
    adam_params = {"N_max": 10, "verbose": False}

    theta_unreg = fastmhn.learn.learn_omhn(data, reg=0, adam_params=adam_params)
    theta_reg = fastmhn.learn.learn_omhn(
        data, reg=5e-2, adam_params=adam_params
    )

    theta_unreg[:d] -= np.diag(np.diag(theta_unreg[:d]))
    theta_reg[:d] -= np.diag(np.diag(theta_reg[:d]))

    assert (
        np.linalg.norm(theta_reg) / np.linalg.norm(theta_unreg) < 0.1
    ), f"Regularization does not suppress off-diagonal entries enough!"


def test_learn_omhn_absent_events():
    """Test oMHN learning handles absent events correctly."""
    np.random.seed(43)
    N = 100
    data = np.zeros((N, d), dtype=np.int32)
    data[: N // 2, 0] = 1
    data[0, 1] = 1
    adam_params = {"N_max": 0, "verbose": False}

    theta = fastmhn.learn.learn_omhn(data, adam_params=adam_params)

    assert (
        theta[2, 2] < theta[1, 1]
    ), f"Base rate of absent event is not smallest!"


def test_learn_convergence():
    """Test that MHN learning converges by verifying score increases over iterations."""
    np.random.seed(43)
    N = 500
    thetaGT = fastmhn.utility.generate_theta(d)
    data = fastmhn.utility.generate_data(thetaGT, N)
    adam_params = {"N_max": 50, "verbose": False}

    # Track scores across iterations
    scores = []

    def grad_and_score_with_tracking(theta):
        g, s = fastmhn.exact.gradient_and_score(theta, data)
        scores.append(s)
        return g, s

    theta_init = fastmhn.utility.create_indep_model(data)
    regularization_mask = np.ones_like(theta_init, dtype=bool)
    regularization_mask ^= np.eye(d, dtype=bool)
    reg_grad_func = lambda theta: -1e-2 * regularization_mask * np.sign(theta)

    fastmhn.learn.adam(
        theta_init, grad_and_score_with_tracking, reg_grad_func, **adam_params
    )

    assert len(scores) > 1, "No iterations were recorded"
    # Score should increase or stay roughly the same, not decrease significantly
    for i in range(1, len(scores)):
        assert (
            scores[i] >= scores[0] - 1e-5
        ), f"Score decreased from {scores[0]} to {scores[i]} at iteration {i}"


# >>> Phase 3: Property-based tests <<<


def test_learn_initialization_is_independence_model():
    """Test that learn_mhn initialization returns independence model."""
    np.random.seed(43)
    N = 10000
    data = rng.integers(2, size=(N, d), dtype=np.int32)
    adam_params = {"N_max": 0, "verbose": False}

    theta = fastmhn.learn.learn_mhn(data, adam_params=adam_params)
    theta_ind = fastmhn.utility.create_indep_model(data)

    # With N_max=0, only initialization is returned
    assert np.linalg.norm(theta - theta_ind) < 1e-12, (
        "Initialization does not match independence model"
    )


def test_learn_theta_regularization_shrinks_values():
    """Test that regularization shrinks parameter magnitudes."""
    np.random.seed(43)
    N = 100
    data = rng.integers(2, size=(N, d), dtype=np.int32)
    adam_params = {"N_max": 20, "verbose": False}

    theta_no_reg = fastmhn.learn.learn_mhn(data, reg=0, adam_params=adam_params)
    theta_with_reg = fastmhn.learn.learn_mhn(data, reg=0.1, adam_params=adam_params)

    # L1 regularization should shrink parameters toward zero
    assert np.linalg.norm(theta_with_reg) < np.linalg.norm(theta_no_reg), (
        "Regularization should shrink parameter magnitudes"
    )
