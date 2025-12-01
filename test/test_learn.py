import numpy as np

import fastmhn

rng = np.random.default_rng(42)
np.random.seed(43)

# >>> setup
d = 3
# <<< setup


def test_learn_mhn_init():
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
    N = 100
    data = np.zeros((N, d), dtype=np.int32)
    data[: N // 2, 0] = 1
    data[0, 1] = 1
    adam_params = {"N_max": 0, "verbose": False}

    theta = fastmhn.learn.learn_omhn(data, adam_params=adam_params)

    assert (
        theta[2, 2] < theta[1, 1]
    ), f"Base rate of absent event is not smallest!"
