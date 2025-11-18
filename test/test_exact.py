import numpy as np
from mhn.training import likelihood_cmhn
from mhn.training.state_containers import StateContainer

import fastmhn

# >>> setup
d = 3
theta = np.random.normal(size=(d, d))
data = np.random.randint(0, high=2, size=(1000, d))
pD = fastmhn.utility.create_pD(data)

p0 = np.zeros(2**d)
p0[0] = 1

apply = lambda v: fastmhn.exact.apply_eye_minus_Q(theta, v, transpose=False)
apply_T = lambda v: fastmhn.exact.apply_eye_minus_Q(theta, v, transpose=True)
# <<< setup


def test_gradient_and_score_explicit():
    gradient, score = fastmhn.exact.gradient_and_score(theta, data)
    Q = fastmhn.explicit.create_full_Q(theta)
    pT = np.linalg.solve(np.eye(2**d) - Q, p0)
    fd_score = np.dot(pD, np.log(pT))
    assert np.abs(fd_score - score) < 1e-10, "Wrong score"

    explicit_grad = np.zeros((d, d))
    q = np.linalg.solve(np.eye(2**d) - Q.T, pD / pT)
    for i in range(d):
        r = (q * fastmhn.explicit.apply_Qdiff_ii(theta, pT, i)).reshape([2] * d)
        for j in range(d):
            if i == j:
                explicit_grad[i, j] = np.sum(r)
            else:
                explicit_grad[i, j] = np.sum(
                    r, axis=tuple(idx for idx in range(d) if idx != j)
                )[1]
            assert (
                np.abs(explicit_grad[i, j] - gradient[i, j]) < 1e-10
            ), f"Wrong gradient at index {i}, {j} when compared to explicit calculation"


def test_gradient_and_score_finite_differences():
    gradient, score = fastmhn.exact.gradient_and_score(theta, data)
    eps = 1e-5
    fd_grad = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            theta_left = theta.copy()
            theta_left[i, j] -= eps
            theta_right = theta.copy()
            theta_right[i, j] += eps
            Q_left = fastmhn.explicit.create_full_Q(theta_left)
            Q_right = fastmhn.explicit.create_full_Q(theta_right)
            pT_left = np.linalg.solve(np.eye(2**d) - Q_left, p0)
            pT_right = np.linalg.solve(np.eye(2**d) - Q_right, p0)
            score_left = np.dot(pD, np.log(pT_left))
            score_right = np.dot(pD, np.log(pT_right))
            fd_grad[i, j] = (score_right - score_left) / (2 * eps)
            assert (
                np.abs(fd_grad[i, j] - gradient[i, j]) < 1e-10
            ), f"Wrong gradient at index {i}, {j} when compared to finite differences"
