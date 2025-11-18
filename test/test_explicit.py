import numpy as np

import fastmhn

rng = np.random.default_rng(42)

# >>> setup
d = 3
N = 100
theta = rng.normal(size=(d, d))
data = rng.integers(2, size=(N, d), dtype=np.int32)
pD = fastmhn.utility.create_pD(data)
# <<< setup


def test_score():
    score = fastmhn.explicit.score(theta, pD)
    score_val = fastmhn.exact.gradient_and_score(theta, data)[1]
    assert (
        np.abs(score - score_val) < 1e-12
    ), f"Explicit score calculation is wrong!"


def test_gradient_and_score():
    gradient, score = fastmhn.explicit.gradient_and_score(theta, data)
    gradient_val, score_val = fastmhn.exact.gradient_and_score(theta, data)
    assert (
        np.abs(score - score_val) < 1e-12
    ), f"Explicit gradient_and_score calculation leads to incorrect score!"
    assert (
        np.linalg.norm(gradient - gradient_val) < 1e-12
    ), f"Explicit gradient_and_score calculation leads to incorrect gradient!"
