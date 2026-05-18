import numpy as np

import fastmhn

rng = np.random.default_rng(42)
np.random.seed(43)

# >>> setup
d = 3
N = 100
theta = rng.normal(size=(d, d))
data = rng.integers(2, size=(N, d), dtype=np.int32)
pD = fastmhn.utility.create_pD(data)
# <<< setup


def test_approx_gradient_and_score():
    """Test approximate gradient and score against exact."""
    np.random.seed(43)
    gradient, score = fastmhn.approx.approx_gradient_and_score(
        theta, data, max_cluster_size=3
    )
    gradient_val, score_val = fastmhn.exact.gradient_and_score(theta, data)
    assert (
        np.abs(score - score_val) < 1e-12
    ), f"Explicit gradient_and_score calculation leads to incorrect score!"
    assert (
        np.linalg.norm(gradient - gradient_val) < 1e-12
    ), f"Explicit gradient_and_score calculation leads to incorrect gradient!"


def test_approx_vs_exact():
    """Test approximate gradient and score against exact implementation."""
    np.random.seed(43)
    test_theta = rng.normal(size=(d, d))
    test_data = rng.integers(2, size=(N, d), dtype=np.int32)

    # Test with cluster sizes == d
    approx_gradient, approx_score = fastmhn.approx.approx_gradient_and_score(
        test_theta, test_data, max_cluster_size=d
    )
    exact_gradient, exact_score = fastmhn.exact.gradient_and_score(
        test_theta, test_data
    )
    assert (
        np.abs(approx_score - exact_score) < 1e-12
    ), f"Approx score differs from exact for max_cluster_size=d"
    assert (
        np.linalg.norm(approx_gradient - exact_gradient) < 1e-12
    ), f"Approx gradient differs from exact for max_cluster_size=d"
