import numpy as np

import fastmhn


def test_approx_gradient_and_score(rng, d, N):
    """Test approximate gradient and score against exact."""
    theta = rng.normal(size=(d, d))
    data = rng.integers(2, size=(N, d), dtype=np.int32)
    pD = fastmhn.utility.create_pD(data)

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


def test_approx_vs_exact(rng, d, N):
    """Test approximate gradient and score against exact implementation."""
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


# >>> Phase 3: Property-based tests <<<


def test_approx_score_is_negative(rng, d, N):
    """Test that approx score is always negative (log probabilities)."""
    test_theta = rng.normal(size=(d, d))
    test_data = rng.integers(2, size=(N, d), dtype=np.int32)

    for max_cluster_size in [d, d + 1]:
        _, score = fastmhn.approx.approx_gradient_and_score(
            test_theta, test_data, max_cluster_size=max_cluster_size
        )
        assert score < 0, f"Approx score should be negative, got {score}"


def test_approx_gradient_shape(rng, d, N):
    """Test that approx gradient has correct shape matching theta."""
    test_theta = rng.normal(size=(d, d))
    test_data = rng.integers(2, size=(N, d), dtype=np.int32)

    gradient, _ = fastmhn.approx.approx_gradient_and_score(
        test_theta, test_data, max_cluster_size=d
    )
    assert (
        gradient.shape == test_theta.shape
    ), f"Gradient shape {gradient.shape} doesn't match theta shape {test_theta.shape}"
