import numpy as np
from mhn.training import likelihood_cmhn
from mhn.training.state_containers import StateContainer

import fastmhn


def test_gradient_and_score_explicit(rng, d):
    """Test gradient and score against explicit Q matrix calculation."""
    theta = rng.normal(size=(d, d))
    data = rng.integers(0, high=2, size=(1000, d))
    pD = fastmhn.utility.create_pD(data)

    p0 = np.zeros(2**d)
    p0[0] = 1

    apply_func = lambda v: fastmhn.exact.apply_eye_minus_Q(
        theta, v, transpose=False
    )
    apply_T_func = lambda v: fastmhn.exact.apply_eye_minus_Q(
        theta, v, transpose=True
    )

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


def test_gradient_and_score_finite_differences(rng, d):
    """Test gradient and score against finite differences."""
    theta = rng.normal(size=(d, d))
    data = rng.integers(0, high=2, size=(1000, d))
    pD = fastmhn.utility.create_pD(data)

    p0 = np.zeros(2**d)
    p0[0] = 1

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


def test_gradient_and_score_small_d(rng):
    """Test gradient_and_score correctness for small dimensions d=2 to 5."""
    for test_d in range(2, 6):
        test_theta = rng.normal(size=(test_d, test_d))
        test_data = rng.integers(0, high=2, size=(100, test_d))
        test_pD = fastmhn.utility.create_pD(test_data)

        gradient, score = fastmhn.exact.gradient_and_score(
            test_theta, test_data
        )

        # Verify against explicit Q matrix calculation
        Q = fastmhn.explicit.create_full_Q(test_theta)
        p0 = np.zeros(2**test_d)
        p0[0] = 1
        pT = np.linalg.solve(np.eye(2**test_d) - Q, p0)
        explicit_score = np.dot(test_pD, np.log(pT))
        assert (
            np.abs(explicit_score - score) < 1e-10
        ), f"Wrong score for d={test_d}"

        explicit_grad = np.zeros((test_d, test_d))
        q = np.linalg.solve(np.eye(2**test_d) - Q.T, test_pD / pT)
        for i in range(test_d):
            r = (
                q * fastmhn.explicit.apply_Qdiff_ii(test_theta, pT, i)
            ).reshape([2] * test_d)
            for j in range(test_d):
                if i == j:
                    explicit_grad[i, j] = np.sum(r)
                else:
                    explicit_grad[i, j] = np.sum(
                        r, axis=tuple(idx for idx in range(test_d) if idx != j)
                    )[1]
                assert (
                    np.abs(explicit_grad[i, j] - gradient[i, j]) < 1e-10
                ), f"Wrong gradient at index {i}, {j} for d={test_d}"


# >>> Phase 3: Property-based tests <<<


def test_score_is_negative(rng):
    """Test that score is always negative (log probabilities)."""
    for test_d in range(2, 5):
        test_theta = rng.normal(size=(test_d, test_d))
        test_data = rng.integers(0, high=2, size=(50, test_d))

        _, score = fastmhn.exact.gradient_and_score(test_theta, test_data)
        assert (
            score < 0
        ), f"Score should be negative, got {score} for d={test_d}"


def test_pTheta_is_probability_distribution(rng):
    """Test that pTheta is a valid probability distribution (sums to 1, non-negative)."""
    for test_d in range(2, 5):
        test_theta = rng.normal(size=(test_d, test_d))

        pTheta = fastmhn.explicit.calculate_pTheta(test_theta)
        assert (
            np.abs(np.sum(pTheta) - 1) < 1e-10
        ), f"pTheta doesn't sum to 1 for d={test_d}"
        assert np.all(
            pTheta >= -1e-12
        ), f"pTheta has negative values for d={test_d}"  # Allow tiny negative due to numerical errors


def test_gradient_zero_for_uniform_theta(rng):
    """Test that gradient is zero when theta is diagonal with uniform base rates."""
    for test_d in range(2, 4):
        # Create uniform theta (all diagonal entries equal)
        uniform_val = -1.0
        theta = np.diag([uniform_val] * test_d)

        # Generate data from this model
        data = fastmhn.utility.generate_data(theta, 100)

        gradient, _ = fastmhn.exact.gradient_and_score(theta, data)

        # Off-diagonal gradients should be near zero (small due to finite samples)
        off_diag = gradient - np.diag(np.diag(gradient))
        assert (
            np.linalg.norm(off_diag) < 0.5
        ), f"Off-diagonal gradient too large for uniform theta, d={test_d}"
