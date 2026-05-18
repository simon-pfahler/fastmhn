import numpy as np
import pytest

import fastmhn


@pytest.fixture
def rng():
    """Provide a seeded numpy random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def d():
    """Default dimension for tests."""
    return 3


@pytest.fixture
def N():
    """Default sample size for tests."""
    return 100


@pytest.fixture
def N_large():
    """Large sample size for statistical tests."""
    return 10000


@pytest.fixture
def theta(rng, d):
    """Random theta matrix of shape (d, d)."""
    return rng.normal(size=(d, d))


@pytest.fixture
def data(rng, N, d):
    """Random binary data of shape (N, d)."""
    return rng.integers(2, size=(N, d), dtype=np.int32)


@pytest.fixture
def pD(data):
    """Probability distribution from data."""
    return fastmhn.utility.create_pD(data)


@pytest.fixture
def theta_diagonal(rng, d):
    """Diagonal theta matrix."""
    return np.diag(rng.normal(size=d))


@pytest.fixture
def uniform_data(rng, N, d):
    """Uniform random binary data."""
    return rng.integers(2, size=(N, d), dtype=np.int32)
