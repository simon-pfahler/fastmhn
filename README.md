![Test](https://img.shields.io/github/actions/workflow/status/simon-pfahler/fastmhn/test.yml.svg?branch=main&label=test)

# FastMHN - Fast inference of MHNs

**FastMHN** is a Python package for approximate learning of Mutational Hierarchical Networks (MHNs) and observation MHNs (oMHNs). It enables fast inference through suitable rank-1 approximations of the time-marginalized probability distributions, making it practical to work with larger datasets where exact methods would be computationally prohibitive.

## Overview

Mutational Hierarchical Networks (MHNs) are probabilistic graphical models used to model the accumulation of mutations in cancer and other evolutionary processes. They capture dependencies between binary events (e.g., mutations, copy-number alterations) through a graph structure.

This package provides:

- **Approximate learning** of MHN and oMHN models using clustering-based approximations
- **Exact learning** methods for smaller datasets (mostly for testing)
- **Cross-validation** support for hyperparameter tuning (e.g., regularization strength)

The approximation methods allow inference on datasets with higher mutational burdens where exact computation of the full state space would be infeasible.

## Installation

The package can be installed directly from PyPI:

```bash
pip install fastmhn
```

Or clone the repository and install manually:

```bash
git clone https://phygit.ur.de/physics/mhn/fastmhn.git
cd fastmhn
pip install -e .
```

### Dependencies

- Python >= 3.11
- NumPy
- joblib
- mhn

## Example Usage

### Learning an MHN model

```python
import numpy as np
import fastmhn

# Generate synthetic data: N samples, d events
# Each row is a binary vector indicating which events occurred
d = 5
N = 100
data = np.random.randint(2, size=(N, d), dtype=np.int32)

# Learn MHN model with approximate gradient computation
theta = fastmhn.learn.learn_mhn(
    data,
    reg=1e-2,  # L1 regularization strength
    gradient_and_score_params={"max_cluster_size": 10},
    adam_params={
        "alpha": 0.1,
        "beta1": 0.7,
        "beta2": 0.9,
        "eps": 1e-8,
        "verbose": True,
    },
)

# theta is a d x d matrix representing the learned MHN
print(f"Learned theta matrix:\n{theta}")
```

Replace `data` with your own dataset, this is just a placeholder in the code snippet.

### Learning an oMHN model

The observation MHN (oMHN) extends MHN by modeling observation rates that the active events can influence:

```python
import numpy as np
import fastmhn

# Generate data
d = 5
N = 100
data = np.random.randint(2, size=(N, d), dtype=np.int32)

# Learn oMHN model
theta = fastmhn.learn.learn_omhn(
    data,
    reg=1e-2,
    gradient_and_score_params={"max_cluster_size": 10},
    adam_params={"alpha": 0.1, "beta1": 0.7, "beta2": 0.9, "eps": 1e-8},
)

# theta is a (d+1) x d matrix
# First d rows: MHN parameters
# Last row: observation rates
print(f"Learned oMHN theta matrix:\n{theta}")
```

### Cross-validation for regularization strength

```python
import numpy as np
import fastmhn

# Generate data
d = 5
N = 100
data = np.random.randint(2, size=(N, d), dtype=np.int32)

# Cross-validation parameters
k = 5  # number of folds
reg = 1e-2  # regularization strength to evaluate

# Shuffle data
rng = np.random.default_rng(42)
shuffled_indices = np.arange(N)
rng.shuffle(shuffled_indices)
data = data[shuffled_indices, :]

# Create folds
fold_sizes = (N // k) * np.ones(k, dtype=int)
fold_sizes[: N % k] += 1

# Get score offset for comparison
score_offset = fastmhn.utility.get_score_offset(data)
average_validation_score = 0

for k_index in range(k):
    # Split into training and validation
    val_start = np.sum(fold_sizes[:k_index])
    val_end = np.sum(fold_sizes[: k_index + 1])
    data_val = data[val_start:val_end]
    data_train = np.concatenate((data[:val_start], data[val_end:]))

    # Learn model on training data
    theta = fastmhn.learn.learn_omhn(
        data_train,
        reg=reg,
        gradient_and_score_params={"max_cluster_size": 10},
        adam_params={"verbose": False},
    )

    # Evaluate on validation data
    ctheta = fastmhn.utility.cmhn_from_omhn(theta)
    _, val_score = fastmhn.approx.approx_gradient_and_score(
        ctheta, data_val, max_cluster_size=10
    )
    average_validation_score += val_score

average_validation_score /= k
print(f"Average validation score: {average_validation_score} (offset: {score_offset})")
```

### Using the command-line scripts

The repository includes convenience scripts for common tasks:

- `learn_approx_mhn.py` - Learn an MHN model
- `learn_approx_omhn.py` - Learn an oMHN model
- `learn_approx_omhn_crossvalidated.py` - Learn oMHN with cross-validation

You can use these as templates or run them directly:

```bash
python learn_approx_omhn.py
```

## API Reference

The main functions are accessible through the `fastmhn` package:

- `fastmhn.learn.learn_mhn()` - Learn an MHN model
- `fastmhn.learn.learn_omhn()` - Learn an oMHN model
- `fastmhn.approx.approx_gradient_and_score()` - Approximate gradient and score computation
- `fastmhn.exact.gradient_and_score()` - Exact gradient and score computation
- `fastmhn.utility.create_pD()` - Create probability distribution
- `fastmhn.utility.generate_data()` - Generate synthetic data

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Repository

- GitHub: https://github.com/simon-pfahler/fastmhn
