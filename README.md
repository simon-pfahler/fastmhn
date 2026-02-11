# FastMHN - Fast inference of MHNs

This project aims to allow for the fast inference of MHNs through suitable rank-1 approximations of the time-marginalized probability distributions.

## Getting started
The easiest way to use `fastmhn` is to clone the repository and install the package using `pip`.
This will automatically install the dependencies `joblib`, `numpy`, `mhn`.

The scripts `learn_approx_mhn.py` and `learn_approx_omhn.py` provide starting points for learning MHNs and oMHNs using this package. Additionally, the script `learn_approx_omhn_crossvalidated.py` performs crossvalidation to find out the optimal regularization strength.
