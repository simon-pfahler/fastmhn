"""
fastmhn -- Fast inference of MHN (Mutual Hazard Networks) models.

Version: 1.0.1

Modules
-------
utility     : Utility functions for data generation, pD creation, etc.
explicit    : Exact calculations using full state space
exact       : Alternative exact implementations
approx      : Approximate calculations using clustering
clustering  : Hierarchical clustering algorithms
learn       : Learning algorithms (Adam, AdamW) and model fitting
"""

__version__ = "1.0.1"

from . import approx, clustering, exact, explicit, learn, utility
from .approx import approx_gradient_and_score
from .clustering import hierarchical_clustering
from .exact import gradient_and_score
from .explicit import (
    apply_eye_minus_Q,
    apply_eye_minus_Q_diag,
    apply_eye_minus_Q_offdiag,
    apply_Qdiff_ii,
    calculate_pTheta,
    create_full_Q,
    score,
)
from .learn import learn_mhn, learn_omhn
from .utility import (
    adam,
    adamW,
    backward_substitution,
    cmhn_from_omhn,
    create_indep_model,
    create_pD,
    forward_substitution,
    generate_data,
    generate_theta,
    get_score_offset,
    get_subdata,
    jacobi,
)

__all__ = [
    # utility
    "adam",
    "adamW",
    "backward_substitution",
    "cmhn_from_omhn",
    "create_indep_model",
    "create_pD",
    "forward_substitution",
    "generate_data",
    "generate_theta",
    "get_score_offset",
    "get_subdata",
    "jacobi",
    # exact
    "gradient_and_score",
    # explicit
    "apply_Qdiff_ii",
    "apply_eye_minus_Q",
    "apply_eye_minus_Q_diag",
    "apply_eye_minus_Q_offdiag",
    "calculate_pTheta",
    "create_full_Q",
    "score",
    # approx
    "approx_gradient_and_score",
    # clustering
    "hierarchical_clustering",
    # learn
    "learn_mhn",
    "learn_omhn",
    # submodules
    "approx",
    "clustering",
    "exact",
    "explicit",
    "learn",
    "utility",
]
