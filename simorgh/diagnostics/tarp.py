"""TARP — Tests of Accuracy with Random Points (Lemos et al. 2023).

Joint, multidimensional expected-coverage test needing only posterior
samples: for each simulation draw a random reference point, compute the
fraction of posterior samples closer to it than the truth is, and check
that these credibility levels are uniform. Catches joint miscalibration
that per-parameter SBC misses.
"""

from __future__ import annotations

import numpy as np


def tarp_coverage(model, prior, theta_true: np.ndarray, tokens: np.ndarray,
                  masks: np.ndarray, n_post: int = 250,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (alpha, ecp): expected coverage probability at credibility
    alpha. Calibrated => ecp ~= alpha. Distances in the unit-box metric."""
    rng = np.random.default_rng(seed)
    n = theta_true.shape[0]
    f = np.empty(n)
    for i in range(n):
        samp_u = prior.to_unit(model.sample(n_post, tokens[i:i + 1],
                                            masks[i:i + 1]))
        true_u = prior.to_unit(theta_true[i])
        ref = rng.uniform(size=true_u.shape)
        d_samp = np.linalg.norm(samp_u - ref, axis=1)
        d_true = np.linalg.norm(true_u - ref)
        f[i] = (d_samp < d_true).mean()
    alpha = np.linspace(0.0, 1.0, 51)
    ecp = np.array([(f < a).mean() for a in alpha])
    return alpha, ecp


def tarp_max_deviation(alpha: np.ndarray, ecp: np.ndarray) -> float:
    """max |ECP - alpha|; ~sqrt(1/n) noise floor for n simulations."""
    return float(np.max(np.abs(ecp - alpha)))
