"""Simulation-based calibration (Cook et al. 2006; Talts et al. 2018).

Per-parameter rank statistics: for each held-out simulation, the rank of
the true parameter among posterior samples should be uniform if the
posterior is calibrated. Necessary but not sufficient (marginal, per
dimension) — pair with TARP for joint coverage.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def sbc_ranks(model, theta_true: np.ndarray, tokens: np.ndarray,
              masks: np.ndarray, n_post: int = 250) -> np.ndarray:
    """Ranks in [0, n_post], shape (n_sims, d)."""
    n, d = theta_true.shape
    ranks = np.empty((n, d), dtype=np.int64)
    for i in range(n):
        samp = model.sample(n_post, tokens[i:i + 1], masks[i:i + 1])
        ranks[i] = (samp < theta_true[i]).sum(axis=0)
    return ranks


def sbc_pvalues(ranks: np.ndarray, n_post: int = 250) -> np.ndarray:
    """KS test of rank uniformity per parameter. Small p = miscalibrated."""
    d = ranks.shape[1]
    return np.array([
        stats.kstest((ranks[:, j] + 0.5) / (n_post + 1), "uniform").pvalue
        for j in range(d)
    ])
