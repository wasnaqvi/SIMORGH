"""Parameter space and priors for the uniform sub-Neptune scope.

The prior object is load-bearing for the population contract: hierarchical
reweighting divides by the *interim* prior pi(theta) used at training time
(Hogg, Myers & Bovy 2010; the GW population playbook, e.g. Leyde et al.
2024). It must therefore be evaluable, not just sampleable, and it is
stored with any trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxPrior:
    """Independent uniform prior on a named parameter box.

    names  : parameter names, length d
    low    : lower bounds, shape (d,)
    high   : upper bounds, shape (d,)
    """

    names: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "low", np.asarray(self.low, dtype=np.float64))
        object.__setattr__(self, "high", np.asarray(self.high, dtype=np.float64))
        d = len(self.names)
        if self.low.shape != (d,) or self.high.shape != (d,):
            raise ValueError("low/high must match names length")
        if np.any(self.high <= self.low):
            raise ValueError("require high > low")

    @property
    def dim(self) -> int:
        return len(self.names)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=(n, self.dim))

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        inside = np.all((theta >= self.low) & (theta <= self.high), axis=1)
        lp = np.full(theta.shape[0], -np.inf)
        lp[inside] = -np.sum(np.log(self.high - self.low))
        return lp

    def to_unit(self, theta: np.ndarray) -> np.ndarray:
        """Map physical parameters to the unit box (for TARP / training)."""
        return (theta - self.low) / (self.high - self.low)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.low + u * (self.high - self.low)


def subneptune_prior() -> BoxPrior:
    """Training prior for the uniform sub-Neptune G395H scope.

    Deliberately wide relative to the Patchwork sample so every survey
    target is interior to the training box. Amortization is over planets:
    r_p, r_s and g are parameters, not constants (FASTER gap G2).

    Parameters
    ----------
    t_eq        : equilibrium temperature [K]
    log10_met   : metallicity [x solar], log10
    log10_pcloud: gray cloud-top pressure [bar], log10 (high value = clear)
    rp_rs       : radius ratio
    log10_g     : surface gravity [cgs], log10
    """
    return BoxPrior(
        names=("t_eq", "log10_met", "log10_pcloud", "rp_rs", "log10_g"),
        low=np.array([300.0, 0.0, -5.0, 0.01, 2.6]),
        high=np.array([1100.0, 3.0, 1.0, 0.08, 3.6]),
    )
