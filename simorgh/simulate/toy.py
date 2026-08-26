"""Analytic toy sub-Neptune transmission simulator.

NOT physics-grade — a stand-in with the right *structure* so the whole
pipeline (embedding, NPE, SBC/TARP, population reweighting) can be built
and tested end-to-end before the TauREx grid exists. It reproduces the
qualitative degeneracies that make sub-Neptune retrieval hard:

  - feature amplitude ~ H/R* = kT / (mu(Z) g R*): temperature, metallicity
    (through mean molecular weight) and gravity are degenerate,
  - a gray cloud deck mutes features (cloud/metallicity degeneracy),
  - CO2 4.3 um amplitude scales super-linearly with metallicity,
  - baseline depth sets rp_rs, features ride on top.

Parameter vector (see simorgh.priors.subneptune_prior):
  t_eq, log10_met, log10_pcloud, rp_rs, log10_g
"""

from __future__ import annotations

import numpy as np

K_BOLTZ = 1.380649e-16   # erg / K
M_H = 1.6605e-24         # g
R_STAR_CM = 0.45 * 6.957e10  # fixed toy host: 0.45 R_sun

# (center um, width um, base strength, metallicity exponent)
_BANDS = (
    (2.95, 0.15, 1.6, 1.0),   # H2O
    (3.35, 0.12, 1.8, 0.7),   # CH4 (falls off at high Z in reality; keep mild)
    (4.30, 0.10, 2.2, 2.0),   # CO2 — the Z^2 marker
    (4.65, 0.12, 0.9, 1.2),   # CO
)


def mean_molecular_weight(log10_met: float | np.ndarray) -> np.ndarray:
    """Smooth mu(Z): 2.3 (solar, H2/He) -> ~19 (1000x solar, steam-rich)."""
    z = np.asarray(log10_met, dtype=np.float64)
    return 2.3 + 16.7 * (10.0 ** z / 1000.0) ** 0.7


class ToySubNeptune:
    """Deterministic analytic forward model. Vectorized over theta rows."""

    param_names = ("t_eq", "log10_met", "log10_pcloud", "rp_rs", "log10_g")

    def __call__(self, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        squeeze = np.asarray(theta).ndim == 1
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        lam = np.asarray(wavelength, dtype=np.float64)
        t_eq, log10_met, log10_pcloud, rp_rs, log10_g = theta.T

        mu = mean_molecular_weight(log10_met)
        g_cgs = 10.0 ** log10_g
        h_cm = K_BOLTZ * t_eq / (mu * M_H * g_cgs)
        h_rs = h_cm / R_STAR_CM                                # (n,)

        # opacity in scale-height units, per band, metallicity-scaled
        met = 10.0 ** log10_met
        n_gas = np.full((theta.shape[0], lam.size), 1.0)       # continuum floor
        for center, width, strength, zexp in _BANDS:
            profile = np.exp(-0.5 * ((lam - center) / width) ** 2)
            amp = strength * (met / 100.0) ** (0.35 * zexp)
            n_gas = n_gas + amp[:, None] * profile[None, :]
        n_gas = np.log1p(n_gas) * 2.5   # opacity -> scale heights (log law)

        # gray cloud deck: altitude in scale heights above the reference level
        n_cloud = 2.0 * (-np.asarray(log10_pcloud) + 1.0)      # p=10 bar -> 0
        n_eff = np.maximum(n_gas, n_cloud[:, None])

        r_eff = rp_rs[:, None] + h_rs[:, None] * n_eff
        depth = r_eff ** 2
        return depth[0] if squeeze else depth
