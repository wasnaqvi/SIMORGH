"""TauREx 3 adapter — the production forward model for the survey scope.

Lazy-imported so the rest of SIMORGH runs without a TauREx install.
This is an adapter around a *pre-configured* TauREx model: opacity paths,
line lists and layer structure are pinned at construction and recorded in
`provenance`, because a trained network is only meaningful relative to a
frozen simulator (the same uniformity discipline as Patchwork's frozen
exoTEDRF config).

Status: interface complete, mapping implemented for an isothermal,
equilibrium-scaled-metallicity sub-Neptune model; requires a local TauREx
+ opacity installation to run. The toy simulator is the default until the
production grid is generated on the cluster.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


class TaurexSubNeptune:
    """theta = (t_eq, log10_met, log10_pcloud, rp_rs, log10_g) -> depth(lam).

    Parameters
    ----------
    opacity_path, cia_path : absolute paths to cross sections / CIA tables
    star_radius_rsun       : frozen toy-host convention is 0.45; production
                             models should pass the per-target value and
                             record it in provenance.
    """

    param_names = ("t_eq", "log10_met", "log10_pcloud", "rp_rs", "log10_g")

    def __init__(self, opacity_path: str, cia_path: str,
                 star_radius_rsun: float = 0.45, nlayers: int = 100):
        # Import here: TauREx pins its own dependency stack.
        from taurex.cache import OpacityCache, CIACache  # noqa: F401

        OpacityCache().set_opacity_path(opacity_path)
        CIACache().set_cia_path(cia_path)
        self._star_radius = star_radius_rsun
        self._nlayers = nlayers
        self.provenance = {
            "engine": "taurex3",
            "opacity_path": opacity_path,
            "cia_path": cia_path,
            "star_radius_rsun": star_radius_rsun,
            "nlayers": nlayers,
        }

    @property
    def provenance_hash(self) -> str:
        blob = json.dumps(self.provenance, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def __call__(self, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        from taurex.model import TransmissionModel
        from taurex.planet import Planet
        from taurex.stellar import BlackbodyStar
        from taurex.temperature import Isothermal
        from taurex.chemistry import TaurexChemistry, ConstantGas
        from taurex.contributions import (AbsorptionContribution,
                                          RayleighContribution,
                                          CIAContribution,
                                          SimpleCloudsContribution)

        squeeze = np.asarray(theta).ndim == 1
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        lam = np.asarray(wavelength, dtype=np.float64)
        wngrid = np.sort(10000.0 / lam)

        out = np.empty((theta.shape[0], lam.size))
        for i, (t_eq, log10_met, log10_pcloud, rp_rs, log10_g) in enumerate(theta):
            r_star_rj = self._star_radius * 6.957e10 / 7.1492e9
            rp_rj = rp_rs * r_star_rj
            # M = g R^2 / G, in Jupiter masses
            g_cgs = 10.0 ** log10_g
            mp_mj = g_cgs * (rp_rj * 7.1492e9) ** 2 / 6.674e-8 / 1.898e30

            star = BlackbodyStar(temperature=3800.0, radius=self._star_radius)
            planet = Planet(planet_radius=rp_rj, planet_mass=mp_mj)
            temperature = Isothermal(T=t_eq)
            chemistry = TaurexChemistry(fill_gases=["H2", "He"], ratio=0.17)
            # metallicity-scaled abundances (solar anchors x 10^log10_met)
            for gas, solar in (("H2O", 1.0e-3), ("CH4", 4.0e-4),
                               ("CO2", 1.0e-7), ("CO", 4.0e-4)):
                zexp = 2.0 if gas == "CO2" else 1.0
                chemistry.addGas(ConstantGas(
                    gas, mix_ratio=min(solar * (10.0 ** log10_met) ** zexp, 0.3)))

            model = TransmissionModel(
                planet=planet, star=star, temperature_profile=temperature,
                chemistry=chemistry, nlayers=self._nlayers,
                atm_min_pressure=1e-1, atm_max_pressure=1e6)  # Pa
            model.add_contribution(AbsorptionContribution())
            model.add_contribution(RayleighContribution())
            model.add_contribution(CIAContribution(cia_pairs=["H2-H2", "H2-He"]))
            model.add_contribution(
                SimpleCloudsContribution(clouds_pressure=10.0 ** log10_pcloud * 1e5))
            model.build()
            native_wn, native_depth, _, _ = model.model(wngrid=wngrid)
            # model() returns on the requested grid when wngrid is passed
            out[i] = np.interp(10000.0 / lam, native_wn[::-1], native_depth[::-1])
        return out[0] if squeeze else out
