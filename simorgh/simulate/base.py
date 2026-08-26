"""Simulator protocol.

A simulator maps parameters to a noiseless transit-depth spectrum on a
requested wavelength grid. Noise is NOT the simulator's job — it is
injected by the training-data generator, so one simulator serves every
noise model and the network can be amortized over noise levels.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Simulator(Protocol):
    """Deterministic forward model."""

    def __call__(self, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        """theta (d,) or (n, d) -> depth (m,) or (n, m) on wavelength (m,)."""
        ...


def g395h_grid(resolution: float = 100.0,
               lam_min: float = 2.85,
               lam_max: float = 5.15) -> np.ndarray:
    """Constant-R wavelength grid over the G395H band (NRS1+NRS2), micron.

    Matches the Patchwork convention of constant-R binning; the detector
    gap is not modeled here — real Patchwork products carry their own grid
    and the network conditions on whatever grid it is given.
    """
    n = int(np.ceil(np.log(lam_max / lam_min) * resolution))
    return lam_min * np.exp(np.arange(n) / resolution)
