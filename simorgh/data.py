"""Training-set generation: simulate, then inject noise.

Noise realism lives here, not in the simulator. Each training spectrum
draws its own noise level (and wavelength-dependent shape), so the
trained network is amortized over noise levels and, by conditioning on
the reported per-channel errors, over reduction variants that mostly
differ in error bars and binning. Channel dropout during training
teaches the embedding to tolerate masked channels (bad columns,
detector-gap edges, spot-crossing masks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .priors import BoxPrior
from .spectra import Spectrum


@dataclass
class NoiseModel:
    """Heteroscedastic Gaussian noise with a per-spectrum level draw.

    ppm_min, ppm_max : range of the per-spectrum median error level;
                       drawn log-uniformly (survey-realistic G395H range).
    red_slope        : linear growth of error toward the red end (thermal
                       background), as a fraction over the band.
    p_drop           : per-channel dropout probability during training.
    """

    ppm_min: float = 30.0
    ppm_max: float = 250.0
    red_slope: float = 0.5
    p_drop: float = 0.05

    def sample_errors(self, wavelength: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
        level = np.exp(rng.uniform(np.log(self.ppm_min), np.log(self.ppm_max)))
        x = (wavelength - wavelength.min()) / np.ptp(wavelength)
        shape = 1.0 + self.red_slope * x
        jitter = rng.lognormal(mean=0.0, sigma=0.15, size=wavelength.size)
        return level * 1e-6 * shape * jitter


def make_training_set(simulator, prior: BoxPrior, wavelength: np.ndarray,
                      n_sims: int, noise: NoiseModel | None = None,
                      seed: int = 0):
    """Returns (theta, tokens, masks) as float32/bool numpy arrays.

    tokens : (n_sims, n_chan, 3) rows (wavelength, noisy depth, error)
    masks  : (n_sims, n_chan) channel validity
    """
    noise = noise or NoiseModel()
    rng = np.random.default_rng(seed)
    theta = prior.sample(n_sims, rng)
    clean = simulator(theta, wavelength)              # (n, m)
    m = wavelength.size

    tokens = np.zeros((n_sims, m, 3), dtype=np.float32)
    masks = np.ones((n_sims, m), dtype=bool)
    for i in range(n_sims):
        err = noise.sample_errors(wavelength, rng)
        noisy = clean[i] + rng.normal(0.0, err)
        drop = rng.random(m) < noise.p_drop
        masks[i, drop] = False
        tokens[i, :, 0] = wavelength
        tokens[i, :, 1] = noisy
        tokens[i, :, 2] = err
    return theta.astype(np.float32), tokens, masks


def spectrum_to_batch(spec: Spectrum, n_chan: int | None = None):
    """One observed Spectrum -> (tokens, mask) batch of size 1."""
    tokens, mask = spec.to_tokens(max_len=n_chan)
    return tokens[None, ...], mask[None, ...]


class NoisyGridDataset:
    """Clean grid + noise resampled on every access.

    The forward-model grid is expensive and fixed; noise is cheap and
    should be fresh. Drawing a new noise level, error shape and dropout
    mask each time an item is served means the network sees every clean
    spectrum under many observing conditions, which is what makes the
    posterior amortized over noise rather than tied to one realization.

    Implements the torch Dataset interface without importing torch, so
    it stays usable from plain numpy code and tests.
    """

    def __init__(self, theta: np.ndarray, depth: np.ndarray,
                 wavelength: np.ndarray, noise: NoiseModel | None = None,
                 seed: int = 0, fixed: bool = False):
        if theta.shape[0] != depth.shape[0]:
            raise ValueError("theta and depth must have the same length")
        if depth.shape[1] != wavelength.shape[0]:
            raise ValueError("depth columns must match the wavelength grid")
        self.theta = np.asarray(theta, dtype=np.float32)
        self.depth = np.asarray(depth, dtype=np.float32)
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.noise = noise or NoiseModel()
        self.seed = seed
        self.fixed = fixed        # True -> deterministic (validation sets)

    def __len__(self) -> int:
        return self.theta.shape[0]

    def _rng(self, idx: int, epoch_salt: int) -> np.random.Generator:
        if self.fixed:
            return np.random.default_rng([self.seed, idx])
        return np.random.default_rng([self.seed, idx, epoch_salt])

    def __getitem__(self, idx: int):
        # Without torch's epoch hook, salt from a global counter bumped by
        # the training loop via `set_epoch`.
        rng = self._rng(idx, getattr(self, "_epoch", 0))
        err = self.noise.sample_errors(self.wavelength, rng)
        noisy = self.depth[idx] + rng.normal(0.0, err)
        mask = rng.random(self.wavelength.size) >= self.noise.p_drop
        tokens = np.stack([self.wavelength, noisy, err], axis=-1)
        return (self.theta[idx],
                tokens.astype(np.float32),
                mask.astype(bool))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
