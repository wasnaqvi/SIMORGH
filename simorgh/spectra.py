"""Spectrum container.

A transmission spectrum is a variable-length set of channels
(wavelength, depth, error) plus a validity mask. Variable length and
per-channel errors are first-class: the network is conditioned on both,
which is what makes the model amortized over instruments, binnings,
reduction variants, and masked channels (the fm4ar noise-conditioning
idea, extended to the wavelength grid itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Spectrum:
    """One transmission spectrum.

    wavelength : micron, shape (n,)
    depth      : transit depth (dimensionless, e.g. 0.005 = 5000 ppm), shape (n,)
    error      : 1-sigma depth uncertainty, shape (n,)
    mask       : True where the channel is valid, shape (n,)
    meta       : free-form provenance (planet, visit, reduction id, ...)
    """

    wavelength: np.ndarray
    depth: np.ndarray
    error: np.ndarray
    mask: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=np.float64)
        self.depth = np.asarray(self.depth, dtype=np.float64)
        self.error = np.asarray(self.error, dtype=np.float64)
        n = self.wavelength.shape[0]
        if self.depth.shape != (n,) or self.error.shape != (n,):
            raise ValueError("wavelength, depth, error must share shape (n,)")
        if self.mask is None:
            self.mask = np.ones(n, dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != (n,):
                raise ValueError("mask must share shape (n,)")
        if np.any(self.error[self.mask] <= 0):
            raise ValueError("errors must be positive on valid channels")

    @property
    def n_channels(self) -> int:
        return int(self.mask.sum())

    def to_tokens(self, max_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Pack into (tokens, mask) for the embedding network.

        tokens : (max_len, 3) rows of (wavelength, depth, error), zero-padded
        mask   : (max_len,) True on real channels
        """
        n = self.wavelength.shape[0]
        L = n if max_len is None else max_len
        if n > L:
            raise ValueError(f"spectrum has {n} channels > max_len {L}")
        tokens = np.zeros((L, 3), dtype=np.float32)
        out_mask = np.zeros(L, dtype=bool)
        tokens[:n, 0] = self.wavelength
        tokens[:n, 1] = self.depth
        tokens[:n, 2] = self.error
        out_mask[:n] = self.mask
        return tokens, out_mask
