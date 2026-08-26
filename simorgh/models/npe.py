"""Noise-conditioned neural posterior estimator with a population contract.

The flow models z = logit((theta - low)/(high - low)): posterior support
is exactly the prior box by construction, and log-density evaluations are
returned in PHYSICAL parameter space (Jacobian included). That evaluable
density, together with the stored interim prior, is the population
contract: hierarchical reweighting (Hogg, Myers & Bovy 2010; Leyde et
al. 2024) consumes log q(theta | x) and log pi(theta) — samples alone,
as delivered by nested sampling or 1D-marginal methods, cannot feed a
population loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zuko

from ..priors import BoxPrior
from .embedding import SpectrumEmbedding

_EPS = 1e-6


class AmortizedPosterior(nn.Module):
    def __init__(self, prior: BoxPrior, context_dim: int = 64,
                 transforms: int = 5, hidden: tuple[int, ...] = (128, 128),
                 provenance: dict | None = None):
        super().__init__()
        self.prior = prior
        self.embedding = SpectrumEmbedding(out_dim=context_dim)
        self.flow = zuko.flows.NSF(features=prior.dim, context=context_dim,
                                   transforms=transforms,
                                   hidden_features=hidden)
        # Full architecture, recorded so load() rebuilds an identical shape.
        self.architecture = {"context_dim": int(context_dim),
                             "transforms": int(transforms),
                             "hidden": [int(h) for h in hidden]}
        self.provenance = dict(provenance or {})
        # Buffers, not plain tensors: they must follow .to(device) and be
        # recorded in the state dict, so a checkpoint is self-describing.
        self.register_buffer("_low", torch.tensor(prior.low, dtype=torch.float32))
        self.register_buffer("_high", torch.tensor(prior.high, dtype=torch.float32))

    @property
    def device(self) -> torch.device:
        return self._low.device

    # -- parameter transforms ------------------------------------------------
    def _to_z(self, theta: torch.Tensor) -> torch.Tensor:
        u = (theta - self._low) / (self._high - self._low)
        u = u.clamp(_EPS, 1.0 - _EPS)
        return torch.logit(u)

    def _from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self._low + torch.sigmoid(z) * (self._high - self._low)

    def _log_jac_z_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """log |dz/dtheta| summed over dims (adds to z-space log density)."""
        u = ((theta - self._low) / (self._high - self._low)).clamp(_EPS, 1 - _EPS)
        return (-torch.log(u) - torch.log1p(-u)
                - torch.log(self._high - self._low)).sum(dim=-1)

    # -- training objective --------------------------------------------------
    def loss(self, theta: torch.Tensor, tokens: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        ctx = self.embedding(tokens, mask)
        z = self._to_z(theta)
        return -self.flow(ctx).log_prob(z).mean()

    # -- inference API (physical space) --------------------------------------
    @torch.no_grad()
    def sample(self, n: int, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        dev = self.device
        ctx = self.embedding(
            torch.as_tensor(tokens, dtype=torch.float32, device=dev),
            torch.as_tensor(mask, dtype=torch.bool, device=dev))
        z = self.flow(ctx).sample((n,))          # (n, b, d)
        return self._from_z(z).squeeze(1).cpu().numpy()

    @torch.no_grad()
    def log_prob(self, theta: np.ndarray, tokens: np.ndarray,
                 mask: np.ndarray) -> np.ndarray:
        """log q(theta | x) in physical units. theta (n, d); one spectrum."""
        dev = self.device
        th = torch.as_tensor(np.atleast_2d(theta), dtype=torch.float32,
                             device=dev)
        ctx = self.embedding(
            torch.as_tensor(tokens, dtype=torch.float32, device=dev),
            torch.as_tensor(mask, dtype=torch.bool, device=dev))
        ctx = ctx.expand(th.shape[0], -1)
        z = self._to_z(th)
        return (self.flow(ctx).log_prob(z)
                + self._log_jac_z_theta(th)).cpu().numpy()

    def population_export(self, tokens: np.ndarray, mask: np.ndarray,
                          n_samples: int = 2000) -> dict:
        """Everything a hierarchical model needs from this spectrum."""
        samples = self.sample(n_samples, tokens, mask)
        return {
            "samples": samples,
            "log_q": self.log_prob(samples, tokens, mask),
            "log_interim_prior": self.prior.log_prob(samples),
            "param_names": list(self.prior.names),
            "provenance": dict(self.provenance),
        }

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "weights.pt")
        meta = {
            "names": list(self.prior.names),
            "low": self.prior.low.tolist(),
            "high": self.prior.high.tolist(),
            "architecture": self.architecture,
            "provenance": self.provenance,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "AmortizedPosterior":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        prior = BoxPrior(names=tuple(meta["names"]),
                         low=np.array(meta["low"]),
                         high=np.array(meta["high"]))
        arch = meta.get("architecture") or {"context_dim": meta["context_dim"]}
        model = cls(prior,
                    context_dim=arch["context_dim"],
                    transforms=arch.get("transforms", 5),
                    hidden=tuple(arch.get("hidden", (128, 128))),
                    provenance=meta["provenance"])
        model.load_state_dict(torch.load(path / "weights.pt",
                                         map_location=map_location,
                                         weights_only=True))
        model.eval()
        return model
