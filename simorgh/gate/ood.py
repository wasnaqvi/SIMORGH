"""Out-of-distribution gate: sim-vs-real density-ratio classifier.

The open problem both incumbents name in print (FASTER Sec. 4.2: whether
observed data are in-distribution w.r.t. the training set "is a key open
issue"; Ariel WP gates SBI adoption on calibration under shift). A
classifier trained to separate simulated from observed spectra yields
(i) an AUC — a scalar "how far is real data from the training
distribution", and (ii) per-spectrum density-ratio weights, the input
weighted conformal prediction (Tibshirani et al. 2019) needs to restore
coverage under covariate shift. Weighted-CP layer itself: next phase.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _summaries(tokens: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Cheap per-spectrum summary features (classifier inputs)."""
    feats = []
    for tok, mk in zip(tokens, masks):
        d, e = tok[mk, 1], tok[mk, 2]
        resid = d - np.median(d)
        feats.append([
            np.median(d), np.std(d), np.median(e), np.std(e),
            np.mean(np.abs(np.diff(resid))),      # channel-to-channel jitter
            np.std(resid / e),                    # scatter in error units
            mk.mean(),                            # masked fraction
        ])
    return np.asarray(feats, dtype=np.float32)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    rank = np.empty_like(order, dtype=float)
    rank[order] = np.arange(len(scores))
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("need both classes in the evaluation split")
    return float((rank[labels == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


class DensityRatioGate:
    """Fit on (simulated, observed) sets; score new spectra."""

    def __init__(self, epochs: int = 200, lr: float = 1e-2, seed: int = 0):
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.net: nn.Module | None = None
        self._mu = self._sd = None

    def fit(self, sim_tokens, sim_masks, obs_tokens, obs_masks) -> float:
        """Returns HELD-OUT AUC (50/50 split). ~0.5 means the observed set
        is indistinguishable from the simulations; near 1.0 means shift."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        xs = _summaries(sim_tokens, sim_masks)
        xo = _summaries(obs_tokens, obs_masks)
        x = np.concatenate([xs, xo])
        y = np.concatenate([np.zeros(len(xs)), np.ones(len(xo))])
        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]
        n_tr = len(x) // 2
        self._mu, self._sd = x[:n_tr].mean(0), x[:n_tr].std(0) + 1e-8
        xt = torch.as_tensor((x - self._mu) / self._sd)
        yt = torch.as_tensor(y, dtype=torch.float32)

        self.net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.GELU(),
                                 nn.Linear(32, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = lossf(self.net(xt[:n_tr]).squeeze(-1), yt[:n_tr])
            loss.backward()
            opt.step()

        with torch.no_grad():
            s = self.net(xt[n_tr:]).squeeze(-1).numpy()
        return _auc(s, y[n_tr:])

    @torch.no_grad()
    def log_ratio(self, tokens, masks) -> np.ndarray:
        """log[p_obs/p_sim] per spectrum — the weighted-CP weight."""
        if self.net is None:
            raise RuntimeError("fit() first")
        x = (_summaries(tokens, masks) - self._mu) / self._sd
        return self.net(torch.as_tensor(x)).squeeze(-1).numpy()
