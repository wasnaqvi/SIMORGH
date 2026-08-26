"""Hierarchical population inference by importance reweighting.

The standard estimator (Hogg, Myers & Bovy 2010; GW population practice,
e.g. Leyde et al. 2024): given per-planet posterior samples theta_ns
drawn under an interim prior pi, the population log-likelihood for
hyperparameters Lambda is

  log L(Lambda) = sum_n log( (1/S) sum_s p(theta_ns | Lambda) / pi(theta_ns) )
                  - N log alpha(Lambda)

alpha(Lambda) is the detection efficiency (selection function); with a
targeted JWST survey it is NOT an easily simulable quantity — targets
were chosen by committees, not thresholds — so the default here is
alpha = 1 with the caveat stated loudly: resulting hyperposteriors are
conditional on the observed target list. Ariel's defined survey tiers
will make alpha computable; the interface anticipates that.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def hyper_log_likelihood(
    events: Sequence[dict],
    pop_logpdf: Callable[[np.ndarray, np.ndarray], np.ndarray],
    hyper: np.ndarray,
    log_alpha: Callable[[np.ndarray], float] | None = None,
    min_ess: float = 10.0,
    on_low_ess: str = "error",
) -> float:
    """events: population_export() dicts (need 'samples',
    'log_interim_prior'). pop_logpdf(samples, hyper) -> log p(theta|Lambda)
    per sample.

    Effective-sample-size collapse means the Monte Carlo estimate at this
    Lambda is unreliable (typically biased low). on_low_ess:
      "error"  raise — the right default inside an MCMC, where silent bias
               near the posterior bulk would corrupt the hyperposterior;
      "neginf" return -inf — for grid scans, where far corners of
               hyperparameter space collapse the weights by construction
               and carry no posterior mass anyway.
    """
    total = 0.0
    for ev in events:
        samples = ev["samples"]
        log_pi = ev["log_interim_prior"]
        log_w = pop_logpdf(samples, hyper) - log_pi
        m = log_w.max()
        if not np.isfinite(m):
            return -np.inf
        w = np.exp(log_w - m)
        ess = w.sum() ** 2 / (w ** 2).sum()
        if ess < min_ess:
            if on_low_ess == "neginf":
                return -np.inf
            raise RuntimeError(
                f"ESS {ess:.1f} < {min_ess} for one event at "
                f"hyper={np.asarray(hyper)}: interim prior too far from "
                "population model; draw more posterior samples or use "
                "on_low_ess='neginf' for exploratory scans.")
        total += m + np.log(w.mean())
    if log_alpha is not None:
        total -= len(events) * log_alpha(hyper)
    return float(total)


def gaussian_population(param_index: int):
    """Demo population model: theta[param_index] ~ N(mu, sigma),
    hyper = (mu, sigma). The real science cases (mass-metallicity slope,
    haze occurrence) plug in here with the same signature."""

    def logpdf(samples: np.ndarray, hyper: np.ndarray) -> np.ndarray:
        mu, sigma = hyper
        x = samples[:, param_index]
        return (-0.5 * ((x - mu) / sigma) ** 2
                - np.log(sigma) - 0.5 * np.log(2.0 * np.pi))

    return logpdf


def grid_hyperposterior(events: Sequence[dict], pop_logpdf, mu_grid, sigma_grid,
                        log_alpha=None) -> np.ndarray:
    """Brute-force normalized hyperposterior on a (mu, sigma) grid — for
    tests and demos; real analyses should MCMC over hyper_log_likelihood."""
    logl = np.full((len(mu_grid), len(sigma_grid)), -np.inf)
    for i, mu in enumerate(mu_grid):
        for j, sg in enumerate(sigma_grid):
            logl[i, j] = hyper_log_likelihood(
                events, pop_logpdf, np.array([mu, sg]), log_alpha,
                on_low_ess="neginf")
    logl -= logl.max()
    post = np.exp(logl)
    return post / post.sum()
