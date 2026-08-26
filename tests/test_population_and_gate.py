"""Population reweighting recovers injected hyperparameters from mock
per-planet posteriors; the OOD gate separates shifted data and is blind
to in-distribution data."""

import numpy as np
import pytest

from simorgh.data import NoiseModel, make_training_set
from simorgh.gate import DensityRatioGate
from simorgh.population import (gaussian_population, grid_hyperposterior,
                                hyper_log_likelihood)
from simorgh.priors import subneptune_prior
from simorgh.simulate import ToySubNeptune, g395h_grid


def _mock_events(rng, n_events, mu, sigma, prior, n_samp=800, width=0.25):
    """Fake population_export dicts: per-planet 'posteriors' are Gaussians
    of the given width centred near a truth drawn from N(mu, sigma), in
    the log10_met dimension (index 1); other dims drawn from the prior."""
    events = []
    for _ in range(n_events):
        truth = np.clip(rng.normal(mu, sigma), 0.05, 2.95)
        samples = prior.sample(n_samp, rng)
        samples[:, 1] = np.clip(rng.normal(truth, width, n_samp), 0.001, 2.999)
        events.append({
            "samples": samples,
            # interim prior is uniform on the box -> constant log pi
            "log_interim_prior": prior.log_prob(samples),
        })
    return events


def test_hyperposterior_recovers_truth():
    prior = subneptune_prior()
    rng = np.random.default_rng(11)
    mu_true, sigma_true = 1.6, 0.4
    events = _mock_events(rng, n_events=40, mu=mu_true, sigma=sigma_true,
                          prior=prior)
    pop = gaussian_population(param_index=1)
    mu_grid = np.linspace(0.8, 2.4, 33)
    sigma_grid = np.linspace(0.1, 1.0, 25)
    post = grid_hyperposterior(events, pop, mu_grid, sigma_grid)

    i, j = np.unravel_index(post.argmax(), post.shape)
    assert abs(mu_grid[i] - mu_true) < 0.2
    assert abs(sigma_grid[j] - sigma_true) < 0.25


def test_ess_guard_trips():
    prior = subneptune_prior()
    rng = np.random.default_rng(5)
    events = _mock_events(rng, n_events=1, mu=0.3, sigma=0.05, prior=prior,
                          n_samp=50, width=0.02)
    pop = gaussian_population(param_index=1)
    # population model far from every sample -> weight collapse -> guard
    with pytest.raises(RuntimeError, match="ESS"):
        hyper_log_likelihood(events, pop, np.array([2.9, 0.01]))


def test_ood_gate_detects_shift_and_passes_indist():
    sim, prior, grid = ToySubNeptune(), subneptune_prior(), g395h_grid()
    noise = NoiseModel(ppm_min=40, ppm_max=100)
    _, tok_sim, mk_sim = make_training_set(sim, prior, grid, 200,
                                           noise=noise, seed=0)
    # "observed" set A: same distribution -> gate should find nothing
    _, tok_ind, mk_ind = make_training_set(sim, prior, grid, 200,
                                           noise=noise, seed=1)
    gate = DensityRatioGate(seed=0)
    auc_ind = gate.fit(tok_sim, mk_sim, tok_ind, mk_ind)
    assert auc_ind < 0.65

    # "observed" set B: correlated noise the simulator never produced
    _, tok_ood, mk_ood = make_training_set(sim, prior, grid, 200,
                                           noise=noise, seed=2)
    rng = np.random.default_rng(3)
    for t in tok_ood:
        wave = 80e-6 * np.sin(2 * np.pi * np.arange(grid.size)
                              / rng.uniform(8, 15))
        t[:, 1] += wave.astype(np.float32)
    gate2 = DensityRatioGate(seed=0)
    auc_ood = gate2.fit(tok_sim, mk_sim, tok_ood, mk_ood)
    assert auc_ood > 0.85
