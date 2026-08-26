"""End-to-end: train a small NPE on the toy simulator, check that the
posterior (a) concentrates on truth for a high-SNR spectrum, (b) passes
SBC/TARP at test-scale tolerances, (c) round-trips through save/load,
and (d) exports a valid population contract. Sized for minutes on CPU;
statistical assertions are deliberately loose."""

import numpy as np
import pytest
import torch

from simorgh.data import NoiseModel, make_training_set
from simorgh.diagnostics import (sbc_pvalues, sbc_ranks, tarp_coverage,
                                 tarp_max_deviation)
from simorgh.priors import subneptune_prior
from simorgh.simulate import ToySubNeptune, g395h_grid
from simorgh.train import train_npe


@pytest.fixture(scope="module")
def trained():
    torch.set_num_threads(2)
    sim, prior, grid = ToySubNeptune(), subneptune_prior(), g395h_grid()
    model, history = train_npe(
        sim, prior, grid, n_sims=15_000, epochs=20, batch_size=256,
        noise=NoiseModel(ppm_min=30, ppm_max=120), seed=0,
        provenance={"simulator": "ToySubNeptune", "test": True},
        verbose=False)
    return sim, prior, grid, model, history


def test_training_converged(trained):
    *_, history = trained
    assert history["val"][-1] < history["val"][0] - 1.0


def test_posterior_concentrates_on_truth(trained):
    sim, prior, grid, model, _ = trained
    rng = np.random.default_rng(7)
    theta_true = np.array([650.0, 1.8, 0.5, 0.045, 3.1], dtype=np.float32)
    err = np.full(grid.size, 40e-6)
    depth = sim(theta_true.astype(np.float64), grid) + rng.normal(0, err)
    tokens = np.stack([grid, depth, err], axis=-1).astype(np.float32)[None]
    mask = np.ones((1, grid.size), dtype=bool)

    samp = model.sample(2000, tokens, mask)
    # truth within central 99% interval per parameter, and posterior much
    # tighter than the prior for the well-constrained parameters
    lo, hi = np.percentile(samp, [0.5, 99.5], axis=0)
    assert np.all(theta_true >= lo) and np.all(theta_true <= hi)
    width = np.percentile(samp, 84, axis=0) - np.percentile(samp, 16, axis=0)
    prior_width = prior.high - prior.low
    assert (width / prior_width)[3] < 0.2       # rp_rs sharply measured


def test_sbc_and_tarp(trained):
    sim, prior, grid, model, _ = trained
    theta, tokens, masks = make_training_set(
        sim, prior, grid, n_sims=150,
        noise=NoiseModel(ppm_min=30, ppm_max=120), seed=99)
    ranks = sbc_ranks(model, theta, tokens, masks, n_post=200)
    pvals = sbc_pvalues(ranks, n_post=200)
    # small net + small n: demand "not catastrophically miscalibrated"
    assert np.all(pvals > 1e-4), f"SBC failed: {pvals}"

    alpha, ecp = tarp_coverage(model, prior, theta, tokens, masks, n_post=200)
    assert tarp_max_deviation(alpha, ecp) < 0.20


def test_save_load_population_export(tmp_path, trained):
    sim, prior, grid, model, _ = trained
    model.save(tmp_path / "model")
    from simorgh.models import AmortizedPosterior
    loaded = AmortizedPosterior.load(tmp_path / "model")
    assert loaded.provenance.get("simulator") == "ToySubNeptune"

    theta, tokens, masks = make_training_set(sim, prior, grid, 1, seed=3)
    export = loaded.population_export(tokens[:1], masks[:1], n_samples=500)
    assert export["samples"].shape == (500, prior.dim)
    assert np.all(np.isfinite(export["log_q"]))
    assert np.all(np.isfinite(export["log_interim_prior"]))
    # samples respect the prior box by construction
    assert np.all(export["samples"] >= prior.low)
    assert np.all(export["samples"] <= prior.high)
    # log_q consistent between saved and loaded model
    lq_orig = model.log_prob(export["samples"][:50], tokens[:1], masks[:1])
    np.testing.assert_allclose(lq_orig, export["log_q"][:50], rtol=1e-4)
