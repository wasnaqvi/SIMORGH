import numpy as np
import pytest

from simorgh.data import NoiseModel, make_training_set
from simorgh.priors import subneptune_prior
from simorgh.simulate import ToySubNeptune, g395h_grid
from simorgh.spectra import Spectrum


@pytest.fixture(scope="module")
def sim():
    return ToySubNeptune()


@pytest.fixture(scope="module")
def grid():
    return g395h_grid()


def test_grid_is_constant_r(grid):
    r = grid[:-1] / np.diff(grid)
    assert np.allclose(r, r[0], rtol=1e-6)
    assert 2.8 < grid[0] < 2.9 and 5.0 < grid[-1] < 5.2


def test_depth_scale_physical(sim, grid):
    prior = subneptune_prior()
    theta = prior.sample(64, np.random.default_rng(0))
    depth = sim(theta, grid)
    assert depth.shape == (64, grid.size)
    assert np.all(depth > 0) and np.all(depth < 0.02)


def test_features_scale_with_temperature(sim, grid):
    # hotter atmosphere, all else equal -> larger features
    cold = np.array([400.0, 1.0, 1.0, 0.04, 3.0])
    hot = np.array([1000.0, 1.0, 1.0, 0.04, 3.0])
    amp = lambda th: np.ptp(sim(th, grid))
    assert amp(hot) > amp(cold)


def test_cloud_mutes_features(sim, grid):
    clear = np.array([700.0, 1.0, 1.0, 0.04, 3.0])
    cloudy = np.array([700.0, 1.0, -4.5, 0.04, 3.0])
    assert np.ptp(sim(cloudy, grid)) < 0.5 * np.ptp(sim(clear, grid))


def test_metallicity_co2_marker(sim, grid):
    # CO2 4.3um band grows relative to CH4 3.35um with metallicity
    lo = sim(np.array([700.0, 0.3, 1.0, 0.04, 3.0]), grid)
    hi = sim(np.array([700.0, 2.7, 1.0, 0.04, 3.0]), grid)
    i_co2 = np.argmin(np.abs(grid - 4.30))
    i_ch4 = np.argmin(np.abs(grid - 3.35))
    base_lo, base_hi = lo.min(), hi.min()
    ratio_lo = (lo[i_co2] - base_lo) / (lo[i_ch4] - base_lo)
    ratio_hi = (hi[i_co2] - base_hi) / (hi[i_ch4] - base_hi)
    assert ratio_hi > ratio_lo


def test_training_set_shapes_and_noise(sim, grid):
    prior = subneptune_prior()
    theta, tokens, masks = make_training_set(
        sim, prior, grid, n_sims=32, noise=NoiseModel(p_drop=0.1), seed=1)
    assert theta.shape == (32, prior.dim)
    assert tokens.shape == (32, grid.size, 3)
    assert masks.shape == (32, grid.size)
    assert 0.0 < (~masks).mean() < 0.3          # dropout happened, bounded
    assert np.all(tokens[:, :, 2] > 0)          # errors positive
    # noise consistent with reported errors: pull distribution ~ unit scale
    clean = sim(theta.astype(np.float64), grid)
    pulls = (tokens[:, :, 1] - clean) / tokens[:, :, 2]
    assert 0.7 < pulls[masks].std() < 1.4


def test_spectrum_tokens_roundtrip(grid):
    spec = Spectrum(wavelength=grid, depth=np.full(grid.size, 5e-3),
                    error=np.full(grid.size, 1e-4))
    tokens, mask = spec.to_tokens()
    assert tokens.shape == (grid.size, 3) and mask.all()
