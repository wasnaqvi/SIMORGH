"""Grid sharding, provenance guards, on-the-fly noise, and training resume.

These guard the cluster failure modes that are expensive to discover
late: a partially generated grid trained on silently, shards mixed
across two grid definitions, and a walltime-killed job restarting from
epoch zero.
"""

import json

import numpy as np
import pytest
import torch

from simorgh.data import NoiseModel, NoisyGridDataset
from simorgh.priors import subneptune_prior
from simorgh.simulate import ToySubNeptune, g395h_grid
from simorgh.simulate.grid import (generate_shard, grid_status, load_grid,
                                   read_grid_meta, shard_path,
                                   write_grid_meta)
from simorgh.train import train_npe_grid


@pytest.fixture
def small_grid(tmp_path):
    prior, wl = subneptune_prior(), g395h_grid()
    write_grid_meta(tmp_path, prior, wl, {"engine": "toy"},
                    n_shards=3, sims_per_shard=200, base_seed=7)
    return tmp_path, prior, wl


def test_shards_generate_and_load(small_grid):
    grid_dir, prior, wl = small_grid
    sim = ToySubNeptune()
    for i in range(3):
        generate_shard(grid_dir, i, sim)
    theta, depth, meta = load_grid(grid_dir)
    assert theta.shape == (600, prior.dim)
    assert depth.shape == (600, wl.size)
    assert np.all(np.isfinite(depth))
    # shards are independent draws, not repeats of the same stream
    assert not np.allclose(theta[:200], theta[200:400])


def test_generation_is_idempotent(small_grid):
    grid_dir, *_ = small_grid
    sim = ToySubNeptune()
    generate_shard(grid_dir, 0, sim)
    first = shard_path(grid_dir, 0).read_bytes()
    generate_shard(grid_dir, 0, sim)          # second call skips
    assert shard_path(grid_dir, 0).read_bytes() == first


def test_incomplete_grid_refuses_to_load(small_grid):
    grid_dir, *_ = small_grid
    sim = ToySubNeptune()
    generate_shard(grid_dir, 0, sim)
    generate_shard(grid_dir, 2, sim)
    st = grid_status(grid_dir)
    assert st["present"] == 2 and st["missing"] == [1]
    with pytest.raises(FileNotFoundError, match="missing shard"):
        load_grid(grid_dir)


def test_meta_hash_guards_hand_edits(small_grid):
    grid_dir, *_ = small_grid
    meta = json.loads((grid_dir / "meta.json").read_text())
    meta["low"][0] = 250.0                     # widen the box by hand
    (grid_dir / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_grid_meta(grid_dir)


def test_grid_can_be_extended(small_grid):
    """n_shards is unhashed on purpose: needing more sims is routine and
    must not invalidate shards already generated."""
    grid_dir, *_ = small_grid
    sim = ToySubNeptune()
    for i in range(3):
        generate_shard(grid_dir, i, sim)
    before = shard_path(grid_dir, 0).read_bytes()

    meta = json.loads((grid_dir / "meta.json").read_text())
    meta["n_shards"] = 5
    (grid_dir / "meta.json").write_text(json.dumps(meta))
    read_grid_meta(grid_dir)                   # still valid
    for i in (3, 4):
        generate_shard(grid_dir, i, sim)
    theta, _, _ = load_grid(grid_dir)
    assert theta.shape[0] == 1000
    assert shard_path(grid_dir, 0).read_bytes() == before


def test_holdout_grid_has_distinct_hash(small_grid, tmp_path):
    """certify.py warns when the certification grid is the training grid,
    by comparing hashes — so a differently seeded holdout must not collide."""
    grid_dir, prior, wl = small_grid
    train_meta = read_grid_meta(grid_dir)
    holdout = write_grid_meta(tmp_path / "holdout", prior, wl,
                              {"engine": "toy"}, n_shards=3,
                              sims_per_shard=200, base_seed=999)
    assert holdout["meta_hash"] != train_meta["meta_hash"]


def test_shard_from_other_grid_rejected(small_grid, tmp_path):
    grid_dir, prior, wl = small_grid
    sim = ToySubNeptune()
    for i in range(3):
        generate_shard(grid_dir, i, sim)

    other = tmp_path / "other"
    write_grid_meta(other, prior, wl, {"engine": "toy"},
                    n_shards=3, sims_per_shard=200, base_seed=99)
    for i in range(3):
        generate_shard(other, i, sim)
    # splice one shard from the other grid in
    shard_path(grid_dir, 1).write_bytes(shard_path(other, 1).read_bytes())
    with pytest.raises(ValueError, match="different meta.json"):
        load_grid(grid_dir)


def test_noise_resampled_across_epochs_but_fixed_when_asked():
    prior, wl = subneptune_prior(), g395h_grid()
    sim = ToySubNeptune()
    rng = np.random.default_rng(0)
    theta = prior.sample(4, rng)
    depth = sim(theta, wl)

    ds = NoisyGridDataset(theta, depth, wl, NoiseModel(), seed=0)
    ds.set_epoch(0)
    a = ds[0][1].copy()
    ds.set_epoch(1)
    b = ds[0][1].copy()
    assert not np.allclose(a[:, 1], b[:, 1])     # fresh noise each epoch
    np.testing.assert_allclose(a[:, 0], b[:, 0])  # same wavelengths

    fixed = NoisyGridDataset(theta, depth, wl, NoiseModel(), seed=0, fixed=True)
    fixed.set_epoch(0)
    c = fixed[0][1].copy()
    fixed.set_epoch(5)
    np.testing.assert_allclose(c, fixed[0][1])   # validation set is stable


def test_training_resumes_from_checkpoint(small_grid, tmp_path):
    grid_dir, *_ = small_grid
    sim = ToySubNeptune()
    for i in range(3):
        generate_shard(grid_dir, i, sim)

    run = tmp_path / "run"
    torch.manual_seed(0)
    _, hist1 = train_npe_grid(grid_dir, run, epochs=2, batch_size=64,
                              num_workers=0, context_dim=32, transforms=2,
                              hidden=(32, 32), verbose=False)
    assert len(hist1["train"]) == 2
    assert (run / "checkpoint.pt").exists()
    assert (run / "best" / "weights.pt").exists()

    # a "requeued" job continuing to epoch 4 keeps the first two epochs
    _, hist2 = train_npe_grid(grid_dir, run, epochs=4, batch_size=64,
                              num_workers=0, context_dim=32, transforms=2,
                              hidden=(32, 32), verbose=False)
    assert len(hist2["train"]) == 4
    assert hist2["train"][:2] == hist1["train"]


def test_nondefault_architecture_round_trips(small_grid, tmp_path):
    """save/load must restore the ARCHITECTURE, not just the weights: the
    training script exposes context-dim/transforms/hidden as flags, and a
    checkpoint that silently rebuilds at default shape fails to load."""
    from simorgh.models import AmortizedPosterior

    grid_dir, prior, wl = small_grid
    sim = ToySubNeptune()
    for i in range(3):
        generate_shard(grid_dir, i, sim)
    run = tmp_path / "arch"
    model, _ = train_npe_grid(grid_dir, run, epochs=1, batch_size=64,
                              num_workers=0, context_dim=32, transforms=2,
                              hidden=(32, 32), verbose=False)

    loaded = AmortizedPosterior.load(run / "best")
    assert loaded.architecture == {"context_dim": 32, "transforms": 2,
                                   "hidden": [32, 32]}
    theta, depth, meta = load_grid(grid_dir)
    ds = NoisyGridDataset(theta[:1], depth[:1], wl, NoiseModel(),
                          seed=0, fixed=True)
    _, tok, mk = ds[0]
    s = loaded.sample(64, tok[None], mk[None])
    assert s.shape == (64, prior.dim)


def test_saved_model_carries_grid_provenance(small_grid, tmp_path):
    grid_dir, *_ = small_grid
    sim = ToySubNeptune()
    for i in range(3):
        generate_shard(grid_dir, i, sim)
    run = tmp_path / "run2"
    model, _ = train_npe_grid(grid_dir, run, epochs=1, batch_size=64,
                              num_workers=0, context_dim=32, transforms=2,
                              hidden=(32, 32), verbose=False)
    meta = read_grid_meta(grid_dir)
    assert model.provenance["meta_hash"] == meta["meta_hash"]
    assert model.provenance["n_sims"] == 600
    assert model.provenance["simulator"]["engine"] == "toy"
