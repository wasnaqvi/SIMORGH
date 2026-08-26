"""Sharded simulation grids for cluster generation.

The expensive product is the CLEAN spectrum set: theta -> noiseless
depth(lambda) from the forward model. Noise is injected at training time
(see `simorgh.data.NoisyGridDataset`), so one grid serves every noise
model and each clean spectrum is seen under many noise realizations —
a free augmentation that is exactly what makes the network amortized
over noise level rather than memorizing one.

Layout on scratch:

    <grid_dir>/
        meta.json          prior box, wavelength grid, simulator provenance
        shard_00000.npz    theta (n, d) float32, depth (n, m) float32
        shard_00001.npz
        ...

Each shard is written by one SLURM array task and is self-validating:
the shard records its own seed and the meta hash it was generated
against, so a grid assembled from mismatched runs is caught on load
rather than silently trained on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..priors import BoxPrior


#: Fields that determine what a shard CONTAINS. Everything here is hashed,
#: so a shard generated against a different value is refused at load.
#: `n_shards` is deliberately excluded: it is bookkeeping, and leaving it
#: out is what allows a grid to be EXTENDED (raise n_shards in meta.json,
#: generate the new shards) without invalidating the existing ones.
#: `base_seed` IS included — two grids that differ only by seed are
#: distinct identities, which is exactly what makes a held-out grid
#: distinguishable from the grid a model trained on.
_HASHED_FIELDS = ("names", "low", "high", "wavelength", "simulator",
                  "base_seed", "sims_per_shard")


def _meta_hash(meta: dict) -> str:
    payload = {k: meta[k] for k in _HASHED_FIELDS}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_grid_meta(grid_dir: str | Path, prior: BoxPrior,
                    wavelength: np.ndarray, simulator_provenance: dict,
                    n_shards: int, sims_per_shard: int,
                    base_seed: int = 0) -> dict:
    """Write meta.json. Call ONCE on the login node before submitting."""
    grid_dir = Path(grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "names": list(prior.names),
        "low": prior.low.tolist(),
        "high": prior.high.tolist(),
        "wavelength": np.asarray(wavelength, dtype=float).tolist(),
        "simulator": simulator_provenance,
        "n_shards": int(n_shards),
        "sims_per_shard": int(sims_per_shard),
        "base_seed": int(base_seed),
    }
    meta["meta_hash"] = _meta_hash(meta)
    (grid_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def read_grid_meta(grid_dir: str | Path) -> dict:
    meta = json.loads((Path(grid_dir) / "meta.json").read_text())
    if meta.get("meta_hash") != _meta_hash(meta):
        raise ValueError(
            f"{grid_dir}/meta.json hash mismatch. One of {_HASHED_FIELDS} was "
            "edited after the grid was defined; the existing shards no longer "
            "match this definition. Define a new grid instead. (n_shards is "
            "not hashed and may be raised to extend a grid.)")
    return meta


def grid_prior(meta: dict) -> BoxPrior:
    return BoxPrior(names=tuple(meta["names"]),
                    low=np.array(meta["low"]),
                    high=np.array(meta["high"]))


def shard_path(grid_dir: str | Path, index: int) -> Path:
    return Path(grid_dir) / f"shard_{index:05d}.npz"


def generate_shard(grid_dir: str | Path, index: int, simulator,
                   overwrite: bool = False) -> Path:
    """Generate one shard. Idempotent: skips an existing valid shard so a
    requeued or partially failed array can be resubmitted wholesale."""
    grid_dir = Path(grid_dir)
    meta = read_grid_meta(grid_dir)
    out = shard_path(grid_dir, index)
    if out.exists() and not overwrite:
        try:
            with np.load(out) as d:
                if str(d["meta_hash"]) == meta["meta_hash"]:
                    return out
        except Exception:
            pass  # corrupt or truncated -> regenerate

    if not 0 <= index < meta["n_shards"]:
        raise IndexError(f"shard {index} outside 0..{meta['n_shards'] - 1}")

    prior = grid_prior(meta)
    wavelength = np.asarray(meta["wavelength"], dtype=np.float64)
    # Independent, reproducible stream per shard.
    rng = np.random.default_rng([meta["base_seed"], index])
    theta = prior.sample(meta["sims_per_shard"], rng)
    depth = simulator(theta, wavelength)

    if not np.all(np.isfinite(depth)):
        bad = int((~np.isfinite(depth)).any(axis=1).sum())
        raise RuntimeError(f"shard {index}: {bad} non-finite spectra")

    # Write through a handle: np.savez_compressed appends '.npz' to a path
    # that lacks it, which would silently defeat the atomic rename.
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, theta=theta.astype(np.float32),
                            depth=depth.astype(np.float32),
                            meta_hash=meta["meta_hash"], index=index)
    tmp.replace(out)          # atomic: no half-written shard is ever loaded
    return out


def load_grid(grid_dir: str | Path, max_shards: int | None = None
              ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load all shards -> (theta, depth, meta). Raises on any missing or
    mismatched shard: training on a partial grid silently changes the
    effective prior, which is exactly the kind of error that produces a
    plausible-looking but wrong posterior."""
    grid_dir = Path(grid_dir)
    meta = read_grid_meta(grid_dir)
    n = meta["n_shards"] if max_shards is None else min(max_shards,
                                                        meta["n_shards"])
    thetas, depths, missing = [], [], []
    for i in range(n):
        p = shard_path(grid_dir, i)
        if not p.exists():
            missing.append(i)
            continue
        with np.load(p) as d:
            if str(d["meta_hash"]) != meta["meta_hash"]:
                raise ValueError(f"shard {i} was generated against a "
                                 "different meta.json — regenerate the grid")
            thetas.append(d["theta"])
            depths.append(d["depth"])
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing shard(s) in {grid_dir}: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
            "Resubmit the array (generation is idempotent) before training.")
    return np.concatenate(thetas), np.concatenate(depths), meta


def grid_status(grid_dir: str | Path) -> dict:
    """Cheap completeness report for the login node."""
    meta = read_grid_meta(grid_dir)
    present = [i for i in range(meta["n_shards"])
               if shard_path(grid_dir, i).exists()]
    return {
        "n_shards": meta["n_shards"],
        "present": len(present),
        "missing": [i for i in range(meta["n_shards"]) if i not in set(present)],
        "n_sims": len(present) * meta["sims_per_shard"],
        "meta_hash": meta["meta_hash"],
    }
