#!/usr/bin/env python
"""Define a simulation grid (writes meta.json). Run ONCE on the login node
before submitting the array.

    python scripts/fir/define_grid.py \
        --grid-dir ~/scratch/simorgh/grids/taurex_v1 \
        --engine taurex3 \
        --opacity-path ~/scratch/linelists/xsec \
        --cia-path ~/scratch/linelists/cia \
        --n-shards 100 --sims-per-shard 5000

meta.json fixes the prior box, the wavelength grid and the forward-model
provenance for the whole grid. It is hashed; every shard records the hash
it was built against, and training refuses a grid whose shards disagree.
Changing any of it means a NEW grid directory, never an edit in place —
the same uniformity discipline the Patchwork frozen config enforces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simorgh.priors import subneptune_prior  # noqa: E402
from simorgh.simulate.base import g395h_grid  # noqa: E402
from simorgh.simulate.grid import write_grid_meta  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--engine", choices=("toy", "taurex3"), default="taurex3")
    ap.add_argument("--opacity-path", help="required for taurex3")
    ap.add_argument("--cia-path", help="required for taurex3")
    ap.add_argument("--star-radius-rsun", type=float, default=0.45)
    ap.add_argument("--nlayers", type=int, default=100)
    ap.add_argument("--resolution", type=float, default=100.0)
    ap.add_argument("--n-shards", type=int, default=100)
    ap.add_argument("--sims-per-shard", type=int, default=5000)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    grid_dir = Path(args.grid_dir).expanduser()
    if (grid_dir / "meta.json").exists():
        raise SystemExit(
            f"{grid_dir}/meta.json already exists. A grid definition is "
            "immutable: use a new --grid-dir rather than editing this one.")

    if args.engine == "toy":
        provenance = {"engine": "toy", "model": "ToySubNeptune"}
    else:
        if not (args.opacity_path and args.cia_path):
            ap.error("--opacity-path and --cia-path are required for taurex3")
        provenance = {
            "engine": "taurex3",
            "opacity_path": str(Path(args.opacity_path).expanduser()),
            "cia_path": str(Path(args.cia_path).expanduser()),
            "star_radius_rsun": args.star_radius_rsun,
            "nlayers": args.nlayers,
        }
        # Record the adapter's own provenance hash when TauREx is importable
        # so shards can verify they run against the same configuration.
        try:
            from simorgh.simulate.taurex_adapter import TaurexSubNeptune
            sim = TaurexSubNeptune(provenance["opacity_path"],
                                   provenance["cia_path"],
                                   star_radius_rsun=args.star_radius_rsun,
                                   nlayers=args.nlayers)
            provenance["provenance_hash"] = sim.provenance_hash
        except Exception as exc:      # login node may lack the opacity data
            print(f"warning: could not instantiate TauREx here ({exc}); "
                  "provenance_hash omitted, shards will not cross-check it.")

    prior = subneptune_prior()
    wavelength = g395h_grid(resolution=args.resolution)
    meta = write_grid_meta(grid_dir, prior, wavelength, provenance,
                           n_shards=args.n_shards,
                           sims_per_shard=args.sims_per_shard,
                           base_seed=args.base_seed)

    total = args.n_shards * args.sims_per_shard
    print(json.dumps({k: meta[k] for k in
                      ("meta_hash", "n_shards", "sims_per_shard")}, indent=2))
    print(f"grid dir     : {grid_dir}")
    print(f"total sims   : {total:,}")
    print(f"channels     : {wavelength.size}  "
          f"({wavelength[0]:.3f}-{wavelength[-1]:.3f} um, R={args.resolution:g})")
    print(f"parameters   : {', '.join(prior.names)}")

    repo = Path(__file__).resolve().parents[2]
    last = args.n_shards - 1
    if grid_dir.name.endswith("_holdout"):
        # A held-out grid reuses the simulate job that already exists; the
        # certify job is generated against the training grid and finds this
        # one by name, so regenerating jobs here would be wrong.
        print("\nHeld-out grid. Build its shards with the existing simulate "
              "job (no need to\nregenerate jobs), from a shell with no venv "
              "active:\n"
              f"  GRID_DIR={grid_dir} sbatch --export=ALL "
              f"--array=0-{last} ~/simorgh/jobs/simulate.sbatch")
    else:
        run_dir = Path("~/scratch/simorgh/runs").expanduser() / grid_dir.name
        print(f"\nNext:\n  python {repo}/scripts/fir/generate_jobs.py \\\n"
              f"      --grid-dir {grid_dir} \\\n"
              f"      --run-dir {run_dir} \\\n"
              f"      --job-dir ~/simorgh/jobs --repo {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
