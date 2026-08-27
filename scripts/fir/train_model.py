#!/usr/bin/env python
"""Train the amortized posterior from a completed grid. One GPU job.

    python scripts/fir/train_model.py \
        --grid-dir ~/scratch/simorgh/grids/taurex_v1 \
        --out-dir  ~/scratch/simorgh/runs/taurex_v1_npe

Resumes automatically from out-dir/checkpoint.pt, so a requeued or
walltime-killed job continues from its last completed epoch. Refuses to
start on an incomplete grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from simorgh.data import NoiseModel  # noqa: E402
from simorgh.simulate.grid import grid_status  # noqa: E402
from simorgh.train import pick_device, train_npe_grid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--context-dim", type=int, default=128)
    ap.add_argument("--transforms", type=int, default=8)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--ppm-min", type=float, default=30.0)
    ap.add_argument("--ppm-max", type=float, default=250.0)
    ap.add_argument("--p-drop", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    grid_dir = Path(args.grid_dir).expanduser()
    status = grid_status(grid_dir)
    if status["missing"]:
        raise SystemExit(
            f"grid incomplete: {status['present']}/{status['n_shards']} shards "
            f"present, missing {status['missing'][:10]}"
            f"{'...' if len(status['missing']) > 10 else ''}.\n"
            "Resubmit the simulation array (generation is idempotent).")

    dev = pick_device(args.device)
    print(f"device       : {dev}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if dev.type == "cuda" else ""))
    print(f"grid         : {grid_dir}")
    print(f"sims         : {status['n_sims']:,}   hash {status['meta_hash']}")
    if dev.type != "cuda":
        print("WARNING: training on CPU. On Fir this should be a GPU job "
              "(--gpus-per-node=1).")

    model, history = train_npe_grid(
        grid_dir, Path(args.out_dir).expanduser(),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        noise=NoiseModel(ppm_min=args.ppm_min, ppm_max=args.ppm_max,
                         p_drop=args.p_drop),
        seed=args.seed, device=args.device, num_workers=args.num_workers,
        context_dim=args.context_dim, transforms=args.transforms,
        hidden=tuple(args.hidden), resume=not args.no_resume)

    gain = history["information_gain_nats"]
    baseline = history["prior_equivalent_loss"]
    print(f"\nbest val loss: {min(history['val']):.3f}")
    print(f"prior-only   : {baseline:.3f}  (a network ignoring the spectrum)")
    print(f"info gain    : {gain:.3f} nats = {gain / 0.693:.1f} bits per spectrum")
    if gain < 0.5:
        print("  *** The network learned almost nothing from the spectra. "
              "Check the\n      grid, the noise level, and that the "
              "simulator varies with theta.")
    print(f"weights      : {Path(args.out_dir).expanduser() / 'best'}")
    print("\nNext: scripts/fir/certify.py to run SBC + TARP on this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
