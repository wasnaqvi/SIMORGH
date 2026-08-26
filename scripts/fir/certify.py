#!/usr/bin/env python
"""Run the in-distribution calibration battery on a trained model.

    python scripts/fir/certify.py \
        --model ~/scratch/simorgh/runs/taurex_v1_npe/best \
        --grid-dir ~/scratch/simorgh/grids/taurex_v1_holdout \
        --n-sims 1000

Emits certificate.json: SBC rank statistics + KS p-values per parameter,
TARP joint expected-coverage curve and its maximum deviation, with the
pass/fail thresholds applied explicitly.

This is the result no published exoplanet SBI paper currently reports;
FASTER (Lueber et al. 2025, Sec 4.2) cites the procedure as possible and
defers it to future work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simorgh.data import NoiseModel, NoisyGridDataset  # noqa: E402
from simorgh.diagnostics import (sbc_pvalues, sbc_ranks,  # noqa: E402
                                 tarp_coverage, tarp_max_deviation)
from simorgh.models import AmortizedPosterior  # noqa: E402
from simorgh.simulate.grid import grid_prior, load_grid  # noqa: E402

# A calibrated posterior gives uniform SBC ranks; 0.01 is loose enough
# not to fire on ordinary sampling noise at ~1000 simulations.
SBC_P_THRESHOLD = 0.01
TARP_MAX_DEV = 0.03


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="directory saved by model.save()")
    ap.add_argument("--grid-dir", required=True,
                    help="HELD-OUT grid (different base seed from training)")
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--n-post", type=int, default=500)
    ap.add_argument("--ppm-min", type=float, default=30.0)
    ap.add_argument("--ppm-max", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = AmortizedPosterior.load(Path(args.model).expanduser())
    theta, depth, meta = load_grid(Path(args.grid_dir).expanduser())
    prior = grid_prior(meta)
    wavelength = np.asarray(meta["wavelength"], dtype=np.float64)

    if model.provenance.get("meta_hash") == meta["meta_hash"]:
        print("WARNING: certifying on the SAME grid the model trained on. "
              "Coverage will be optimistic — use a held-out grid.")

    n = min(args.n_sims, theta.shape[0])
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(theta.shape[0], size=n, replace=False)
    ds = NoisyGridDataset(theta[idx], depth[idx], wavelength,
                          NoiseModel(ppm_min=args.ppm_min, ppm_max=args.ppm_max),
                          seed=args.seed, fixed=True)
    th = np.stack([ds[i][0] for i in range(n)])
    tokens = np.stack([ds[i][1] for i in range(n)])
    masks = np.stack([ds[i][2] for i in range(n)])

    print(f"SBC over {n} simulations x {args.n_post} posterior draws...")
    ranks = sbc_ranks(model, th, tokens, masks, n_post=args.n_post)
    pvals = sbc_pvalues(ranks, n_post=args.n_post)

    print("TARP joint coverage...")
    alpha, ecp = tarp_coverage(model, prior, th, tokens, masks,
                               n_post=args.n_post, seed=args.seed)
    maxdev = tarp_max_deviation(alpha, ecp)
    noise_floor = float(np.sqrt(1.0 / n))

    sbc_pass = bool(np.all(pvals > SBC_P_THRESHOLD))
    tarp_pass = bool(maxdev < max(TARP_MAX_DEV, noise_floor))

    cert = {
        "model": str(Path(args.model).expanduser().resolve()),
        "model_provenance": model.provenance,
        "holdout_grid": str(Path(args.grid_dir).expanduser().resolve()),
        "holdout_meta_hash": meta["meta_hash"],
        "n_sims": n, "n_post": args.n_post,
        "parameters": list(prior.names),
        "sbc": {"ks_pvalues": pvals.tolist(),
                "threshold": SBC_P_THRESHOLD, "pass": sbc_pass},
        "tarp": {"alpha": alpha.tolist(), "ecp": ecp.tolist(),
                 "max_deviation": maxdev, "noise_floor": noise_floor,
                 "threshold": TARP_MAX_DEV, "pass": tarp_pass},
        "verdict": "PASS" if (sbc_pass and tarp_pass) else "FAIL",
    }
    out = Path(args.out).expanduser() if args.out else (
        Path(args.model).expanduser().parent / "certificate.json")
    out.write_text(json.dumps(cert, indent=2))

    print("\n--- calibration certificate ---")
    for name, p in zip(prior.names, pvals):
        flag = "ok" if p > SBC_P_THRESHOLD else "FAIL"
        print(f"  SBC {name:<14s} p = {p:.4f}   {flag}")
    print(f"  TARP max |ECP-alpha| = {maxdev:.4f}  "
          f"(threshold {max(TARP_MAX_DEV, noise_floor):.4f})  "
          f"{'ok' if tarp_pass else 'FAIL'}")
    print(f"  VERDICT: {cert['verdict']}")
    print(f"  written: {out}")
    return 0 if cert["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
