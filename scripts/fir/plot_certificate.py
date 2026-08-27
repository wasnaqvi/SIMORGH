#!/usr/bin/env python
"""Turn a calibration certificate into figures you can actually read.

    python scripts/fir/plot_certificate.py \
        --run-dir ~/scratch/simorgh/runs/toy_v1

Writes into the run directory:

  certificate.png   training curve, SBC rank histograms, TARP coverage
  posterior.png     one example posterior, with the truth marked

Reading the SBC histograms is the point of them: a flat histogram is a
calibrated posterior, a SLOPE means the posterior is biased in that
parameter, a U (piling up at both ends) means it is too narrow -
overconfident - and an ARCH (piling up in the middle) means it is too
wide. The KS p-value collapses all of that into one number and tells you
none of it.

Runs anywhere - it only needs the certificate and history files. The
posterior figure additionally needs the model and a grid, so it is
skipped if those are not given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display on a compute node
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LABEL = {
    "t_eq": r"$T_{\rm eq}$ [K]",
    "log10_met": r"$\log_{10} Z$ [$\times$ solar]",
    "log10_pcloud": r"$\log_{10} P_{\rm cloud}$ [bar]",
    "rp_rs": r"$R_p/R_\star$",
    "log10_g": r"$\log_{10} g$ [cgs]",
}


def plot_certificate(cert: dict, history: dict | None, out: Path) -> Path:
    names = cert["parameters"]
    d = len(names)
    ncol = max(d, 3)
    fig = plt.figure(figsize=(3.1 * ncol, 8.2))
    gs = fig.add_gridspec(3, ncol, hspace=0.55, wspace=0.32,
                          top=0.90, bottom=0.08, left=0.07, right=0.97)

    # --- row 1: training curve -------------------------------------------
    ax = fig.add_subplot(gs[0, :max(ncol // 2, 1)])
    if history and history.get("val"):
        ep = np.arange(1, len(history["val"]) + 1)
        ax.plot(ep, history["train"], lw=1.2, label="train", color="#4C72B0")
        ax.plot(ep, history["val"], lw=1.6, label="validation", color="#C44E52")
        base = history.get("prior_equivalent_loss")
        if base is not None:
            ax.axhline(base, ls="--", lw=1.2, color="0.35",
                       label=f"prior only ({base:.1f})")
            gain = base - min(history["val"])
            ax.annotate(f"information gain {gain:+.2f} nats\n"
                        f"({gain / 0.693:+.1f} bits per spectrum)",
                        xy=(0.97, 0.93), xycoords="axes fraction",
                        ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF6D8",
                                  ec="0.7", lw=0.6))
        ax.set_xlabel("epoch")
        ax.set_ylabel("negative log density [nats]")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "no history.json", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Training", loc="left", fontsize=11, weight="bold")

    # --- row 1 right: verdict card ---------------------------------------
    ax = fig.add_subplot(gs[0, max(ncol // 2, 1):])
    ax.axis("off")
    verdict = cert["verdict"]
    colour = "#2E7D32" if verdict == "PASS" else "#C62828"
    prov = cert.get("model_provenance", {})
    lines = [
        f"simulations (certification): {cert['n_sims']:,}",
        f"posterior draws each: {cert['n_post']:,}",
        f"training grid: {prov.get('n_sims', '?'):,}"
        if isinstance(prov.get("n_sims"), int) else "training grid: ?",
        f"forward model: {prov.get('simulator', {}).get('engine', '?')}",
        f"holdout hash: {cert['holdout_meta_hash']}",
        f"TARP max |ECP-alpha|: {cert['tarp']['max_deviation']:.3f}"
        f"  (threshold {max(cert['tarp']['threshold'], cert['tarp']['noise_floor']):.3f})",
        f"SBC worst p-value: {min(cert['sbc']['ks_pvalues']):.4f}"
        f"  (threshold {cert['sbc']['threshold']})",
    ]
    ax.text(0.0, 0.98, verdict, fontsize=26, weight="bold", color=colour,
            va="top", transform=ax.transAxes)
    ax.text(0.0, 0.72, "\n".join(lines), fontsize=9, va="top", family="monospace",
            transform=ax.transAxes, linespacing=1.7)

    # --- row 2: SBC rank histograms --------------------------------------
    ranks = np.asarray(cert["sbc"].get("ranks", []))
    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[1, j])
        if ranks.size:
            n_sims = ranks.shape[0]
            nbins = 20
            ax.hist(ranks[:, j], bins=nbins,
                    range=(0, cert["n_post"]), color="#4C72B0",
                    edgecolor="white", linewidth=0.5)
            # 99% band for a uniform histogram (binomial, per bin)
            exp = n_sims / nbins
            sd = np.sqrt(n_sims * (1 / nbins) * (1 - 1 / nbins))
            ax.axhspan(exp - 2.576 * sd, exp + 2.576 * sd,
                       color="0.5", alpha=0.22, lw=0)
            ax.axhline(exp, color="0.35", ls="--", lw=1.0)
        ax.set_title(LABEL.get(name, name), fontsize=10)
        ax.set_xlabel("rank")
        if j == 0:
            ax.set_ylabel("count")
        p = cert["sbc"]["ks_pvalues"][j]
        ok = p > cert["sbc"]["threshold"]
        ax.annotate(f"p = {p:.3f}", xy=(0.5, 0.94), xycoords="axes fraction",
                    ha="center", va="top", fontsize=9,
                    color="#2E7D32" if ok else "#C62828",
                    weight="normal" if ok else "bold")
    fig.text(0.07, 0.635, "Simulation-based calibration — flat is calibrated; "
             "sloped = biased, U = overconfident, arched = too wide",
             fontsize=10, weight="bold")

    # --- row 3: TARP ------------------------------------------------------
    ax = fig.add_subplot(gs[2, :max(ncol // 2, 1)])
    alpha = np.asarray(cert["tarp"]["alpha"])
    ecp = np.asarray(cert["tarp"]["ecp"])
    ax.plot([0, 1], [0, 1], ls="--", color="0.4", lw=1.2, label="calibrated")
    ax.plot(alpha, ecp, lw=2.0, color="#C44E52", label="measured")
    floor = max(cert["tarp"]["threshold"], cert["tarp"]["noise_floor"])
    ax.fill_between(alpha, alpha - floor, alpha + floor, color="0.5",
                    alpha=0.20, lw=0, label=f"tolerance ±{floor:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("credibility level")
    ax.set_ylabel("expected coverage")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("TARP joint coverage", loc="left", fontsize=11, weight="bold")

    ax = fig.add_subplot(gs[2, max(ncol // 2, 1):])
    ax.plot(alpha, ecp - alpha, lw=2.0, color="#C44E52")
    ax.axhline(0, ls="--", color="0.4", lw=1.2)
    ax.axhspan(-floor, floor, color="0.5", alpha=0.20, lw=0)
    ax.set_xlabel("credibility level")
    ax.set_ylabel("coverage − nominal")
    ax.set_title("Residual: above zero = conservative, below = overconfident",
                 loc="left", fontsize=10)

    fig.suptitle("SIMORGH calibration certificate", fontsize=14,
                 weight="bold", x=0.07, ha="left", y=0.97)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_posterior(model, prior, theta_true, tokens, mask, out: Path,
                   n_samples: int = 4000) -> Path:
    """Minimal corner plot: 1D marginals on the diagonal, 2D below."""
    samples = model.sample(n_samples, tokens, mask)
    names = list(prior.names)
    d = len(names)
    fig, axes = plt.subplots(d, d, figsize=(2.0 * d, 2.0 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=40, color="#4C72B0",
                        histtype="stepfilled", alpha=0.85,
                        range=(prior.low[i], prior.high[i]))
                ax.axvline(theta_true[i], color="#C44E52", lw=1.6)
                lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
                ax.set_title(f"{med:.2f}$^{{+{hi-med:.2f}}}_{{-{med-lo:.2f}}}$",
                             fontsize=9)
                ax.set_yticks([])
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=40,
                          range=[[prior.low[j], prior.high[j]],
                                 [prior.low[i], prior.high[i]]],
                          cmap="Blues")
                ax.plot(theta_true[j], theta_true[i], "*", color="#C44E52",
                        ms=11, mec="white", mew=0.6)
            ax.set_xlim(prior.low[j], prior.high[j])
            if i != j:
                ax.set_ylim(prior.low[i], prior.high[i])
            if i == d - 1:
                ax.set_xlabel(LABEL.get(names[j], names[j]), fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LABEL.get(names[i], names[i]), fontsize=9)
            else:
                if i != j:
                    ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle("Example posterior — red marks the truth", fontsize=12,
                 weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="training run directory (holds history.json, best/)")
    ap.add_argument("--certificate", default=None,
                    help="default: <run-dir>/certificate.json")
    ap.add_argument("--grid-dir", default=None,
                    help="grid for the example posterior (default: the "
                         "holdout recorded in the certificate)")
    ap.add_argument("--index", type=int, default=0,
                    help="which simulation to show as the example posterior")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    cert_path = Path(args.certificate).expanduser() if args.certificate \
        else run_dir / "certificate.json"
    if not cert_path.exists():
        raise SystemExit(f"no certificate at {cert_path}. Run certify.py first.")
    cert = json.loads(cert_path.read_text())

    hist_path = run_dir / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else None

    out = plot_certificate(cert, history, run_dir / "certificate.png")
    print(f"wrote {out}")

    grid_dir = Path(args.grid_dir).expanduser() if args.grid_dir \
        else Path(cert["holdout_grid"])
    model_dir = run_dir / "best"
    if grid_dir.exists() and model_dir.exists():
        from simorgh.data import NoiseModel, NoisyGridDataset
        from simorgh.models import AmortizedPosterior
        from simorgh.simulate.grid import grid_prior, load_grid

        model = AmortizedPosterior.load(model_dir)
        theta, depth, meta = load_grid(grid_dir)
        prior = grid_prior(meta)
        wl = np.asarray(meta["wavelength"], dtype=np.float64)
        i = args.index % theta.shape[0]
        ds = NoisyGridDataset(theta[i:i + 1], depth[i:i + 1], wl,
                              NoiseModel(), seed=cert.get("n_sims", 0),
                              fixed=True)
        th, tok, mk = ds[0]
        out2 = plot_posterior(model, prior, th, tok[None], mk[None],
                              run_dir / "posterior.png")
        print(f"wrote {out2}")
    else:
        print("skipped posterior.png (need --grid-dir and <run-dir>/best)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
