# SIMORGH

Fast atmospheric retrieval for sub-Neptunes, with the calibration tests
that tell you whether the answers can be trusted.

SIMORGH trains a neural network once on a grid of forward models, after
which retrieving a transmission spectrum takes about a second instead of
hours of nested sampling. It is built for JWST NIRSpec/G395H
observations of sub-Neptunes, and designed so the resulting posteriors
can be fed straight into a population-level analysis.

## The problem

A single atmospheric retrieval with nested sampling needs 10^5–10^7
forward model evaluations. That is fine for one planet. It becomes
painful when you want to compare several model assumptions on the same
data, and it becomes impractical for the kind of study the field is
moving towards: dozens of planets reduced uniformly, analysed
identically, and interpreted together as a population.

Simulation-based inference offers a way out. Instead of evaluating the
forward model during inference, you spend the compute up front —
simulate a large grid, train a network to map spectra to posteriors, and
then apply it to as many spectra as you like at negligible cost. The
idea has been demonstrated several times for exoplanet atmospheres:

- **Zingales & Waldmann (2018)** — ExoGAN, the first amortized posterior
  for transmission spectra.
- **Vasist et al. (2023)** — neural posterior estimation with
  petitRADTRANS, seconds per spectrum, with coverage and
  posterior-predictive checks.
- **Ardévol Martínez et al. (2024)** — FlopPITy, sequential neural
  posterior estimation inside ARCiS, including self-consistent models.
- **Gebhard et al. (2025)** — fm4ar, flow matching combined with neural
  importance sampling, which recovers the exact posterior and the
  Bayesian evidence.
- **Lueber et al. (2025)** — FASTER, which showed that a neural ratio
  estimator reproduces both nested-sampling marginals *and* Bayesian
  model probabilities, on mock spectra and on real WASP-39b NIRSpec
  PRISM data, at roughly one second per retrieval against 8–10 GPU-hours
  for the nested-sampling equivalent.

So the speed problem is, in an important sense, solved. Several times
over. **Building another estimator is not the point of this project.**

## What is still missing

Two things, and both are acknowledged in the literature rather than
invented here.

**Whether the posteriors are actually calibrated.** Lueber et al. note
that amortization makes rigorous calibration possible and leave it to
future work; their validation is agreement with MultiNest on one mock
and one real spectrum. The Ariel Machine Learning Working Group white
paper (Yip et al. 2026) is blunter: posterior calibration is
"underexplored," and full simulation-based inference is held back
pending better demonstration of "calibration under distribution shift."
Appendix F3 of that paper sets out exactly what a validation protocol
should contain. Nobody has run it end to end.

**Whether real data resembles the training set.** Fitting the real
WASP-39b spectrum, Lueber et al. find roughly 100 ppm of excess scatter
and conclude the observation may lie outside the distribution the
network was trained on — calling this "a key open issue." This matters
because a network given a spectrum unlike anything it was trained on
will still return a confident-looking posterior. Nothing about it looks
wrong.

There is a third gap that only appears at population scale. Published
demonstrations estimate one-dimensional marginals for a single planet.
A population study needs the joint posterior, one network valid across
many planets, and posterior densities that can be reweighted under a
population model. It also needs the per-planet posteriors to be
genuinely calibrated, because a bias that is tolerable in one retrieval
does not average away when you combine fifty of them — statistical
uncertainty shrinks as 1/sqrt(N) while a systematic offset stays put,
so it eventually dominates.

## What SIMORGH does

- **One network for many planets.** Planet radius ratio and surface
  gravity are parameters the network is trained over, not constants
  fixed per target, so a single trained model applies across a survey.
- **Conditioned on the observation, not just the fluxes.** The network
  reads each channel as a triplet of wavelength, depth and uncertainty,
  and handles missing channels. One model therefore copes with different
  binnings, different noise levels, and masked bad channels — including
  the differences between independent reductions of the same data.
- **Calibration is run, not assumed.** Simulation-based calibration
  (Talts et al. 2018) and TARP joint coverage (Lemos et al. 2023) are
  computed against a held-out grid, and the results are written out with
  the model. A model without them is not treated as a usable result.
- **A check on whether real data is in distribution.** A classifier
  trained to separate simulated from observed spectra gives a single
  number for how far real observations sit from the training set, and
  per-spectrum weights for correcting coverage under that shift.
- **Built for population work from the start.** Every trained model can
  export posterior samples together with their densities and the prior
  used during training, which is what hierarchical reweighting needs
  (Hogg, Myers & Bovy 2010). Marginals alone are not enough, so this is
  wired in from the beginning rather than added later.

## Why sub-Neptunes, and why G395H

The narrow scope is deliberate. A restricted parameter space —
equilibrium temperatures of a few hundred to about 1100 K, high
metallicities, clouds and hazes — keeps the training grid affordable. A
uniformly reduced survey supplies matched nested-sampling posteriors to
validate against, and lets the same planet be pushed through several
controlled reductions to see how much the answer moves. And
sub-Neptunes are themselves the interesting population: mass–metallicity
trends and the chemistry across the radius valley are open questions
that need many planets analysed the same way.

Everything here generalizes to Ariel, which will observe this population
at scale, but nothing depends on Ariel having launched.

## How it works

The forward model grid is generated once and stored. Observational noise
is added during training and redrawn every epoch, so a fixed grid
supplies effectively unlimited training examples and the network learns
to handle a range of noise levels rather than memorizing one. Spectra
are summarized by a small network applied to each channel and pooled in
a way that does not care how many channels there are or what order they
come in. A normalizing flow then maps that summary to the posterior.

The flow works internally in transformed coordinates chosen so that the
posterior can never place probability outside the prior box, and
densities are converted back to physical units before they are handed
out.

```
simorgh/
  spectra.py      spectra with per-channel uncertainties and masks
  priors.py       parameter ranges
  simulate/       forward models (analytic test model, TauREx) and grids
  models/         spectrum summary network and normalizing flow
  diagnostics/    simulation-based calibration, TARP coverage
  population/     hierarchical reweighting
  gate/           in-distribution check for real data
  io/             reading survey spectra from disk
scripts/fir/      cluster scripts for DRAC Fir
```

## Getting started

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

The tests run in about a minute on a laptop and include training a small
network end to end and checking that it recovers injected truths.

```python
from simorgh.simulate import ToySubNeptune, g395h_grid
from simorgh.priors import subneptune_prior
from simorgh.train import train_npe

sim, prior, grid = ToySubNeptune(), subneptune_prior(), g395h_grid()
model, history = train_npe(sim, prior, grid, n_sims=15_000, epochs=20)
model.save("runs/toy_v0")
```

`ToySubNeptune` is an analytic stand-in with the right qualitative
behaviour — feature amplitude scaling with temperature and mean
molecular weight, clouds muting features, a CO2 band that strengthens
with metallicity. It exists so the machinery can be developed and tested
without waiting on a real grid. **It is not physics-grade and no
scientific claim should rest on it.** Production runs use TauREx.

## Running on the cluster

Real grids and training run on DRAC Fir. [FIR.md](FIR.md) has the
runbook: environment setup, generating the grid as an array job,
checking it is complete, training on a GPU, and running the calibration
tests. Grid definitions are fixed and checksummed once created, and
loading refuses a grid that is incomplete or assembled from
inconsistent pieces, since training on part of a grid quietly changes
the prior you are sampling from.

[TARGETS.md](TARGETS.md) records what the results should look like and
what would count as failure.

## Status

Working: the network and training, calibration diagnostics, hierarchical
reweighting, the in-distribution check, the cluster pipeline with
checkpointing and restart, and 27 tests.

Not yet done, and worth knowing before relying on any of it:

- The TauREx interface is written but has not been run against a real
  TauREx installation, so the production grid does not exist yet.
- Correcting coverage under distribution shift is not implemented. The
  in-distribution check currently reports how far real data sits from
  the training set without adjusting for it.
- Importance sampling to recover exact posteriors and evidences, as in
  fm4ar, is planned.
- Selection effects are not modelled. Population results are therefore
  conditional on whichever targets happen to have been observed, which
  is a real limitation for JWST programmes chosen by committee.

## References

Zingales & Waldmann (2018), AJ 156, 268 · Nixon & Madhusudhan (2020),
MNRAS 496, 269 · Vasist et al. (2023), A&A 672, A147 · Aubin et al.
(2023), ECML PKDD · Ardévol Martínez et al. (2024), A&A 681, A44 · Yip
et al. (2024), MNRAS · Gebhard et al. (2025), A&A 693, A42 · Lueber et
al. (2025), ApJL 984, L1 · Yip et al. (2026), Ariel MLWG white paper,
RASTI · Talts et al. (2018), arXiv:1804.06788 · Lemos et al. (2023),
ICML · Hogg, Myers & Bovy (2010), ApJ 725, 2166 · Rackham et al. (2018),
ApJ 853, 122
