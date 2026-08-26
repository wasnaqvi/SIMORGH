# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Overview

SIMORGH is an **amortized atmospheric retrieval engine for uniform
sub-Neptune surveys, with the calibration tests that show whether its
posteriors can be trusted**. JWST NIRSpec/G395H transmission first;
Ariel-ready by construction.

It performs inference *and* checks it. A trained model is not treated as
usable until its calibration results exist alongside it. That is the
whole design thesis — see [Invariants](#invariants-do-not-break-these).

Development happens on the laptop; **all real compute runs on DRAC Fir**
(H100 GPUs). See [FIR.md](FIR.md) for the runbook.

## Scope, and why it is this narrow

**Uniform sub-Neptunes, G395H transmission.** The narrow scope is
deliberate and load-bearing:

- The parameter space is tractable (T_eq 300–1100 K, high metallicity,
  clouds/haze), so a training grid is affordable.
- A homogeneous survey supplies uniform reductions, per-planet
  nested-sampling posteriors for validation, and controlled reduction
  variants — the validation data that makes calibration claims testable
  rather than asserted.
- Sub-Neptunes are themselves the population science case
  (mass–metallicity, radius-valley chemistry), so the first targets and
  the eventual science are the same planets.

**Do not widen the scope opportunistically.** Adding emission, direct
imaging, or hot Jupiters multiplies the grid cost and dilutes the
validation set without adding a result.

## What is new here (state accurately; do not overclaim)

Published record, as of the design date:

- Amortized SBI retrieval has been demonstrated repeatedly (ExoGAN,
  Vasist et al. 2023, FlopPITy/Ardévol Martínez et al. 2024, Aubin et
  al. 2023, Yip et al. 2024a, fm4ar/Gebhard et al. 2025, FASTER/Lueber
  et al. 2025). **Building a seventh estimator has near-zero marginal
  value** and is not what this repo is.
- FASTER (ApJL 984 L1) matched nested sampling on **one-dimensional
  marginals**, for **one planet**, validating on one mock and one real
  spectrum. Its Sec. 4.2 defers coverage calibration to future work and
  calls verifying that observed data are in-distribution "a key open
  issue."
- The Ariel MLWG white paper (Yip et al. 2026) gates full SBI adoption
  on "calibration under distribution shift," asks for one consolidated
  validated tool (Priority 3), and specifies a validation protocol
  (Appendix F3).
- The hierarchical machinery is **not new**. Posterior reweighting under
  an interim prior is standard practice (Hogg, Myers & Bovy 2010, ApJ
  725, 2166) and has been applied at scale in other areas of astronomy.
  Cite it properly; never present it as invented here. What is unclaimed
  is applying it to amortized exoplanet retrieval posteriors that have
  been calibrated first.

SIMORGH's contribution is therefore: joint (not 1D) posteriors, one
network across a planet population, coverage actually performed at
scale, an OOD gate whose score is shown to predict posterior error, and
reduction-variant robustness measured on real survey data.

## Architecture

```
simorgh/
  spectra.py            Spectrum: variable-length channels + validity mask
  priors.py             BoxPrior (evaluable), subneptune_prior()
  data.py               noise injection, NoisyGridDataset (noise per epoch)
  train.py              train_npe (in-memory) / train_npe_grid (cluster)
  simulate/
    base.py             Simulator protocol, g395h_grid()
    toy.py              analytic ToySubNeptune (structure, NOT physics)
    taurex_adapter.py   TauREx 3, pinned provenance (production)
    grid.py             sharded cluster grids: hashing, idempotence, guards
  models/
    embedding.py        mask-aware DeepSets over (wavelength, depth, error)
    npe.py              AmortizedPosterior: flow + population exports
  diagnostics/
    sbc.py              simulation-based calibration ranks + KS p-values
    tarp.py             TARP joint expected coverage
  population/
    reweight.py         hierarchical reweighting w/ interim prior, ESS guard
  gate/
    ood.py              sim-vs-real density-ratio classifier (held-out AUC)
  io/
    patchwork.py        file-boundary loader for survey products

scripts/fir/            cluster chain; generate_jobs.py sizes all SLURM jobs
```

### Design notes worth knowing before editing

- **The flow works in logit-box coordinates.** Posterior support equals
  the prior box by construction, and `log_prob` returns **physical-space**
  densities with the Jacobian included. If you change the parameter
  transform, you must change `_log_jac_z_theta` with it or every density
  the population layer consumes becomes silently wrong.
- **The embedding is permutation-invariant and length-agnostic**
  (DeepSets over channel tokens). That is what lets one network serve
  different binnings, reduction variants and masked channels. Attention
  pooling is the known upgrade path; it is isolated in `embedding.py` and
  nothing downstream changes.
- **Noise is not the simulator's job.** The simulator returns clean
  depths; noise is injected in `data.py` at training time and resampled
  every epoch. One grid therefore serves every noise model, and the
  network amortizes over noise level instead of memorizing one
  realization.

## Invariants (do not break these)

1. **Population exports.** Every trained model must provide
   `sample`, physical-space `log_prob`, the stored interim prior, and
   `population_export()`. Hierarchical reweighting divides by the interim
   prior; samples alone cannot feed a population loop. This is enforced
   by tests from v0.1 so the hierarchical layer stays a consumer, never
   a retrofit.
2. **Grid definitions are immutable.** `meta.json` fixes the prior box,
   wavelength grid, simulator provenance, `base_seed` and
   `sims_per_shard`, and is hashed. Changing any of it means a NEW grid
   directory — never an edit in place. `n_shards` is deliberately
   *unhashed* so a grid can be extended when more simulations are needed.
3. **Never train on an incomplete or mixed grid.** `load_grid` raises on
   missing shards and on shards carrying a different `meta_hash`. A
   partial grid silently changes the effective prior.
4. **SIMORGH never imports the reduction pipeline.** Survey products
   arrive as plain tables plus JSON sidecars through `simorgh.io`. Keep
   the boundary at files so the two codebases version independently.
5. **A model without a certificate is not a result.** `certify.py`
   against a **held-out** grid (different `--base-seed`), before any
   scientific claim.

## Compute discipline (DRAC Fir)

Full detail in [FIR.md](FIR.md). The rules that matter:

- **Never generate a grid or train on a login node.** Login node is for
  `define_grid`, `grid_status` and `generate_jobs`; everything else is
  SLURM.
- **Submit from a shell with no venv active** — SLURM exports the
  submitting environment and `module purge` cannot undo an activation.
- **Simulation is 1 core per array task.** The forward model is
  single-threaded; a larger request only lengthens the queue wait.
- **`scripts/fir/generate_jobs.py` is the single source of truth for
  walltime and memory.** Never hand-write an sbatch.
- Training checkpoints every epoch and resumes, so prefer a short
  walltime (starts sooner) plus requeue over a long one.
- SIMORGH has its **own** virtualenv (`~/simorgh/simorgh-env`), separate
  from any reduction environment. `zuko` is not in the Alliance
  wheelhouse and installs from PyPI on the login node; **compute nodes
  have no internet**, so install everything before submitting.

## VERIFY THE STATISTICS, NOT THE EXIT CODE

A training job that exits 0 has proven only that a loss went down. The
failure modes that matter all exit 0:

- **The network learned the prior and ignored the spectrum.** Symptom:
  validation loss plateaus almost immediately; posterior widths equal
  prior widths. This is the amortized analogue of a retrieval "returning
  the prior."
- **Miscalibration.** Symptom: nothing visible at all — posteriors look
  fine and are wrong. Only SBC/TARP catch it. During development, SBC
  caught a genuinely miscalibrated `rp_rs` marginal (p ≈ 8e-6) that eye
  inspection and TARP both missed.
- **Trained on the wrong grid.** Check `provenance.meta_hash` on the
  saved model against the grid you believe you used.
- **Certified on the training grid.** `certify.py` warns; heed it. A
  certificate computed in-distribution on training data is meaningless.

Per-planet miscalibration is *not* a cosmetic problem: bias does not
average out under hierarchical aggregation, so it compounds across the
population rather than shrinking as 1/sqrt(N). That is why the
calibration layer gates the population layer.

## Testing

```bash
python -m pytest tests/ -q      # ~1 min on CPU; trains a small NPE for real
```

27 tests, no network and no cluster required. They cover the simulator's
physical structure (feature amplitude with temperature, cloud muting,
the metallicity CO2 marker), noise-injection statistics, an end-to-end
NPE that must concentrate on truth and pass SBC/TARP, save/load
round-trips **including non-default architectures**, grid sharding and
its provenance guards, training resume from checkpoint, population
hyperparameter recovery, the ESS guard, and the OOD gate's behaviour on
in-distribution versus shifted data.

The statistical assertions are deliberately loose (small nets, few
sims). Tighten them only alongside a bigger training budget, or they
will flake.

## Common workflows

**Local development loop** — use the toy simulator; it has the right
degeneracy structure for exercising the whole pipeline but is **not
physics-grade**. Never make a scientific claim from it.

```python
from simorgh.simulate import ToySubNeptune, g395h_grid
from simorgh.priors import subneptune_prior
from simorgh.train import train_npe

sim, prior, grid = ToySubNeptune(), subneptune_prior(), g395h_grid()
model, history = train_npe(sim, prior, grid, n_sims=15_000, epochs=20)
```

**Production run on Fir** — `define_grid` → `sbatch --array` simulate →
`grid_status` (must be COMPLETE) → `sbatch` train → `sbatch` certify
against a held-out grid. See [FIR.md](FIR.md).

**Adding a forward model** — implement the `Simulator` protocol
(`theta, wavelength -> depth`), record provenance, and add a branch in
`scripts/fir/simulate_shard.py::build_simulator`. Do not add noise
inside the simulator.

## Known gaps

Kept visible on purpose; do not paper over them in docs or commit
messages.

- The toy simulator is structural, not physics-grade.
- The TauREx adapter has not been run against a real TauREx install.
- The selection function defaults to alpha = 1, so hyperposteriors are
  conditional on the observed target list. JWST target selection is
  committee-driven and not easily simulable; Ariel's defined survey
  tiers will make alpha computable.
- Weighted conformal prediction is not implemented: the gate currently
  *scores* shift but does not restore coverage under it.
- Importance-sampling exactness recovery (the fm4ar-style correction) is
  planned, not built.

## Do not

- Build another standalone estimator, or widen scope to emission/direct
  imaging "while we're here."
- Add noise inside a simulator, or bake a fixed noise level into a grid.
- Edit a grid's `meta.json` in place (except raising `n_shards`).
- Hand-write sbatch files.
- Import the reduction pipeline from SIMORGH.
- Claim the hierarchical machinery as novel — it is standard practice from the
  wider statistics literature and must be cited as such.
