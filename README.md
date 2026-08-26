# SIMORGH

Amortized, calibrated atmospheric retrieval for **uniform sub-Neptune
surveys**, with a **population-inference contract**. JWST NIRSpec/G395H
transmission first; Ariel-ready by construction.

SIMORGH performs inference *and* audits it: every trained model ships with
its calibration record (SBC + TARP), an out-of-distribution gate, and the
density-evaluation exports a hierarchical population analysis needs.

## Why this exists (design record, 2026-08-26)

Conclusions from the audited planning record (`AUDIT_BRIEF_FOR_FABLE5.txt`
plus an independent verification pass):

1. **Do not build another standalone estimator.** Six-plus groups have
   shipped amortized SBI retrieval demos (ExoGAN, Vasist+23, FlopPITy,
   Aubin+23, Yip+24a, fm4ar, FASTER). None is a community tool.
2. **The amortization-across-choices problem is the moat.** Training is per
   forward model / prior / noise model. Whoever ships *one* network that is
   amortized over planets, noise levels, binnings and masked channels — with
   trust attached — ships the tool. Nobody has.
3. **The gaps the incumbents named in print** (the reason this scope):
   FASTER (Lueber et al. 2025, ApJL 984 L1) shipped 1D marginals, one
   planet, no coverage test, and calls in-distribution verification of real
   data "a key open issue." The Ariel MLWG white paper (Yip et al. 2026)
   gates SBI adoption on "calibration under distribution shift" and asks
   for one consolidated validated tool (Priority 3, Appendix F3).
4. **The gravitational-wave field already solved the population side**
   (Dingo; Leyde et al. 2024, PRD 109 064056 — NPE hierarchical inference
   with selection effects). SIMORGH ports that playbook to transmission
   spectroscopy; it does not pretend to invent it.
5. **Uniform sub-Neptunes first.** A homogeneous survey (uniform reduction,
   uniform fitting, per-planet nested-sampling posteriors, controlled
   reduction variants) is both the validation set and the first population
   science case (mass–metallicity, radius-valley chemistry). The Patchwork
   survey delivers exactly this.

### GW → exoplanet mapping

| GW (Dingo / Leyde et al.)          | SIMORGH                                        |
|------------------------------------|------------------------------------------------|
| event strain                       | G395H transmission spectrum                    |
| detector PSD conditioning          | per-channel error bars + mask conditioning     |
| extrinsic parameters               | r_p/r_s, log g amortized as parameters (FASTER gap G2) |
| dingo-IS exactness recovery        | importance sampling vs the forward-model likelihood (planned) |
| injection campaigns → p(det)       | selection-function interface (`population/`)   |
| catalog → hyperposterior reweight  | `population_export()` + `hyper_log_likelihood` |

## What is implemented (v0.1)

- **`simorgh.spectra`** — `Spectrum`: variable-length channels
  (wavelength, depth, error) + validity mask, all first-class.
- **`simorgh.priors`** — evaluable `BoxPrior`; `subneptune_prior()` spans
  the survey box (T_eq 300–1100 K, Z 1–1000× solar, gray cloud deck,
  r_p/r_s, log g) so amortization is over planets, not per planet.
- **`simorgh.simulate`** — simulator protocol; analytic `ToySubNeptune`
  with the right degeneracy structure (H/R\*, mu(Z), cloud muting, CO2 ∝
  Z² marker) for end-to-end development; pinned-provenance TauREx adapter
  for the production grid.
- **`simorgh.data`** — noise injection *outside* the simulator:
  per-spectrum noise-level draws, wavelength-dependent shape, channel
  dropout → the network trains amortized over noise levels and masks
  (fm4ar's conditioning idea, extended to the grid itself).
- **`simorgh.models`** — mask-aware DeepSets embedding (grid-length
  agnostic; attention upgrade isolated) + neural spline flow (zuko).
  The flow models logit-box coordinates: posterior support equals the
  prior box by construction, and `log_prob` returns **physical-space
  densities** (Jacobian included).
- **`simorgh.train`** — plain training loop (val selection, cosine decay);
  cluster scale-out deliberately kept out of the library.
- **`simorgh.diagnostics`** — SBC (Talts et al. 2018) and TARP (Lemos et
  al. 2023). In the test suite these caught a genuinely miscalibrated
  undertrained flow on `rp_rs` — they are load-bearing, not decoration.
- **`simorgh.population`** — hierarchical reweighting with the interim
  prior (Hogg, Myers & Bovy 2010), ESS guard (raise inside MCMC, −inf for
  grid scans), Gaussian demo population, selection-function interface.
  **Honest caveat:** JWST target selection is committee-driven; the
  default alpha=1 means hyperposteriors are conditional on the observed
  target list. Ariel's defined tiers make alpha computable later.
- **`simorgh.gate`** — sim-vs-real density-ratio classifier: held-out AUC
  as the shift score, per-spectrum log-ratios as the future weighted-CP
  input (Tibshirani et al. 2019).
- **`simorgh.simulate.grid`** — sharded cluster grids: immutable hashed
  definitions, idempotent shard generation, atomic writes, and a load
  that refuses incomplete or cross-grid shard sets rather than training
  on them.
- **`scripts/fir/`** — the cluster chain (`define_grid`,
  `simulate_shard`, `grid_status`, `train_model`, `certify`) plus
  `generate_jobs.py`, the single source of truth for SLURM resources.
- **`simorgh.io`** — file-boundary loader for survey products.
  **SIMORGH never imports the survey pipeline**; spectra arrive as plain
  tables + JSON sidecars, keeping both codebases independently versioned.

The **population contract**: every trained model must provide
`sample`, physical-space `log_prob`, the stored interim prior, and
`population_export()`. This is enforced by tests from v0.1 so the
hierarchical layer is a consumer, never a retrofit — the structural
mistake to avoid from FASTER's 1D marginals.

## Where it runs

Development and diagnostics on the laptop; **all real compute on DRAC
Fir** (H100s), via `scripts/fir/`. See [FIR.md](FIR.md) for the runbook:
environment setup, the define → simulate → verify → train → certify
chain, and the compute-discipline rules inherited from Patchwork.

The expensive object is the **clean forward-model grid**, generated once
as a SLURM array and reused across every noise model, architecture and
seed. Noise is injected at training time and resampled each epoch, so a
fixed grid yields effectively unlimited training realizations — this is
what makes the network amortized over noise level rather than tied to
one realization.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ -q          # ~1 min CPU; trains a small NPE end to end
```

```python
import numpy as np
from simorgh.simulate import ToySubNeptune, g395h_grid
from simorgh.priors import subneptune_prior
from simorgh.train import train_npe
from simorgh.diagnostics import sbc_ranks, sbc_pvalues

sim, prior, grid = ToySubNeptune(), subneptune_prior(), g395h_grid()
model, hist = train_npe(sim, prior, grid, n_sims=30_000, epochs=20)
model.save("runs/toy_v0")
```

## Roadmap

- **Phase 0 (done):** working core on the toy simulator; calibration
  diagnostics; population contract; OOD gate skeleton.
- **Phase 1:** TauREx production grid on the cluster (~1e5–1e6 sims);
  train the sub-Neptune network; validate planet-by-planet against the
  survey's nested-sampling posteriors; reduction-variant stress test
  (same planet, controlled reduction deltas — the sim-to-real number
  nobody has measured for amortized methods). → Paper 1.
- **Phase 2:** importance-sampling exactness layer; weighted conformal
  prediction on the gate; accept/escalate rule with a guaranteed error
  rate.
- **Phase 3:** hierarchical population results on the survey; Ariel-tier
  simulations; selection function for defined survey tiers.

## Known gaps (kept visible on purpose)

- Toy simulator is structural, not physics-grade; nothing scientific may
  be claimed from it.
- TauREx adapter is untested against a real TauREx install.
- Selection function defaults to alpha=1 (see caveat above).
- Weighted CP not yet implemented; the gate currently scores shift but
  does not yet restore coverage under it.
- Embedding is DeepSets; attention pooling is the known upgrade if
  fine spectral correlations start to matter.
