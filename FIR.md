# SIMORGH on DRAC Fir

Everything heavy runs on Fir: `fir.alliancecan.ca`, 165 k cores, 640×
H100 (80 GB), 7-day max walltime. The Mac is for writing code and reading
certificates.

**Compute discipline** (same rules as Patchwork, same reasons):

- Never generate a grid or train on a **login node**. Grid definition,
  status checks and job generation are login-node work; everything else
  is SLURM.
- **Submit from a shell with no venv active.** SLURM exports the
  submitting environment and `module purge` cannot undo an activation.
- Simulation is **1 core per array task** — the forward model is
  single-threaded, and a larger request only lengthens the queue wait.
- Never point two jobs at one output directory.
- `scripts/fir/generate_jobs.py` is the **single source of truth for
  walltime and memory**. Never hand-write an sbatch.

## One-time setup

SIMORGH gets its **own** virtualenv, deliberately separate from
`aster-env`: it does not import the reduction pipeline, and coupling the
two would tie together release cycles that have no reason to move
together.

```bash
mkdir -p ~/simorgh && cd ~/simorgh
git clone <your-remote> SIMORGH

module load StdEnv/2023 python/3.13.2 cuda/12.6
virtualenv --no-download ~/simorgh/simorgh-env
source ~/simorgh/simorgh-env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index numpy scipy torch          # Alliance wheelhouse
pip install zuko                                  # login node has internet
pip install -e ~/simorgh/SIMORGH --no-deps
```

`zuko` is pure Python and is not in the Alliance wheelhouse; it installs
from PyPI on the login node. **Compute nodes have no internet**, so every
dependency must be installed before submitting.

Verify, on the login node:

```bash
python -c "import torch, zuko, simorgh; print(torch.__version__, zuko.__version__, simorgh.__version__)"
```

For TauREx grids, the opacity data must also live on scratch and the
paths recorded in the grid definition (see below).

## The chain

```
define_grid  ->  simulate (array)  ->  grid_status  ->  train (GPU)  ->  certify
   login            compute            login            compute        compute
```

### 1. Define the grid (login node, seconds)

```bash
python scripts/fir/define_grid.py \
    --grid-dir ~/scratch/simorgh/grids/taurex_v1 \
    --engine taurex3 \
    --opacity-path ~/scratch/linelists/xsec \
    --cia-path ~/scratch/linelists/cia \
    --n-shards 100 --sims-per-shard 5000
```

Writes `meta.json`: the prior box, wavelength grid, forward-model
provenance, and a hash over all of it. **A grid definition is
immutable.** Changing the prior, the wavelength grid or the simulator
means a new `--grid-dir`, never an edit in place — the same discipline
as Patchwork's frozen exoTEDRF config, for the same reason: a network
trained on shards from two different definitions has no consistent
interpretation. The hash is checked on every load, and each shard records
the hash it was generated against.

`n_shards` is deliberately *not* hashed, so a grid can be **extended**
(raise `n_shards` in `meta.json`, generate the new shards) when it turns
out more simulations are needed.

### 2. Generate the jobs (login node)

```bash
python scripts/fir/generate_jobs.py \
    --grid-dir ~/scratch/simorgh/grids/taurex_v1 \
    --run-dir  ~/scratch/simorgh/runs/taurex_v1_npe \
    --job-dir  ~/simorgh/jobs
```

Prints the submission plan and writes `simulate.sbatch`, `train.sbatch`,
`certify.sbatch` sized from the table at the top of the generator.

### 3. Simulate (array job)

```bash
sbatch --array=0-99 ~/simorgh/jobs/simulate.sbatch
```

Shard generation is **idempotent**: a shard that already exists with a
matching hash is skipped. After a partial failure, resubmit the whole
array — only the missing shards cost anything.

### 4. Verify completeness BEFORE training

```bash
python scripts/fir/grid_status.py --grid-dir ~/scratch/simorgh/grids/taurex_v1
```

Exits non-zero and prints a ready-to-paste `--array=` range for the
missing shards. **Do not skip this.** `load_grid` refuses an incomplete
grid rather than training on it, because a partial grid silently changes
the effective prior — the class of error that yields a plausible-looking
network with no valid meaning.

### 5. Train (one H100)

```bash
sbatch ~/simorgh/jobs/train.sbatch
```

Checkpoints every epoch to `<run-dir>/checkpoint.pt` and resumes
automatically. A walltime kill costs one epoch, not the run — so prefer
a short walltime (which starts sooner) and requeue, over a long one.

Best weights land in `<run-dir>/best`, final in `<run-dir>/final`, loss
curves in `history.json`. The saved model records the grid hash, the
simulator provenance, the noise model and the seed.

### 6. Certify (against a HELD-OUT grid)

Define a second grid with the same box but a different `--base-seed`, so
it is a distinct identity:

```bash
python scripts/fir/define_grid.py \
    --grid-dir ~/scratch/simorgh/grids/taurex_v1_holdout \
    --engine taurex3 --opacity-path ... --cia-path ... \
    --n-shards 10 --sims-per-shard 5000 --base-seed 999

sbatch --array=0-9 ~/simorgh/jobs/simulate.sbatch    # after regenerating jobs
sbatch ~/simorgh/jobs/certify.sbatch
```

Writes `certificate.json` and exits non-zero on FAIL, so a failed
calibration is visible in `sacct` rather than buried in a log. `certify`
warns if the certification grid hash matches the training grid.

## When a job fails

Two different questions, with different answers. Ask both.

### Which SLURM tasks failed

```bash
# everything that did not end cleanly, since a given date
sacct -X -s FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL --starttime=today \
      --format=JobID%20,JobName%18,State%16,Elapsed,ExitCode
```

`-X` collapses the `.batch`/`.extern` steps so each job is one row. Drop
it when you want `MaxRSS`, which only appears on the step rows:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ReqMem,ExitCode
seff <JOBID>            # same thing, human-readable, incl. memory efficiency
```

What the states mean here:

| State | Cause | Fix |
|---|---|---|
| `TIMEOUT` | hit `--time` | training resumes from its checkpoint — just resubmit. For simulation, raise the walltime in `generate_jobs.py` |
| `OUT_OF_MEMORY` | exceeded `--mem` | check `MaxRSS`, raise the value in `generate_jobs.py`, never in the sbatch by hand |
| `NODE_FAIL` | hardware glitch | rare; resubmit |
| `CANCELLED` | you or the scheduler killed it | check whether the allocation is depleted |

For an array, the failed task indices in a form you can paste straight
back into `--array`:

```bash
sacct -X -n -j <ARRAY_JOB_ID> -s FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL \
      --format=JobID | sed 's/.*_//' | paste -sd,
```

Python tracebacks land in the job logs, not in `sacct`:

```bash
grep -l -iE "error|traceback" ~/simorgh/jobs/logs/*.out
```

### Which shards are actually missing

```bash
python scripts/fir/grid_status.py --grid-dir ~/scratch/simorgh/grids/taurex_v1
```

**This is the authoritative one for deciding what to resubmit**, and it
does not always agree with `sacct`. A task can fail *after* writing a
complete shard, in which case there is nothing to redo; a task can also
report `COMPLETED` having skipped a shard that already existed. The
scheduler knows about processes, not about products.

`grid_status.py` exits non-zero and prints a ready-made `--array=` range
covering exactly the missing shards. Since generation is idempotent you
can also just resubmit the whole array — completed shards are skipped in
milliseconds — but the narrow range is kinder to the queue.

## VERIFY THE STATISTICS, NOT THE EXIT CODE

A training job that exits 0 has proven only that the loss went down.
Before believing anything scientific:

- **`certificate.json` verdict.** SBC KS p-values per parameter and TARP
  joint max deviation, against the thresholds in `certify.py`. A model that
  fails here is not usable for population work no matter how good the
  posteriors look by eye.
- **Loss curve shape** in `history.json`. A validation loss that plateaus
  immediately usually means the network learned the prior and ignored the
  spectrum — the amortized analogue of Patchwork's "the fit returns the
  prior" failure.
- **Posterior width against the prior width.** If every posterior is
  prior-width, the spectrum is not informing the network.
- **`provenance.meta_hash`** on the saved model matches the grid you
  think you trained on.

## Cost estimates (update as real numbers land)

| Stage | Resources | Estimate |
|---|---|---|
| Simulate 5e5 TauREx sims | 100 tasks × 1 core, 3 h cap | 20–85 min per task |
| Train, 5e5 grid, 100 epochs | 1 × H100, 8 cores, 64 G, 6 h cap | 1–4 h |
| Certify, 1000 sims × 500 draws | 8 cores, 1 h cap | minutes |

The grid dominates and is generated **once**; retraining with a different
noise model, architecture or seed reuses it. That is the practical form
of the amortization argument: the expensive object is the clean forward
model set, and it is a fixed cost.
