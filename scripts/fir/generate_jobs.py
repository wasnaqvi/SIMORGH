#!/usr/bin/env python
"""Write sized sbatch scripts for SIMORGH on DRAC Fir, and print the
submission plan.

SINGLE SOURCE OF TRUTH for walltime, memory and resource requests — never
hand-write an sbatch for this project. (Same discipline as Patchwork's
generate_survey_jobs.py, and for the same reason: hand-edited resources
drift, and a drifted request is discovered only after a job dies.)

    python scripts/fir/generate_jobs.py \
        --grid-dir ~/scratch/simorgh/grids/taurex_v1 \
        --run-dir  ~/scratch/simorgh/runs/taurex_v1_npe \
        --job-dir  ~/simorgh/jobs

Sizing notes (update these as real timings land — they are the honest
current estimates, not measurements):

  SIMULATE  TauREx transmission forward model, ~0.2-1 s per spectrum on
            one core. At 5000 sims/shard that is 20-85 min; the 3 h
            walltime is ~2x the pessimistic end. CPU only, 1 core per
            task: the model is single-threaded and a bigger ask only
            lengthens the queue wait. Fir bills by core-hour, so a
            100-task array at 1 core is cheap and starts fast.

  TRAIN     One H100. A 5e5-sim grid at 60 channels is <1 GB, so the GPU
            is never the memory constraint; CPU memory is for the data
            loader workers. ~100 epochs of a small flow is well under
            6 h, but the job checkpoints every epoch and resumes, so a
            timeout costs one epoch rather than the run.

  CERTIFY   CPU, minutes. 1000 sims x 500 posterior draws is seconds of
            GPU work but the sampling loop is python-side; give it an
            hour and 8 cores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simorgh.simulate.grid import read_grid_meta  # noqa: E402

# --- resource table: the single source of truth -------------------------
SIMULATE_TIME = "3:00:00"
SIMULATE_MEM = "8G"
SIMULATE_CPUS = 1

TRAIN_TIME = "6:00:00"
TRAIN_MEM = "64G"
TRAIN_CPUS = 8
TRAIN_GPUS = 1

CERTIFY_TIME = "1:00:00"
CERTIFY_MEM = "16G"
CERTIFY_CPUS = 8

HEADER = """\
#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name={name}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{extra}#SBATCH --output={log_dir}/%x-%j.out

# --- SIMORGH on DRAC Fir ------------------------------------------------
# Separate environment from aster-env on purpose: SIMORGH does not import
# the reduction pipeline, and pinning them together would couple two
# release cycles that have no reason to move at the same time.
module load StdEnv/2023 {python_module}{cuda_module}
source {venv}/bin/activate
export PYTHONPATH={repo}${{PYTHONPATH:+:$PYTHONPATH}}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

"""

SIMULATE_BODY = """\
# Idempotent: hash-matching shards are skipped, so resubmitting the whole
# array after partial failures costs only the missing shards.
python -u {repo}/scripts/fir/simulate_shard.py \\
    --grid-dir {grid_dir} \\
    --index $SLURM_ARRAY_TASK_ID
"""

TRAIN_BODY = """\
# Resumes from {run_dir}/checkpoint.pt if present. Requeue-safe.
python -u {repo}/scripts/fir/train_model.py \\
    --grid-dir {grid_dir} \\
    --out-dir {run_dir} \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --num-workers $SLURM_CPUS_PER_TASK
"""

CERTIFY_BODY = """\
# Held-out grid: a certificate computed on the training grid is optimistic
# and the script warns loudly if the hashes match.
python -u {repo}/scripts/fir/certify.py \\
    --model {run_dir}/best \\
    --grid-dir {holdout_dir} \\
    --n-sims {n_certify}
"""


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--holdout-dir", default=None,
                    help="held-out grid for certify (default: <grid-dir>_holdout)")
    ap.add_argument("--job-dir", default="~/simorgh/jobs")
    ap.add_argument("--log-dir", default=None, help="default: <job-dir>/logs")
    ap.add_argument("--account", default="def-ncowan")
    ap.add_argument("--venv", default="~/simorgh/simorgh-env")
    ap.add_argument("--repo", default="~/simorgh/SIMORGH")
    ap.add_argument("--python-module", default="python/3.13.2")
    ap.add_argument("--cuda-module", default="cuda/12.6")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-certify", type=int, default=1000)
    args = ap.parse_args()

    grid_dir = str(Path(args.grid_dir).expanduser())
    run_dir = str(Path(args.run_dir).expanduser())
    holdout = str(Path(args.holdout_dir).expanduser()) if args.holdout_dir \
        else grid_dir + "_holdout"
    job_dir = Path(args.job_dir).expanduser()
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else job_dir / "logs"
    venv = str(Path(args.venv).expanduser())
    repo = str(Path(args.repo).expanduser())
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = read_grid_meta(Path(grid_dir))
    n_shards = meta["n_shards"]
    total = n_shards * meta["sims_per_shard"]

    common = dict(account=args.account, log_dir=str(log_dir), venv=venv,
                  repo=repo, python_module=args.python_module)

    sim_sh = write(job_dir / "simulate.sbatch",
                   HEADER.format(name="simorgh_sim", time=SIMULATE_TIME,
                                 cpus=SIMULATE_CPUS, mem=SIMULATE_MEM,
                                 extra="", cuda_module="", **common)
                   + SIMULATE_BODY.format(repo=repo, grid_dir=grid_dir))

    train_sh = write(job_dir / "train.sbatch",
                     HEADER.format(name="simorgh_train", time=TRAIN_TIME,
                                   cpus=TRAIN_CPUS, mem=TRAIN_MEM,
                                   extra=f"#SBATCH --gpus-per-node={TRAIN_GPUS}\n",
                                   cuda_module=f" {args.cuda_module}", **common)
                     + TRAIN_BODY.format(repo=repo, grid_dir=grid_dir,
                                         run_dir=run_dir, epochs=args.epochs,
                                         batch_size=args.batch_size))

    cert_sh = write(job_dir / "certify.sbatch",
                    HEADER.format(name="simorgh_certify", time=CERTIFY_TIME,
                                  cpus=CERTIFY_CPUS, mem=CERTIFY_MEM,
                                  extra="", cuda_module="", **common)
                    + CERTIFY_BODY.format(repo=repo, run_dir=run_dir,
                                          holdout_dir=holdout,
                                          n_certify=args.n_certify))

    plan = {
        "grid": grid_dir, "meta_hash": meta["meta_hash"],
        "n_shards": n_shards, "total_sims": total,
        "scripts": {"simulate": str(sim_sh), "train": str(train_sh),
                    "certify": str(cert_sh)},
    }
    (job_dir / "plan.json").write_text(json.dumps(plan, indent=2))

    print(f"grid       : {grid_dir}")
    print(f"hash       : {meta['meta_hash']}")
    print(f"shards     : {n_shards}   total sims: {total:,}")
    print(f"scripts    : {job_dir}\n")
    print("SUBMISSION PLAN — submit from a shell with NO venv active")
    print("(SLURM exports the submitting environment; module purge cannot")
    print(" undo an activation):\n")
    print("  # 1. simulation grid (array; idempotent, safe to resubmit)")
    print(f"  sbatch --array=0-{n_shards - 1} {sim_sh}\n")
    print("  # 2. verify completeness BEFORE training")
    print(f"  python {repo}/scripts/fir/grid_status.py --grid-dir {grid_dir}\n")
    print("  # 3. train (one H100; resumes on requeue)")
    print(f"  sbatch {train_sh}\n")
    print("  # 4. certify against a HELD-OUT grid")
    print(f"  #    define it first with a different --base-seed:")
    print(f"  #    python {repo}/scripts/fir/define_grid.py --grid-dir {holdout} ... --base-seed 999")
    print(f"  sbatch {cert_sh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
