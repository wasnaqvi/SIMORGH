#!/usr/bin/env python
"""Report grid completeness. Login node, seconds.

    python scripts/fir/grid_status.py --grid-dir ~/scratch/simorgh/grids/v1

Run this BEFORE submitting training. Training on a partial grid silently
changes the effective prior — a failure mode that produces a
plausible-looking network with no valid interpretation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simorgh.simulate.grid import grid_status  # noqa: E402


def _ranges(idx: list[int]) -> str:
    """Compress [0,1,2,7,8] -> '0-2,7-8' for a resubmission --array string."""
    if not idx:
        return ""
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = i
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    args = ap.parse_args()

    st = grid_status(Path(args.grid_dir).expanduser())
    pct = 100.0 * st["present"] / max(st["n_shards"], 1)
    print(f"grid       : {Path(args.grid_dir).expanduser()}")
    print(f"hash       : {st['meta_hash']}")
    print(f"shards     : {st['present']}/{st['n_shards']}  ({pct:.1f}%)")
    print(f"sims       : {st['n_sims']:,}")
    if st["missing"]:
        print(f"MISSING    : {len(st['missing'])} shard(s)")
        print(f"resubmit   : sbatch --array={_ranges(st['missing'])} "
              "<job-dir>/simulate.sbatch")
        return 1
    print("status     : COMPLETE — ready to train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
