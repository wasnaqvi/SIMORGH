"""File-boundary loader for Patchwork survey products.

SIMORGH never imports aster_toolkit. Patchwork exports transmission
spectra as plain files; this module reads them. Expected format: a text
table with columns (wavelength_um, depth, depth_err), '#' comments,
optional fourth column (bin width, ignored), plus an optional sidecar
<name>.json with provenance (planet, visit, reduction id, fit_version).
Keeping the boundary at files keeps provenance auditable and the two
codebases independently versioned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..spectra import Spectrum


def load_spectrum(path: str | Path) -> Spectrum:
    path = Path(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path}: need >= 3 columns (wavelength, depth, err)")
    meta: dict = {"source_file": str(path)}
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        meta.update(json.loads(sidecar.read_text()))
    finite = np.all(np.isfinite(data[:, :3]), axis=1)
    return Spectrum(wavelength=data[:, 0], depth=data[:, 1],
                    error=data[:, 2], mask=finite, meta=meta)


def load_survey(directory: str | Path, pattern: str = "*.dat") -> list[Spectrum]:
    """All spectra under a directory, sorted by filename."""
    directory = Path(directory)
    specs = [load_spectrum(p) for p in sorted(directory.glob(pattern))]
    if not specs:
        raise FileNotFoundError(f"no '{pattern}' files under {directory}")
    return specs
