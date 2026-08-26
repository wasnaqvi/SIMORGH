import json

import numpy as np

from simorgh.io import load_spectrum, load_survey


def test_load_spectrum_with_sidecar(tmp_path):
    lam = np.linspace(2.9, 5.1, 40)
    depth = np.full(40, 4.8e-3)
    err = np.full(40, 8e-5)
    depth[7] = np.nan                       # bad channel -> masked
    arr = np.column_stack([lam, depth, err])
    f = tmp_path / "toi270d_visit1_nominal.dat"
    np.savetxt(f, arr, header="wavelength_um depth depth_err")
    f.with_suffix(".json").write_text(json.dumps(
        {"planet": "TOI-270 d", "reduction": "nominal", "fit_version": "1.3"}))

    spec = load_spectrum(f)
    assert spec.n_channels == 39
    assert not spec.mask[7]
    assert spec.meta["planet"] == "TOI-270 d"
    assert spec.meta["fit_version"] == "1.3"


def test_load_survey_sorted(tmp_path):
    lam = np.linspace(2.9, 5.1, 10)
    for name in ("b_planet.dat", "a_planet.dat"):
        np.savetxt(tmp_path / name,
                   np.column_stack([lam, np.full(10, 5e-3), np.full(10, 1e-4)]))
    specs = load_survey(tmp_path)
    assert len(specs) == 2
    assert specs[0].meta["source_file"].endswith("a_planet.dat")
