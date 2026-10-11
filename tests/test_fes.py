import json
import numpy as np
import pytest

from enhanced_sampling_workflows.config import ConfigurationError, validate_collective_variables
from enhanced_sampling_workflows.fes import (
    build_fes_metadata, load_fes_bundle, openmm_axis_values, plot_fes,
    plot_fes_slices, save_fes_bundle,
)


def cv_list_2d():
    return [
        {"name": "d1", "kind": "distance", "unit": "nm", "min_value": 1.0, "max_value": 2.0},
        {"name": "d2", "kind": "distance", "unit": "nm", "min_value": 0.0, "max_value": 1.0},
    ]


def test_openmm_axis_does_not_include_upper_boundary():
    assert np.allclose(openmm_axis_values(1.0, 2.0, 4), [1.0, 1.25, 1.5, 1.75])


def test_fes_bundle_round_trip(tmp_path):
    fes = np.arange(12, dtype=float).reshape(3, 4)
    cvs = cv_list_2d()
    meta = build_fes_metadata(fes, cvs, extra={"step": 10})
    npy, sidecar = save_fes_bundle(fes, tmp_path / "fes_final", meta)
    loaded, loaded_meta = load_fes_bundle(npy)
    assert np.array_equal(loaded, fes)
    assert loaded_meta["extra"]["step"] == 10
    assert sidecar.exists()


def test_save_rejects_shape_metadata_mismatch(tmp_path):
    fes = np.zeros((3, 4))
    with pytest.raises(ConfigurationError, match="does not match"):
        save_fes_bundle(fes, tmp_path / "fes", {"fes_shape": [4, 3], "axes": []})


def test_plot_2d_fes(tmp_path):
    pytest.importorskip("matplotlib")
    fes = np.arange(12, dtype=float).reshape(3, 4)
    meta = build_fes_metadata(fes, cv_list_2d())
    path = plot_fes(fes, meta, tmp_path / "fes.png")
    assert path.exists() and path.stat().st_size > 1000


def test_plot_3d_slices(tmp_path):
    pytest.importorskip("matplotlib")
    fes = np.random.default_rng(9).normal(size=(5, 4, 3))
    cvs = [
        {"name": "d1", "kind": "distance", "unit": "nm", "min_value": 1.0, "max_value": 2.0},
        {"name": "d2", "kind": "distance", "unit": "nm", "min_value": 0.0, "max_value": 1.0},
        {"name": "d3", "kind": "distance", "unit": "nm", "min_value": 0.2, "max_value": 0.8},
    ]
    meta = build_fes_metadata(fes, cvs)
    files = plot_fes_slices(fes, meta, tmp_path / "slices", number_of_slices=3)
    assert len(files) == 3
    assert all(path.exists() for path in files)


def test_cv_validation_rejects_duplicate_names_and_bad_units():
    good = {
        "name": "d1", "kind": "distance", "atoms": [
            {"segid": "A", "resid": 1, "name": "CA"},
            {"segid": "B", "resid": 2, "name": "CA"}],
        "min_value": 1, "max_value": 2, "bias_width": 0.1, "unit": "nm"
    }
    with pytest.raises(ConfigurationError, match="Duplicate"):
        validate_collective_variables([good, good])
    bad = {**good, "name": "d2", "unit": "angstrom"}
    with pytest.raises(ConfigurationError, match="must use nm"):
        validate_collective_variables([bad])


def test_config_json_sidecar_is_plain_json(tmp_path):
    fes = np.ones((2, 2))
    meta = build_fes_metadata(fes, cv_list_2d())
    _, json_path = save_fes_bundle(fes, tmp_path / "fes", meta)
    payload = json.loads(json_path.read_text())
    assert payload["axes"][0]["coordinates"].startswith("min + i")
