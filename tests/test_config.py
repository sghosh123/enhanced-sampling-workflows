from pathlib import Path

import pytest

from enhanced_sampling_workflows.config import (
    ConfigurationError, load_json, validate_collective_variables, validate_trajectory_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_example_cv_configs_validate():
    for name in ("trajectory_cv.example.json", "metadynamics_3cv.example.json",
                 "metadynamics_distance_angle.example.json", "umbrella_distances.example.json"):
        config = load_json(ROOT / "configs" / name)
        assert validate_collective_variables(config["collective_variables"])


def test_trajectory_config_preserves_stable_ids_and_resolves_paths():
    config = load_json(ROOT / "configs" / "trajectory_cv.example.json")
    trajectories, cvs = validate_trajectory_config(config)
    assert [item["id"] for item in trajectories] == ["T4", "T7", "T8", "T26"]
    assert Path(trajectories[0]["topology"]).is_absolute()
    assert cvs[0]["name"] == "d_A1X1"


def test_angle_requires_four_atoms_and_radians():
    with pytest.raises(ConfigurationError, match="requires exactly 4"):
        validate_collective_variables([{
            "name": "angle", "kind": "vector_angle", "atoms": [],
            "min_value": 0, "max_value": 3.14, "bias_width": 0.05, "unit": "rad"
        }])
    atoms = [{"segid": "A", "resid": i, "name": "CA"} for i in range(4)]
    with pytest.raises(ConfigurationError, match="must use radians"):
        validate_collective_variables([{
            "name": "angle", "kind": "vector_angle", "atoms": atoms,
            "min_value": 0, "max_value": 3.14, "bias_width": 0.05, "unit": "degrees"
        }])


def test_config_requires_unique_trajectory_ids():
    config = load_json(ROOT / "configs" / "trajectory_cv.example.json")
    config["trajectories"].append(dict(config["trajectories"][0]))
    with pytest.raises(ConfigurationError, match="Duplicate trajectory id"):
        validate_trajectory_config(config)


def test_metadynamics_settings_validate_hill_cadence():
    from enhanced_sampling_workflows.openmm_runner import _validate_metadynamics_settings

    valid = _validate_metadynamics_settings({
        "hill_frequency_steps": 100,
        "save_frequency_steps": 1000,
        "steps_per_block": 2000,
        "total_steps": 8000,
    })
    assert valid["steps_per_block"] == 2000
    with pytest.raises(ConfigurationError, match="multiple of hill_frequency_steps"):
        _validate_metadynamics_settings({"hill_frequency_steps": 100, "save_frequency_steps": 150})


def test_cv_ranges_and_periodicity_are_checked():
    base = {
        "name": "distance", "kind": "distance",
        "atoms": [{"segid": "A", "resid": 1, "name": "CA"},
                  {"segid": "B", "resid": 2, "name": "CA"}],
        "min_value": 0.0, "max_value": 2.0, "bias_width": 0.05, "unit": "nm",
    }
    with pytest.raises(ConfigurationError, match="not a wrapped periodic variable"):
        validate_collective_variables([{**base, "periodic": True}])
    with pytest.raises(ConfigurationError, match="negative minimum"):
        validate_collective_variables([{**base, "min_value": -0.1}])


def test_metadynamics_total_steps_must_align_with_hill_frequency():
    from enhanced_sampling_workflows.openmm_runner import _validate_metadynamics_settings

    with pytest.raises(ConfigurationError, match="total_steps must be divisible"):
        _validate_metadynamics_settings({
            "hill_frequency_steps": 100, "save_frequency_steps": 1000,
            "steps_per_block": 2000, "total_steps": 2050,
        })


def test_runner_refuses_to_mix_existing_output(tmp_path):
    from enhanced_sampling_workflows.openmm_runner import _create_fresh_output_dir

    fresh = tmp_path / "new-run"
    _create_fresh_output_dir(fresh)
    assert fresh.is_dir()
    (fresh / "previous-result.txt").write_text("existing run")
    with pytest.raises(FileExistsError, match="not empty"):
        _create_fresh_output_dir(fresh)
