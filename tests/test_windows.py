import pandas as pd
import pytest

from enhanced_sampling_workflows.config import ConfigurationError
from enhanced_sampling_workflows.windows import assign_grid_windows


def sample_table():
    return pd.DataFrame({
        "trajectory_id": ["T4", "T7"],
        "frame": [0, 0],
        "cv_x": [0.2, 0.8],
        "cv_y": [0.2, 0.8],
    })


def test_grid_assigns_occupied_bins_and_includes_upper_boundary():
    table = pd.DataFrame({
        "trajectory_id": ["T4", "T8"],
        "frame": [1, 3],
        "x": [0.25, 1.0],
        "y": [0.25, 1.0],
    })
    result = assign_grid_windows(
        table, cv1="x", cv2="y", cv1_range=(0.0, 1.0), cv2_range=(0.0, 1.0),
        spacing=0.5, empty_policy="skip",
    )
    assert len(result) == 4
    assert (result["status"] == "occupied").sum() == 2
    upper = result[(result["cv1_center"] == 0.75) & (result["cv2_center"] == 0.75)].iloc[0]
    assert upper["trajectory_id"] == "T8"
    assert int(upper["frame"]) == 3


def test_skip_is_safe_default_for_empty_windows():
    result = assign_grid_windows(
        sample_table(), cv1="cv_x", cv2="cv_y", cv1_range=(0.0, 1.0), cv2_range=(0.0, 1.0),
        spacing=0.5,
    )
    assert (result["status"] == "unseeded").sum() >= 1
    empty = result[result["status"] == "unseeded"]
    assert empty["frame"].isna().all()


def test_nearest_fallback_reports_offset_and_reuse():
    result = assign_grid_windows(
        sample_table(), cv1="cv_x", cv2="cv_y", cv1_range=(0.0, 1.0), cv2_range=(0.0, 1.0),
        spacing=0.5, empty_policy="nearest", max_seed_distance=2.0,
    )
    nearest = result[result["status"] == "nearest"]
    assert len(nearest) >= 1
    assert nearest["seed_distance_grid_units"].notna().all()
    assert nearest["seed_reuse_count"].ge(1).all()
    assert (nearest["offset_cv1"].abs() + nearest["offset_cv2"].abs()).gt(0).all()


def test_nearest_cutoff_can_leave_window_unseeded():
    result = assign_grid_windows(
        sample_table(), cv1="cv_x", cv2="cv_y", cv1_range=(0.0, 1.0), cv2_range=(0.0, 1.0),
        spacing=0.5, empty_policy="nearest", max_seed_distance=0.1,
    )
    assert (result["status"] == "unseeded").any()


def test_error_policy_fails_on_empty_grid_cell():
    with pytest.raises(ConfigurationError, match="has no frame"):
        assign_grid_windows(
            pd.DataFrame({"trajectory_id": ["T"], "frame": [0], "x": [0.1], "y": [0.1]}),
            cv1="x", cv2="y", cv1_range=(0, 1), cv2_range=(0, 1), spacing=0.5,
            empty_policy="error",
        )


def test_non_multiple_range_is_rejected():
    with pytest.raises(ConfigurationError, match="integer multiple"):
        assign_grid_windows(
            sample_table(), cv1="cv_x", cv2="cv_y", cv1_range=(0, 1), cv2_range=(0, 0.9),
            spacing=0.5,
        )


def test_missing_cv_or_identity_columns_is_rejected():
    with pytest.raises(ConfigurationError, match="must contain columns"):
        assign_grid_windows(
            pd.DataFrame({"trajectory_id": ["T"], "frame": [0], "x": [0.2]}),
            cv1="x", cv2="y", cv1_range=(0, 1), cv2_range=(0, 1), spacing=0.5,
        )
