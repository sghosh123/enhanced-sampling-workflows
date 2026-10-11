import numpy as np
import pytest

from enhanced_sampling_workflows.config import ConfigurationError
from enhanced_sampling_workflows.convergence import ConvergenceTracker, compare_fes, local_minima, normalize_fes


def single_basin():
    return np.array([[3.0, 2.0, 3.0], [2.0, 0.0, 2.0], [3.0, 2.0, 3.0]])


def test_normalize_uses_finite_minimum():
    surface = np.array([[np.nan, 4.0], [2.0, 3.0]])
    normalized = normalize_fes(surface)
    assert np.isnan(normalized[0, 0])
    assert normalized[1, 0] == 0.0
    assert normalized[0, 1] == 2.0


def test_flat_plateau_is_one_local_minimum():
    surface = np.ones((5, 5))
    minima = local_minima(surface, max_count=10)
    assert len(minima) == 1
    assert minima[0][0] == 1.0


def test_compare_identical_fes_is_zero():
    surface = single_basin()
    result = compare_fes(surface, surface.copy(), minima_count=1)
    assert result.max_abs_delta_kj_mol == 0.0
    assert result.rmse_delta_kj_mol == 0.0
    assert result.max_minima_shift_grid == 0.0
    assert result.matched_minima == 1


def test_compare_aligns_surfaces_by_minimum():
    a = single_basin()
    b = a + 50.0
    result = compare_fes(a, b, minima_count=1)
    assert result.max_abs_delta_kj_mol == 0.0
    assert result.rmse_delta_kj_mol == 0.0


def test_shape_mismatch_is_rejected():
    with pytest.raises(ConfigurationError, match="shapes differ"):
        compare_fes(np.zeros((2, 3)), np.zeros((3, 2)))


def test_tracker_requires_consecutive_comparisons():
    surface = single_basin()
    tracker = ConvergenceTracker(
        max_delta_f=0.1, max_rmse=0.1, max_minima_shift_grid=0.0,
        minima_count=1, consecutive_needed=2,
    )
    first = tracker.observe(surface, step=100)
    second = tracker.observe(surface, step=200)
    third = tracker.observe(surface, step=300)
    assert first["converged"] is False
    assert second["stable_count"] == 1
    assert second["converged"] is False
    assert third["stable_count"] == 2
    assert third["converged"] is True


def test_maximum_delta_threshold_is_enforced_not_just_rmse():
    a = single_basin()
    b = a.copy()
    b[0, 0] += 2.0
    tracker = ConvergenceTracker(
        max_delta_f=0.5, max_rmse=10.0, max_minima_shift_grid=0.0,
        minima_count=1, consecutive_needed=1,
    )
    tracker.observe(a, step=100)
    result = tracker.observe(b, step=200)
    assert result["comparison"]["max_abs_delta_kj_mol"] > 0.5
    assert result["thresholds_passed"] is False
    assert result["converged"] is False


def test_non_increasing_step_is_rejected():
    tracker = ConvergenceTracker(minima_count=1)
    tracker.observe(single_basin(), step=100)
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        tracker.observe(single_basin(), step=100)
