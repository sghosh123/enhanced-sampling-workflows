"""Diagnostics for comparing successive free-energy surfaces (FESs).

These metrics are heuristics for assessing stability, not proof of thermodynamic
convergence. They should be combined with sampling diagnostics and replicate runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label, minimum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import ConfigurationError


def normalize_fes(fes: np.ndarray) -> np.ndarray:
    """Return an FES shifted so its finite global minimum is zero."""
    values = np.asarray(fes, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ConfigurationError("FES has no finite values.")
    return values - float(np.min(values[finite]))


def local_minima(fes: np.ndarray, *, max_count: int = 10) -> list[tuple[float, tuple[int, ...]]]:
    """Find discrete local minima and collapse adjacent cells on flat minima plateaus.

    Returned indices are array indices (zero-based), not physical CV coordinates.
    """
    values = np.asarray(fes, dtype=float)
    if values.ndim < 1 or values.ndim > 3:
        raise ConfigurationError("Only 1D, 2D, or 3D FES arrays are supported.")
    if max_count < 1:
        raise ConfigurationError("max_count must be >= 1.")
    finite = np.isfinite(values)
    if not finite.any():
        raise ConfigurationError("FES has no finite values.")
    safe = np.where(finite, values, np.inf)
    footprint = (3,) * values.ndim
    local_mask = finite & (safe <= minimum_filter(safe, size=footprint, mode="nearest"))
    # Treat an adjacent flat plateau as one minimum, rather than counting each cell.
    structure = np.ones((3,) * values.ndim, dtype=bool)
    labels, n_labels = label(local_mask, structure=structure)
    found: list[tuple[float, tuple[int, ...]]] = []
    for component in range(1, n_labels + 1):
        coords = np.argwhere(labels == component)
        if coords.size == 0:
            continue
        component_values = np.array([values[tuple(idx)] for idx in coords])
        position = tuple(int(v) for v in coords[int(np.argmin(component_values))])
        found.append((float(values[position]), position))
    found.sort(key=lambda item: item[0])
    return found[:max_count]


@dataclass(frozen=True)
class FESComparison:
    """Numeric comparison between two minimum-shifted FES arrays."""

    max_abs_delta_kj_mol: float
    rmse_delta_kj_mol: float
    mean_abs_delta_kj_mol: float
    compared_points: int
    minima_count_a: int
    minima_count_b: int
    matched_minima: int
    max_minima_shift_grid: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_fes(fes_a: np.ndarray, fes_b: np.ndarray, *, minima_count: int = 3) -> FESComparison:
    """Compare two same-shaped surfaces after independently shifting each minimum to zero.

    Energy metrics are evaluated only where both surfaces are finite. The minima
    shift is the largest spatial displacement (in grid-index units) after minimum
    basins from each surface are matched by minimum total spatial distance.
    """
    a = np.asarray(fes_a, dtype=float)
    b = np.asarray(fes_b, dtype=float)
    if a.shape != b.shape:
        raise ConfigurationError(f"FES shapes differ: {a.shape} versus {b.shape}.")
    if minima_count < 1:
        raise ConfigurationError("minima_count must be >= 1.")
    an, bn = normalize_fes(a), normalize_fes(b)
    common = np.isfinite(an) & np.isfinite(bn)
    if not common.any():
        raise ConfigurationError("The two FES arrays have no overlapping finite grid points.")
    delta = an[common] - bn[common]
    aa = local_minima(an, max_count=minima_count)
    bb = local_minima(bn, max_count=minima_count)
    na, nb = len(aa), len(bb)
    if na == 0 or nb == 0:
        shift = None
        matched = 0
    else:
        pos_a = np.asarray([item[1] for item in aa], dtype=float)
        pos_b = np.asarray([item[1] for item in bb], dtype=float)
        distance = cdist(pos_a, pos_b, metric="euclidean")
        row_idx, col_idx = linear_sum_assignment(distance)
        shifts = distance[row_idx, col_idx]
        matched = int(len(shifts))
        # A mismatch in the number of candidate basins is explicitly penalized.
        shift = float(np.max(shifts)) if na == nb else None
    return FESComparison(
        max_abs_delta_kj_mol=float(np.max(np.abs(delta))),
        rmse_delta_kj_mol=float(np.sqrt(np.mean(delta**2))),
        mean_abs_delta_kj_mol=float(np.mean(np.abs(delta))),
        compared_points=int(delta.size),
        minima_count_a=na,
        minima_count_b=nb,
        matched_minima=matched,
        max_minima_shift_grid=shift,
    )


@dataclass
class ConvergenceTracker:
    """Track consecutive FES comparisons and require every configured threshold.

    ``max_delta_f`` is checked independently from RMSE. Convergence cannot be
    reported until there are ``consecutive_needed`` consecutive passing comparisons.
    """

    max_delta_f: float = 5.0
    max_rmse: float = 2.5
    max_minima_shift_grid: float = 1.0
    minima_count: int = 3
    consecutive_needed: int = 4
    previous_fes: np.ndarray | None = None
    stable_count: int = 0
    last_step: int | None = None

    def __post_init__(self) -> None:
        if self.max_delta_f < 0 or self.max_rmse < 0 or self.max_minima_shift_grid < 0:
            raise ConfigurationError("Convergence thresholds must be non-negative.")
        if self.minima_count < 1 or self.consecutive_needed < 1:
            raise ConfigurationError("minima_count and consecutive_needed must be >= 1.")

    def observe(self, fes: np.ndarray, *, step: int) -> dict[str, Any]:
        current = np.asarray(fes, dtype=float).copy()
        if self.last_step is not None and step <= self.last_step:
            raise ConfigurationError("Convergence observations must have strictly increasing step values.")
        self.last_step = int(step)
        if self.previous_fes is None:
            self.previous_fes = current
            self.stable_count = 0
            return {
                "step": int(step), "comparison": None, "stable_count": 0,
                "converged": False, "reason": "first_surface_has_no_previous_comparison",
            }
        comparison = compare_fes(self.previous_fes, current, minima_count=self.minima_count)
        passed = (
            comparison.max_abs_delta_kj_mol <= self.max_delta_f
            and comparison.rmse_delta_kj_mol <= self.max_rmse
            and comparison.max_minima_shift_grid is not None
            and comparison.max_minima_shift_grid <= self.max_minima_shift_grid
            and comparison.minima_count_a == self.minima_count
            and comparison.minima_count_b == self.minima_count
        )
        self.stable_count = self.stable_count + 1 if passed else 0
        converged = self.stable_count >= self.consecutive_needed
        self.previous_fes = current
        return {
            "step": int(step), "comparison": comparison.to_dict(),
            "thresholds_passed": bool(passed), "stable_count": self.stable_count,
            "converged": bool(converged),
            "reason": "thresholds_passed" if passed else "one_or_more_thresholds_failed",
        }
