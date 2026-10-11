"""Umbrella-window grid generation and source-frame assignment."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from .config import ConfigurationError

EmptyPolicy = Literal["skip", "nearest", "error"]


def _centers(low: float, high: float, spacing: float, *, label: str) -> np.ndarray:
    low, high, spacing = float(low), float(high), float(spacing)
    if not all(np.isfinite(v) for v in (low, high, spacing)):
        raise ConfigurationError(f"{label} range and spacing must be finite.")
    if high <= low or spacing <= 0:
        raise ConfigurationError(f"{label} requires high > low and spacing > 0.")
    count_float = (high - low) / spacing
    count = int(round(count_float))
    if count < 1 or not math.isclose(count_float, count, rel_tol=1e-8, abs_tol=1e-8):
        raise ConfigurationError(
            f"{label} range width ({high-low:g}) must be an integer multiple of spacing ({spacing:g})."
        )
    return low + (np.arange(count, dtype=float) + 0.5) * spacing


def assign_grid_windows(
    cv_table: pd.DataFrame,
    *,
    cv1: str,
    cv2: str,
    cv1_range: tuple[float, float],
    cv2_range: tuple[float, float],
    spacing: float,
    empty_policy: EmptyPolicy = "skip",
    max_seed_distance: float | None = None,
    trajectory_column: str = "trajectory_id",
    frame_column: str = "frame",
) -> pd.DataFrame:
    """Create a rectangular 2D window grid and choose a source frame for each.

    ``max_seed_distance`` is measured in grid-spacing units (scaled Euclidean
    distance in CV space). Empty bins are skipped by default. With ``nearest``, an
    out-of-bin frame may seed an empty window; the offset is always recorded and an
    optional maximum distance can prevent arbitrarily poor matches.

    Frames in a single occupied bin are assigned by closest distance to its center.
    Nearest-frame fallback may reuse a frame in more than one window; the final
    ``seed_reuse_count`` column makes this explicit.
    """
    if empty_policy not in {"skip", "nearest", "error"}:
        raise ConfigurationError("empty_policy must be 'skip', 'nearest', or 'error'.")
    if cv1 not in cv_table or cv2 not in cv_table:
        raise ConfigurationError(f"CV table must contain columns {cv1!r} and {cv2!r}.")
    for column in (trajectory_column, frame_column):
        if column not in cv_table:
            raise ConfigurationError(f"CV table must contain identity column {column!r}.")
    if cv_table.empty:
        raise ConfigurationError("CV table has no rows.")
    if max_seed_distance is not None and (not np.isfinite(max_seed_distance) or max_seed_distance < 0):
        raise ConfigurationError("max_seed_distance must be a finite non-negative value.")

    x_centers = _centers(*cv1_range, spacing, label=cv1)
    y_centers = _centers(*cv2_range, spacing, label=cv2)
    x = pd.to_numeric(cv_table[cv1], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(cv_table[cv2], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.any():
        identity = cv_table.loc[valid, [trajectory_column, frame_column]]
        if identity.isna().any().any():
            raise ConfigurationError("Rows with finite CVs must also have trajectory IDs and frame numbers.")
        numeric_frames = pd.to_numeric(identity[frame_column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(numeric_frames)) or not np.all(numeric_frames >= 0) or not np.all(numeric_frames == np.floor(numeric_frames)):
            raise ConfigurationError("Frame numbers for finite CV rows must be non-negative integers.")
        if any(not str(value).strip() for value in identity[trajectory_column]):
            raise ConfigurationError("Trajectory IDs for finite CV rows must be non-empty.")
    if not np.any(valid):
        raise ConfigurationError("CV table contains no rows with finite values for both CVs.")
    # Keep source row indices so trajectories and local frame numbers remain exact.
    source = cv_table.reset_index(drop=True)
    valid_indices = np.flatnonzero(valid)
    x_valid, y_valid = x[valid], y[valid]

    x_low, x_high = float(cv1_range[0]), float(cv1_range[1])
    y_low, y_high = float(cv2_range[0]), float(cv2_range[1])
    # Assign bin indices with a closed upper boundary on the final bin only.
    in_range = (
        (x_valid >= x_low) & (x_valid <= x_high)
        & (y_valid >= y_low) & (y_valid <= y_high)
    )
    xv, yv = x_valid[in_range], y_valid[in_range]
    source_idx = valid_indices[in_range]
    nx, ny = len(x_centers), len(y_centers)
    ix = np.floor((xv - x_low) / spacing).astype(int)
    iy = np.floor((yv - y_low) / spacing).astype(int)
    ix = np.where(ix == nx, nx - 1, ix)
    iy = np.where(iy == ny, ny - 1, iy)
    bins: dict[tuple[int, int], list[int]] = {}
    for row_index, bi, bj in zip(source_idx, ix, iy):
        bins.setdefault((int(bi), int(bj)), []).append(int(row_index))

    records: list[dict[str, object]] = []
    for i, xc in enumerate(x_centers):
        for j, yc in enumerate(y_centers):
            lo1, hi1 = float(xc - spacing / 2), float(xc + spacing / 2)
            lo2, hi2 = float(yc - spacing / 2), float(yc + spacing / 2)
            candidates = bins.get((i, j), [])
            status = "occupied"
            selected_index: int | None = None
            if candidates:
                cand_x = x[candidates]
                cand_y = y[candidates]
                dist2 = ((cand_x - xc) / spacing) ** 2 + ((cand_y - yc) / spacing) ** 2
                selected_index = candidates[int(np.argmin(dist2))]
            elif empty_policy == "error":
                raise ConfigurationError(
                    f"Window ({xc:g}, {yc:g}) has no frame in its bin; choose another empty_policy to continue."
                )
            elif empty_policy == "nearest":
                distances = np.sqrt(((x_valid - xc) / spacing) ** 2 + ((y_valid - yc) / spacing) ** 2)
                nearest_position = int(np.argmin(distances))
                nearest_dist = float(distances[nearest_position])
                if max_seed_distance is None or nearest_dist <= max_seed_distance:
                    selected_index = int(valid_indices[nearest_position])
                    status = "nearest"
                else:
                    status = "unseeded"
            else:
                status = "unseeded"

            record: dict[str, object] = {
                "window_id": f"{cv1}_{xc:.4f}__{cv2}_{yc:.4f}",
                "cv1_name": cv1,
                "cv2_name": cv2,
                "cv1_center": float(xc),
                "cv2_center": float(yc),
                "cv1_low": lo1,
                "cv1_high": hi1,
                "cv2_low": lo2,
                "cv2_high": hi2,
                "spacing": float(spacing),
                "trajectory_id": None,
                "frame": None,
                "status": status,
                "seed_cv1": np.nan,
                "seed_cv2": np.nan,
                "offset_cv1": np.nan,
                "offset_cv2": np.nan,
                "seed_distance_grid_units": np.nan,
            }
            if selected_index is not None:
                selected = source.iloc[selected_index]
                seed_x, seed_y = float(x[selected_index]), float(y[selected_index])
                record["trajectory_id"] = str(selected[trajectory_column])
                record["frame"] = int(selected[frame_column])
                record["seed_cv1"] = seed_x
                record["seed_cv2"] = seed_y
                record["offset_cv1"] = seed_x - float(xc)
                record["offset_cv2"] = seed_y - float(yc)
                record["seed_distance_grid_units"] = float(
                    math.sqrt(((seed_x - xc) / spacing) ** 2 + ((seed_y - yc) / spacing) ** 2)
                )
            records.append(record)

    result = pd.DataFrame.from_records(records)
    counts: dict[tuple[str, int], int] = {}
    assigned = result[result["status"] != "unseeded"]
    for _, row in assigned.iterrows():
        key = (str(row["trajectory_id"]), int(row["frame"]))
        counts[key] = counts.get(key, 0) + 1
    result["seed_reuse_count"] = [
        counts.get((str(row.trajectory_id), int(row.frame)), 0)
        if row.status != "unseeded" else 0
        for row in result.itertuples(index=False)
    ]
    result.attrs["dropped_nonfinite_rows"] = int((~valid).sum())
    result.attrs["out_of_range_rows"] = int((~in_range).sum())
    result.attrs["empty_policy"] = empty_policy
    return result
