"""Local-minimum detection, frame representatives, and mechanistic FES overlays.

These are geometric tools: a representative frame is selected by CV-space proximity to a
FES minimum, not by a claim that the selected structure is a transition state or reactive
configuration. FES metadata must accurately describe the axis conventions and units.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import ndimage

from .config import ConfigurationError
from .fes import _axes_for, _plotting_imports, openmm_axis_values


def detect_fes_minima(
    fes: np.ndarray,
    metadata: dict[str, Any],
    *,
    min_separation_grid: float = 2.0,
    plateau_atol: float = 1e-8,
    max_minima: int = 20,
) -> pd.DataFrame:
    """Return spatially separated local minima sorted by free energy.

    Adjacent cells belonging to the same flat minimum are first collapsed into one
    connected component. A grid-space non-maximum-suppression step then prevents nearby
    cells on one broad basin from being reported as many distinct minima. Separation is
    measured in grid-index units, not physical CV units.
    """
    values = np.asarray(fes, dtype=float)
    if values.ndim not in (1, 2, 3):
        raise ConfigurationError("FES must be 1D, 2D, or 3D.")
    if not np.isfinite(values).any():
        raise ConfigurationError("FES contains no finite energy values.")
    if not np.isfinite(min_separation_grid) or min_separation_grid < 0:
        raise ConfigurationError("min_separation_grid must be finite and non-negative.")
    if not np.isfinite(plateau_atol) or plateau_atol < 0:
        raise ConfigurationError("plateau_atol must be finite and non-negative.")
    if not isinstance(max_minima, int) or max_minima < 1:
        raise ConfigurationError("max_minima must be a positive integer.")
    axes = _axes_for(values, metadata)
    finite = np.isfinite(values)
    safe = np.where(finite, values, np.inf)
    local = ndimage.minimum_filter(safe, size=3, mode="nearest")
    neighborhood_max = ndimage.maximum_filter(np.where(finite, values, -np.inf), size=3, mode="nearest")
    # A uniform high-energy plateau is not a meaningful basin minimum. Require at least
    # one strictly higher neighboring cell in addition to having no lower neighbor.
    candidate = finite & (values <= local + plateau_atol) & (neighborhood_max > values + plateau_atol)
    if not candidate.any():
        raise ConfigurationError("No strict local minima found; check for a flat or malformed FES.")
    structure = ndimage.generate_binary_structure(values.ndim, values.ndim)
    components, count = ndimage.label(candidate, structure=structure)
    candidates = []
    coord_arrays = [openmm_axis_values(ax["min_value"], ax["max_value"], values.shape[i]) for i, ax in enumerate(metadata["axes"])]
    for component_id in range(1, count + 1):
        idx = np.argwhere(components == component_id)
        if not len(idx):
            continue
        energies = values[tuple(idx.T)]
        best_energy = float(np.min(energies))
        best = idx[np.isclose(energies, best_energy, rtol=0.0, atol=plateau_atol)]
        if len(best) > 1:
            centroid = best.mean(axis=0)
            best_idx = best[int(np.argmin(np.sum((best - centroid) ** 2, axis=1)))]
        else:
            best_idx = best[0]
        candidates.append((best_energy, tuple(int(x) for x in best_idx)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    accepted: list[tuple[float, tuple[int, ...]]] = []
    for candidate_item in candidates:
        energy, index = candidate_item
        if any(np.linalg.norm(np.asarray(index) - np.asarray(previous[1])) < min_separation_grid for previous in accepted):
            continue
        accepted.append((energy, index))
        if len(accepted) >= max_minima:
            break
    rows = []
    for rank, (energy, index) in enumerate(accepted, start=1):
        row: dict[str, Any] = {
            "basin_id": f"basin_{rank:03d}", "rank_by_energy": rank,
            "grid_indices_json": __import__("json").dumps(list(index)), "free_energy_kJ_mol": energy,
        }
        for dim, axis in enumerate(metadata["axes"]):
            row[f"{axis['name']}_index"] = index[dim]
            row[f"{axis['name']}_value_{axis.get('unit', 'unknown')}"] = float(coord_arrays[dim][index[dim]])
        rows.append(row)
    return pd.DataFrame(rows)


def select_basin_representatives(
    cv_table: pd.DataFrame,
    fes: np.ndarray,
    metadata: dict[str, Any],
    *,
    cv_columns: Sequence[str],
    trajectory_column: str = "trajectory_id",
    segment_column: str | None = "segment_id",
    frame_column: str = "frame",
    min_separation_grid: float = 2.0,
    max_minima: int = 20,
    basin_radius_grid: float = 1.5,
    max_sample_distance_grid: float = 2.5,
) -> pd.DataFrame:
    """Match separated FES minima to unique observed frames near each minimum.

    CVs are mapped to the FES grid using the metadata convention ``min + i*range/grid``.
    Any minima without a sampled frame within the allowed scaled grid distance are marked
    ``unrepresented`` rather than assigned a globally nearest frame.
    """
    values = np.asarray(fes, dtype=float)
    axes = metadata.get("axes", [])
    _axes_for(values, metadata)
    if len(cv_columns) != values.ndim:
        raise ConfigurationError(f"Need exactly {values.ndim} CV columns, got {len(cv_columns)}.")
    if len(axes) != len(cv_columns):
        raise ConfigurationError("CV column count does not match FES metadata axes.")
    if not np.isfinite(basin_radius_grid) or basin_radius_grid < 0:
        raise ConfigurationError("basin_radius_grid must be finite and non-negative.")
    if not np.isfinite(max_sample_distance_grid) or max_sample_distance_grid < 0:
        raise ConfigurationError("max_sample_distance_grid must be finite and non-negative.")
    keys = [trajectory_column] + ([segment_column] if segment_column and segment_column in cv_table.columns else []) + [frame_column]
    missing = [c for c in list(cv_columns) + keys if c not in cv_table.columns]
    if missing:
        raise ConfigurationError(f"CV table is missing columns: {missing}")
    if cv_table[keys].isna().any().any() or any(cv_table[c].astype(str).str.strip().eq("").any() for c in keys):
        raise ConfigurationError("CV table contains missing/empty source-frame identity values.")
    if cv_table.duplicated(keys).any():
        raise ConfigurationError(f"CV table contains duplicate source-frame identity keys: {keys}")
    frame_values = pd.to_numeric(cv_table[frame_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(frame_values).all() or np.any(frame_values < 0) or np.any(frame_values != np.floor(frame_values)):
        raise ConfigurationError("Source frame values must be finite non-negative integers.")
    minima = detect_fes_minima(values, metadata, min_separation_grid=min_separation_grid, max_minima=max_minima)
    numeric = cv_table[list(cv_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric).all(axis=1)
    indices = np.full(numeric.shape, -1, dtype=int)
    coord_step = []
    valid = finite.copy()
    for d, axis in enumerate(axes):
        minimum = float(axis["min_value"])
        maximum = float(axis["max_value"])
        width = int(values.shape[d])
        step = (maximum - minimum) / width
        coord_step.append(step)
        within = (numeric[:, d] >= minimum) & (numeric[:, d] < maximum)
        valid &= within
        finite_d = np.isfinite(numeric[:, d])
        indices[finite_d, d] = np.rint((numeric[finite_d, d] - minimum) / step).astype(int)
        indices[:, d] = np.clip(indices[:, d], 0, width - 1)
    used_keys: set[tuple[str, ...]] = set()
    rows: list[dict[str, Any]] = []
    for _, minimum_row in minima.iterrows():
        min_idx = np.array([int(minimum_row[f"{axis['name']}_index"]) for axis in axes], dtype=int)
        center = np.array([float(minimum_row[f"{axis['name']}_value_{axis.get('unit', 'unknown')}"]) for axis in axes])
        distances = np.sqrt(np.sum(((numeric - center) / np.asarray(coord_step)) ** 2, axis=1))
        grid_distances = np.sqrt(np.sum((indices - min_idx) ** 2, axis=1))
        in_basin = valid & (grid_distances <= basin_radius_grid) & np.isfinite(distances)
        candidate_indices = np.flatnonzero(in_basin)
        candidate_indices = sorted(candidate_indices, key=lambda pos: (distances[pos], tuple(str(cv_table.iloc[pos][k]) for k in keys)))
        chosen = None
        for pos in candidate_indices:
            identity = tuple(str(cv_table.iloc[pos][k]) for k in keys)
            if identity not in used_keys and distances[pos] <= max_sample_distance_grid:
                chosen = int(pos)
                used_keys.add(identity)
                break
        row = minimum_row.to_dict()
        row["n_candidate_frames_in_radius"] = int(len(candidate_indices))
        if chosen is None:
            row.update({"representative_status": "unrepresented", "representative_trajectory_id": None,
                        "representative_segment_id": None, "representative_frame": None,
                        "representative_distance_grid": np.nan})
        else:
            source = cv_table.iloc[chosen]
            row.update({"representative_status": "matched", "representative_trajectory_id": str(source[trajectory_column]),
                        "representative_segment_id": str(source[segment_column]) if segment_column and segment_column in source.index else "0",
                        "representative_frame": int(source[frame_column]),
                        "representative_distance_grid": float(distances[chosen])})
            for column in cv_columns:
                row[f"representative_{column}"] = float(pd.to_numeric(source[column], errors="raise"))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fes_overlay(
    fes: np.ndarray,
    metadata: dict[str, Any],
    cv_table: pd.DataFrame,
    *,
    cv_columns: Sequence[str],
    color_column: str,
    output_path: str | Path,
    max_points: int = 12000,
    title: str | None = None,
) -> Path:
    """Plot 2D FES contours with sampled frames colored by a mechanistic observable."""
    values = np.asarray(fes, dtype=float)
    if values.ndim != 2 or len(cv_columns) != 2:
        raise ConfigurationError("FES overlay requires a 2D surface and exactly two CV columns.")
    coordinates = _axes_for(values, metadata)
    missing = [c for c in list(cv_columns) + [color_column] if c not in cv_table.columns]
    if missing:
        raise ConfigurationError(f"CV table is missing overlay columns: {missing}")
    x = pd.to_numeric(cv_table[cv_columns[0]], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(cv_table[cv_columns[1]], errors="coerce").to_numpy(dtype=float)
    color = pd.to_numeric(cv_table[color_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(color)
    positions = np.flatnonzero(valid)
    if len(positions) == 0:
        raise ConfigurationError("No finite CV/color rows are available to plot.")
    if max_points < 1:
        raise ConfigurationError("max_points must be >= 1.")
    if len(positions) > max_points:
        positions = positions[np.linspace(0, len(positions) - 1, max_points).round().astype(int)]
    plt = _plotting_imports()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    contour = ax.contourf(coordinates[0], coordinates[1], values.T, levels=40)
    fig.colorbar(contour, ax=ax, label=f"Free energy [{metadata.get('energy_unit', 'kJ/mol')}]")
    scatter = ax.scatter(x[positions], y[positions], c=color[positions], s=12, alpha=0.8)
    fig.colorbar(scatter, ax=ax, label=color_column)
    ax.set_xlabel(f"{metadata['axes'][0]['name']} [{metadata['axes'][0]['unit']}]")
    ax.set_ylabel(f"{metadata['axes'][1]['name']} [{metadata['axes'][1]['unit']}]")
    ax.set_title(title or f"FES with samples colored by {color_column}")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output
