"""Collective-variable extraction from trajectory files using MDAnalysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ConfigurationError, validate_trajectory_config


def _mda_imports():
    try:
        import MDAnalysis as mda
        from MDAnalysis.lib.distances import distance_array, minimize_vectors
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Trajectory CV extraction requires MDAnalysis. Install with: "
            "python -m pip install 'enhanced-sampling-workflows[trajectory]'"
        ) from exc
    return mda, distance_array, minimize_vectors


def _selection_string(atom: dict[str, Any]) -> str:
    if atom.get("selection"):
        return str(atom["selection"])
    parts: list[str] = []
    if atom.get("segid") is not None:
        parts.append(f"segid {atom['segid']}")
    elif atom.get("chain_id") is not None:
        parts.append(f"chainID {atom['chain_id']}")
    elif atom.get("chain") is not None:
        parts.append(f"chainID {atom['chain']}")
    parts.extend((f"resid {atom['resid']}", f"name {atom['name']}"))
    return " and ".join(parts)


def _single_atom(universe, atom: dict[str, Any], *, cv_name: str, atom_number: int):
    selection = _selection_string(atom)
    group = universe.select_atoms(selection)
    if len(group) != 1:
        raise ConfigurationError(
            f"CV '{cv_name}' atom {atom_number} selection {selection!r} matched "
            f"{len(group)} atoms; expected exactly one. Check residue numbering, segid/chain, and atom name."
        )
    return group


def _get_box(ts, *, pbc: bool, cv_name: str):
    if not pbc:
        return None
    dimensions = ts.dimensions
    if dimensions is None or len(dimensions) < 6 or not np.all(np.asarray(dimensions[:3]) > 0):
        raise ConfigurationError(
            f"CV '{cv_name}' requests periodic-boundary correction, but this frame has no valid box. "
            "Set pbc=false only when the system and coordinate convention justify non-periodic distances."
        )
    return dimensions


def _evaluate_cv(cv: dict[str, Any], groups: list[Any], ts, distance_array, minimize_vectors) -> float:
    box = _get_box(ts, pbc=cv["pbc"], cv_name=cv["name"])
    coordinates = [group.positions[0].astype(np.float64, copy=False) for group in groups]
    if cv["kind"] == "distance":
        distance_angstrom = distance_array(
            coordinates[0][None, :], coordinates[1][None, :], box=box
        )[0, 0]
        return float(distance_angstrom * 0.1)  # MDAnalysis coordinates are in Å; report nm.

    # The input order is (p1, p2, p3, p4); the angle is between p1-p2 and p3-p4.
    vector_1 = coordinates[0] - coordinates[1]
    vector_2 = coordinates[2] - coordinates[3]
    if box is not None:
        vector_1, vector_2 = minimize_vectors(np.asarray([vector_1, vector_2]), box)
    norm_1 = float(np.linalg.norm(vector_1))
    norm_2 = float(np.linalg.norm(vector_2))
    if norm_1 <= 0 or norm_2 <= 0:
        raise ConfigurationError(f"CV '{cv['name']}' has a zero-length vector in a frame.")
    cosine = float(np.dot(vector_1, vector_2) / (norm_1 * norm_2))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compute_cv_table(config: dict[str, Any]) -> pd.DataFrame:
    """Compute configured CVs for every frame of each configured trajectory.

    Output distances are in nm and vector angles are in radians. Frame numbers are
    zero-based local frame indices; ``trajectory_id`` disambiguates source files.
    """
    mda, distance_array, minimize_vectors = _mda_imports()
    trajectories, cvs = validate_trajectory_config(config)
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        try:
            universe = mda.Universe(trajectory["topology"], trajectory["trajectory"])
        except Exception as exc:
            raise RuntimeError(
                f"Could not load trajectory {trajectory['id']!r}: topology={trajectory['topology']!r}, "
                f"trajectory={trajectory['trajectory']!r}: {exc}"
            ) from exc
        groups_by_cv = {
            cv["name"]: [
                _single_atom(universe, atom, cv_name=cv["name"], atom_number=j + 1)
                for j, atom in enumerate(cv["atoms"])
            ]
            for cv in cvs
        }
        n_frames = len(universe.trajectory)
        for frame_index, ts in enumerate(universe.trajectory):
            row: dict[str, Any] = {
                "trajectory_id": str(trajectory["id"]),
                "frame": int(frame_index),
                "time_ps": float(ts.time) if np.isfinite(ts.time) else np.nan,
            }
            for cv in cvs:
                row[cv["name"]] = _evaluate_cv(
                    cv, groups_by_cv[cv["name"]], ts, distance_array, minimize_vectors
                )
            rows.append(row)
        if n_frames == 0:
            raise RuntimeError(f"Trajectory {trajectory['id']!r} contains no frames.")
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No trajectory frames were processed.")
    return result
