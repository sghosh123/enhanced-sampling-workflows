"""Configuration loading and validation helpers.

The on-disk configuration is intentionally JSON so the workflow is easy to archive
alongside simulation outputs and parse without another dependency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SUPPORTED_CV_KINDS = {"distance", "vector_angle"}


class ConfigurationError(ValueError):
    """Raised when a workflow configuration is incomplete or inconsistent."""


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from *path* and validate its root type."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("The configuration JSON root must be an object.")
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.parent)
    return data


def validate_atom_reference(atom: Any, *, context: str) -> None:
    if not isinstance(atom, dict):
        raise ConfigurationError(f"{context} must be an object describing an atom.")
    if "selection" in atom and str(atom["selection"]).strip():
        return
    if not ("segid" in atom or "chain" in atom or "chain_id" in atom):
        raise ConfigurationError(
            f"{context} needs 'segid', 'chain', 'chain_id', or a direct 'selection'."
        )
    for key in ("resid", "name"):
        if key not in atom or str(atom[key]).strip() == "":
            raise ConfigurationError(f"{context} is missing required field '{key}'.")


def validate_collective_variables(variables: Any) -> list[dict[str, Any]]:
    """Validate common CV definitions used by trajectory analysis and OpenMM runs.

    A distance uses two atom references and is reported in nm. A vector angle uses
    four ordered atom references and is reported in radians. For vector_angle, the
    atom order is (point_1, point_2, point_3, point_4), giving the angle between
    (point_1 - point_2) and (point_3 - point_4).
    """
    if not isinstance(variables, list) or not variables:
        raise ConfigurationError("'collective_variables' must be a non-empty list.")
    seen_names: set[str] = set()
    result: list[dict[str, Any]] = []
    for i, original in enumerate(variables):
        if not isinstance(original, dict):
            raise ConfigurationError(f"CV entry {i} must be an object.")
        cv = dict(original)
        name = str(cv.get("name", "")).strip()
        if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ConfigurationError(
                f"CV entry {i} needs a Python/OpenMM-safe name (letters, digits, underscores)."
            )
        if name in seen_names:
            raise ConfigurationError(f"Duplicate collective-variable name: {name}")
        seen_names.add(name)
        kind = cv.get("kind")
        if kind not in SUPPORTED_CV_KINDS:
            raise ConfigurationError(
                f"CV '{name}' has unsupported kind {kind!r}; use one of {sorted(SUPPORTED_CV_KINDS)}."
            )
        atoms = cv.get("atoms")
        needed = 2 if kind == "distance" else 4
        if not isinstance(atoms, list) or len(atoms) != needed:
            raise ConfigurationError(f"CV '{name}' ({kind}) requires exactly {needed} atom references.")
        for j, atom in enumerate(atoms):
            validate_atom_reference(atom, context=f"CV '{name}' atom {j + 1}")
        for key in ("min_value", "max_value", "bias_width"):
            try:
                number = float(cv[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigurationError(f"CV '{name}' requires numeric '{key}'.") from exc
            if not math.isfinite(number):
                raise ConfigurationError(f"CV '{name}' field '{key}' must be finite.")
            cv[key] = number
        if cv["max_value"] <= cv["min_value"]:
            raise ConfigurationError(f"CV '{name}' must have max_value > min_value.")
        if cv["bias_width"] <= 0:
            raise ConfigurationError(f"CV '{name}' bias_width must be positive.")
        grid_width = cv.get("grid_width")
        if grid_width is not None and (not isinstance(grid_width, int) or grid_width < 2):
            raise ConfigurationError(f"CV '{name}' grid_width must be an integer >= 2.")
        cv["periodic"] = bool(cv.get("periodic", False))
        cv["pbc"] = bool(cv.get("pbc", True))
        cv["unit"] = cv.get("unit", "nm" if kind == "distance" else "rad")
        if cv["periodic"]:
            raise ConfigurationError(
                f"CV '{name}' is a {kind}, which is not a wrapped periodic variable; set periodic=false. "
                "The separate pbc flag controls minimum-image geometry."
            )
        if kind == "distance" and cv["unit"] != "nm":
            raise ConfigurationError(f"Distance CV '{name}' must use nm in this toolkit.")
        if kind == "distance" and cv["min_value"] < 0:
            raise ConfigurationError(f"Distance CV '{name}' cannot have a negative minimum value.")
        if kind == "vector_angle" and cv["unit"] not in {"rad", "radian", "radians"}:
            raise ConfigurationError(f"Vector-angle CV '{name}' must use radians.")
        if kind == "vector_angle" and (cv["min_value"] < 0 or cv["max_value"] > math.pi):
            raise ConfigurationError(f"Vector-angle CV '{name}' must lie within [0, pi] radians.")
        result.append(cv)
    return result


def resolve_config_path(value: str | Path, config_dir: str | Path) -> Path:
    """Resolve an input path relative to its JSON configuration file."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(config_dir) / path).resolve()


def validate_trajectory_config(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return validated trajectory and CV entries for MDAnalysis workflows."""
    trajectories = config.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ConfigurationError("'trajectories' must be a non-empty list.")
    checked_traj: list[dict[str, Any]] = []
    ids: set[str] = set()
    base = config.get("_config_dir", ".")
    for i, original in enumerate(trajectories):
        if not isinstance(original, dict):
            raise ConfigurationError(f"Trajectory entry {i} must be an object.")
        item = dict(original)
        tid = str(item.get("id", "")).strip()
        if not tid:
            raise ConfigurationError(f"Trajectory entry {i} is missing a non-empty 'id'.")
        if tid in ids:
            raise ConfigurationError(f"Duplicate trajectory id: {tid}")
        ids.add(tid)
        for key in ("topology", "trajectory"):
            if not item.get(key):
                raise ConfigurationError(f"Trajectory '{tid}' is missing '{key}'.")
            item[key] = str(resolve_config_path(item[key], base))
        checked_traj.append(item)
    cvs = validate_collective_variables(config.get("collective_variables"))
    return checked_traj, cvs
