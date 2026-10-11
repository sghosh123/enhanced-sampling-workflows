"""Free-energy-surface serialization and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConfigurationError


def openmm_axis_values(minimum: float, maximum: float, grid_width: int) -> np.ndarray:
    """Coordinate values matching OpenMM Metadynamics' documented grid convention.

    OpenMM reports axis point ``i`` as ``min + i*(max-min)/grid_width``.
    The maximum is therefore the upper boundary, not the last grid point.
    """
    if grid_width < 2 or maximum <= minimum:
        raise ConfigurationError("Axis requires grid_width >= 2 and maximum > minimum.")
    return float(minimum) + np.arange(grid_width, dtype=float) * (
        float(maximum) - float(minimum)
    ) / int(grid_width)


def build_fes_metadata(
    fes: np.ndarray,
    collective_variables: list[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(fes)
    if values.ndim != len(collective_variables):
        raise ConfigurationError(
            f"FES dimensionality ({values.ndim}) does not match CV count ({len(collective_variables)})."
        )
    if any(values.shape[i] < 2 for i in range(values.ndim)):
        raise ConfigurationError("Every FES axis must contain at least two grid points.")
    axes = []
    for index, cv in enumerate(collective_variables):
        axes.append({
            "name": cv["name"],
            "unit": cv.get("unit", "nm" if cv.get("kind") == "distance" else "rad"),
            "min_value": float(cv["min_value"]),
            "max_value": float(cv["max_value"]),
            "grid_width": int(values.shape[index]),
            "coordinates": "min + i * (max - min) / grid_width",
        })
    result: dict[str, Any] = {
        "schema_version": 1,
        "fes_shape": list(values.shape),
        "energy_unit": "kJ/mol",
        "axes": axes,
    }
    if extra:
        result["extra"] = extra
    return result


def save_fes_bundle(
    fes: np.ndarray,
    output_prefix: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Write an FES ``.npy`` and adjacent JSON metadata file."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npy_path = prefix if prefix.suffix == ".npy" else prefix.with_suffix(".npy")
    json_path = npy_path.with_suffix(".json")
    values = np.asarray(fes, dtype=float)
    if values.ndim < 1 or values.ndim > 3:
        raise ConfigurationError("Only 1D, 2D, and 3D FES arrays are supported.")
    if list(values.shape) != metadata.get("fes_shape"):
        raise ConfigurationError("Metadata fes_shape does not match the supplied FES array.")
    np.save(npy_path, values)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return npy_path, json_path


def load_fes_bundle(npy_path: str | Path, metadata_path: str | Path | None = None):
    """Load an FES and metadata, validating shape consistency."""
    path = Path(npy_path)
    if metadata_path is None:
        metadata_path = path.with_suffix(".json")
    values = np.load(path, allow_pickle=False)
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if list(values.shape) != metadata.get("fes_shape"):
        raise ConfigurationError("Loaded FES shape does not match the metadata.")
    if len(metadata.get("axes", [])) != values.ndim:
        raise ConfigurationError("Metadata axis count does not match the FES dimensionality.")
    return values, metadata


def _plotting_imports():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "FES plotting requires matplotlib. Install with: "
            "python -m pip install 'enhanced-sampling-workflows[plot]'"
        ) from exc
    return plt


def _axes_for(values: np.ndarray, metadata: dict[str, Any]) -> list[np.ndarray]:
    axes = metadata.get("axes", [])
    if len(axes) != values.ndim:
        raise ConfigurationError("Metadata axis count does not match the FES dimensionality.")
    coordinates = []
    for i, axis in enumerate(axes):
        if int(axis.get("grid_width", -1)) != values.shape[i]:
            raise ConfigurationError(f"Metadata grid width does not match axis {i} shape.")
        coordinates.append(openmm_axis_values(axis["min_value"], axis["max_value"], values.shape[i]))
    return coordinates


def plot_fes(
    fes: np.ndarray,
    metadata: dict[str, Any],
    output_path: str | Path,
    *,
    levels: int = 40,
    title: str | None = None,
) -> Path:
    """Plot a 1D or 2D FES from its axis metadata."""
    plt = _plotting_imports()
    values = np.asarray(fes, dtype=float)
    if values.ndim not in (1, 2):
        raise ConfigurationError("plot_fes supports 1D or 2D surfaces; use plot_fes_slices for 3D.")
    coords = _axes_for(values, metadata)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if values.ndim == 1:
        ax.plot(coords[0], values)
        ax.set_xlabel(f"{metadata['axes'][0]['name']} [{metadata['axes'][0]['unit']}]")
        ax.set_ylabel(f"Free energy [{metadata.get('energy_unit', 'kJ/mol')}]")
    else:
        contour = ax.contourf(coords[0], coords[1], values.T, levels=levels)
        bar = fig.colorbar(contour, ax=ax)
        bar.set_label(f"Free energy [{metadata.get('energy_unit', 'kJ/mol')}]")
        ax.set_xlabel(f"{metadata['axes'][0]['name']} [{metadata['axes'][0]['unit']}]")
        ax.set_ylabel(f"{metadata['axes'][1]['name']} [{metadata['axes'][1]['unit']}]")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def plot_fes_slices(
    fes: np.ndarray,
    metadata: dict[str, Any],
    output_dir: str | Path,
    *,
    slice_axis: int = 2,
    number_of_slices: int = 8,
    levels: int = 40,
) -> list[Path]:
    """Plot 2D slices from a 3D FES by holding one CV grid index fixed."""
    values = np.asarray(fes, dtype=float)
    if values.ndim != 3:
        raise ConfigurationError("plot_fes_slices requires a 3D FES array.")
    if slice_axis not in (0, 1, 2) or number_of_slices < 1:
        raise ConfigurationError("slice_axis must be 0, 1, or 2 and number_of_slices must be >= 1.")
    _axes_for(values, metadata)
    plt = _plotting_imports()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    remaining = [i for i in range(3) if i != slice_axis]
    fixed_axis = metadata["axes"][slice_axis]
    fixed_values = openmm_axis_values(
        fixed_axis["min_value"], fixed_axis["max_value"], values.shape[slice_axis]
    )
    indices = np.unique(np.linspace(0, values.shape[slice_axis] - 1,
                                   min(number_of_slices, values.shape[slice_axis])).round().astype(int))
    paths: list[Path] = []
    for idx in indices:
        slab = np.take(values, int(idx), axis=slice_axis)
        # The remaining axis order follows numpy's array order, matching FES metadata.
        c1 = openmm_axis_values(metadata["axes"][remaining[0]]["min_value"],
                                metadata["axes"][remaining[0]]["max_value"], slab.shape[0])
        c2 = openmm_axis_values(metadata["axes"][remaining[1]]["min_value"],
                                metadata["axes"][remaining[1]]["max_value"], slab.shape[1])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        contour = ax.contourf(c1, c2, slab.T, levels=levels)
        fig.colorbar(contour, ax=ax, label=f"Free energy [{metadata.get('energy_unit', 'kJ/mol')}]")
        a1, a2 = metadata["axes"][remaining[0]], metadata["axes"][remaining[1]]
        ax.set_xlabel(f"{a1['name']} [{a1['unit']}]")
        ax.set_ylabel(f"{a2['name']} [{a2['unit']}]")
        ax.set_title(f"{fixed_axis['name']} = {fixed_values[int(idx)]:.4g} {fixed_axis['unit']}")
        fig.tight_layout()
        target = output / f"slice_axis{slice_axis}_index{int(idx):03d}.png"
        fig.savefig(target, dpi=300)
        plt.close(fig)
        paths.append(target)
    return paths
