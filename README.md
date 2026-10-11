# Enhanced Sampling Workflows

**Development release:** `0.1.0a1`  

**Purpose:** reusable utilities for collective-variable analysis, umbrella-window preparation, OpenMM metadynamics/umbrella runs, and free-energy-surface (FES) diagnostics.

This repository generalizes methods from a copper-trafficking protein–protein simulation workflow into a small package with explicit units, configuration files, tests, and reproducible outputs. It is designed to complement—not duplicate—the more general `copper-trafficking-mechanism` analysis repository.

> **Research-use warning:** this is an alpha integration draft. Its automated tests use synthetic data; the OpenMM runners have not been runtime-tested here because OpenMM is not installed in the build environment. Before production use, validate atom selections, custom force-field compatibility, periodic-boundary handling, restraints, and output CVs on a short pilot simulation. The templates deliberately do not recreate the copper-specific harmonic bond and positional restraints present in some original study scripts.

## Capabilities

- Compute pair distances (nm) and four-atom vector angles (radians) across one or more MDAnalysis trajectories.
- Preserve trajectory ID plus zero-based local frame index in a single CV table.
- Build a regular 2D umbrella window grid and assign source frames, while reporting offsets and frame reuse.
- Export assigned frames as per-window PDB seeds.
- Run configurable OpenMM well-tempered metadynamics with 1–3 distance/vector-angle CVs, block-wise FES output, and optional heuristic convergence monitoring.
- Run one harmonic umbrella window for one or more distance CVs.
- Compare successive FES arrays, detect local minima without counting every adjacent grid cell as a separate basin, and export FES metadata/plots.

## Install

Base utilities:

```bash
python -m pip install -e .
```

All optional workflow dependencies (MDAnalysis, matplotlib, and OpenMM):

```bash
python -m pip install -e '.[all,dev]'
```

For GPU simulations, install an OpenMM build compatible with the target machine, CUDA driver, and available platform. Installing OpenMM in a plotting or analysis environment does not guarantee that CUDA is available there.

## Workflow A: compute CVs from trajectories

Copy `configs/trajectory_cv.example.json` and update the paths. The example uses the residue/segment convention from the original study; verify it against the exact topology before analysis.

```bash
esw compute-cvs configs/trajectory_cv.example.json --output cv_table.csv
```

The output includes `trajectory_id`, `frame` (zero-based within that file), `time_ps`, and one column per configured CV. Distances are converted from MDAnalysis' Å coordinates to nm; angles are radians. Atom selections must match exactly one atom. Periodic CVs require valid box dimensions.

## Workflow B: generate umbrella-window assignments

Use the CV table to define a rectangular window grid. The following matches the study's two-distance grid (1.6–3.0 nm and 0.2–1.0 nm, 0.05 nm spacing):

```bash
esw assign-windows cv_table.csv \
  --cv1 d_A1X1 --cv2 d_A2X2 \
  --cv1-range 1.6 3.0 --cv2-range 0.2 1.0 \
  --spacing 0.05 --empty-policy skip \
  --output US_windows_summary.tsv
```

`skip` is the safe default: empty bins have no assigned seed. To follow the original workflow's nearest-frame fallback while keeping it auditable, use `--empty-policy nearest --max-seed-distance 2.0`. That maximum is in scaled grid-spacing units, not nm. Every nearest fallback records its CV offsets, scaled distance, and `seed_reuse_count`; inspect these before launching production windows.

Export seed structures for assigned rows:

```bash
esw export-seeds US_windows_summary.tsv configs/trajectory_cv.example.json \
  --outdir US_windows
```

Do not use the placeholder topology/trajectory paths in the example unedited. `export-seeds` refuses to overwrite existing seed PDBs unless `--overwrite` is supplied.

## Workflow C: OpenMM metadynamics

Copy and edit either `configs/metadynamics_3cv.example.json` or `configs/metadynamics_distance_angle.example.json`. Verify force-field XML files, PDB topology, nonbonded settings, atom identifiers, box vectors, and any metal-specific/custom restraints first. Run a brief pilot on CPU before using CUDA or long sampling.

```bash
esw run-metadynamics configs/metadynamics_3cv.example.json
```

The runner records `cv_history.csv`, `state.tsv`/`production.dcd` when enabled, block-wise `fes_block_*.npy` plus JSON metadata, a final FES bundle, convergence history, and `run_summary.json`. `save_frequency_steps` must be a multiple of `hill_frequency_steps`, and block lengths are validated against hill deposition cadence.

The built-in convergence tracker checks **all** configured thresholds: maximum absolute change after shifting each FES minimum to zero, root-mean-square change, and displacement of matched local minima in grid-index units. It also requires the requested number of consecutive passing comparisons. This is a heuristic stability signal, not proof of thermodynamic convergence; check sampling, hysteresis, multiple replicas, and physically meaningful observables as well.

## Workflow D: one umbrella window

Use `configs/umbrella_distances.example.json` and provide window centers in nm:

```bash
esw run-umbrella configs/umbrella_distances.example.json \
  --centers d_A1X1=2.2,d_A2X2=0.6 --label x2.20_y0.60 \
  --pdb US_windows/example_window/seed_frame.pdb
```

The runner is for **one window per command** and distance CVs only. Its shared force constant is in kJ mol⁻¹ nm⁻². For a real umbrella campaign, validate the restraint centers, force constants, equilibration and production lengths, seed reuse, and output CVs. This runner does not automatically apply study-specific copper coordination or positional-restraint forces.

## Workflow E: compare and plot FES arrays

An FES bundle is an `.npy` array plus a matching JSON sidecar with axis labels, units, grid widths, and ranges. For bundle files:

```bash
esw compare-fes fes_block_00010.npy fes_block_00011.npy --minima-count 3
esw plot-fes fes_final.npy --output fes_final.png
esw plot-fes-slices fes_3d.npy --output-dir fes_slices --slice-axis 2 --number-of-slices 8
```

`compare-fes` accepts raw arrays too, but the shapes must match. FES arrays are shifted by their respective finite minima before comparing energy differences. Grid axis coordinates follow OpenMM's documented convention `min + i*(max-min)/grid_width`, so the final point is below the upper boundary.

## Units and indexing conventions

| Quantity | Convention |
|---|---|
| MDAnalysis input coordinates | Å |
| Distance CV columns and umbrella centers | nm |
| Vector angle CVs | radians |
| OpenMM energy/FES | kJ/mol |
| Umbrella spring constant | kJ mol⁻¹ nm⁻² |
| Trajectory frame | zero-based local frame index |
| Trajectory identity | explicit `trajectory_id` string |
| `seed_distance_grid_units` | Euclidean CV offset scaled by grid spacing |

CV unit conventions are explicit by design. Do not compare unlabelled distances from different scripts without confirming units.

## Testing

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Tests cover configuration validation, grid assignment, nearest-seed limits, reuse reporting, FES normalization/comparison, convergence criteria, metadata shape checks, and command-line parsing. Actual production-system/OpenMM runtime validation remains required in the research environment.

## Provenance and scope

The design was informed by project scripts for OpenMM metadynamics (including 2D distance/angle and 3D distance CV workflows), FES evolution/convergence analysis, and umbrella grid generation. The original project scripts contain system-specific values, hard-coded paths, and implementation defects; this repository rewrites the reusable parts rather than shipping those scripts unchanged. No production trajectory, topology, force-field XML, or generated FES files are bundled.

See `docs/source-inspired-corrections.md`, `docs/methods-and-validation.md`, and `LICENSE-STATUS.md` before publishing.
