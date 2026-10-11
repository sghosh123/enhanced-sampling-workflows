# Changelog

## 0.1.0a1 — initial alpha portfolio release

- Added reusable MDAnalysis collective-variable extraction with explicit trajectory IDs and zero-based frame indices.
- Added 2D umbrella-window generation with explicit empty-window policies, nearest-seed cutoffs, offsets, and reuse accounting.
- Added FES bundle metadata and plotting helpers for 1D/2D surfaces and 3D slices.
- Added FES comparison metrics and a heuristic convergence tracker that enforces maximum-delta, RMSE, basin-location, and consecutive-comparison criteria.
- Added optional OpenMM metadynamics and single-window harmonic umbrella runners with JSON configuration.
- Added example configurations, synthetic data demo, documentation, unit tests, and GitHub Actions test workflow.
- Marked OpenMM runtime validation and production-system validation as outstanding; see `docs/methods-and-validation.md`.

## 0.1.0a2

- Added connected-plateau-aware, spatially separated FES local-minimum detection.
- Added representative-frame selection with a maximum CV-space offset; distant frames remain explicitly unrepresented.
- Added optional whole-system representative PDB export and 2D FES overlays colored by mechanistic observables.
