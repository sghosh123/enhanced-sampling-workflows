# Source-inspired corrections and design decisions

This repository generalizes the reusable parts of project scripts found in the uploaded research archive. It does not claim the original production simulations can be reconstructed from this repository alone.

## Umbrella-window assignment

The original grid generator selected a globally nearest frame when a window contained no frame and labelled it `nearest_empty`. It did not report the CV-space distance from that frame to the requested center or cap that distance. The generalized implementation:

- defaults to `skip` for empty bins;
- supports `nearest` only as an explicit policy;
- supports a maximum scaled distance for nearest-frame fallback;
- records CV offsets, scaled distance, status, and how often a seed frame is reused;
- uses explicit trajectory IDs and zero-based local frame indices;
- validates that the requested range is an integer multiple of grid spacing;
- includes the maximum boundary in the final bin, preventing a CV value exactly at the upper bound from being silently dropped.

A nearest structure is only a seed candidate, not evidence that the window is equilibrated. Equilibration/overlap between windows still needs to be checked.

## FES convergence diagnostics

Earlier project scripts defined a maximum absolute FES difference tolerance but did not enforce it in the final condition. Some implementations selected the lowest `k` individual grid cells rather than local minima, which could count adjacent cells in one basin several times. The generalized tracker:

- normalizes each surface by subtracting its finite global minimum;
- compares only grid points finite in both arrays;
- evaluates maximum absolute difference and RMSE independently;
- collapses neighboring cells on flat minima plateaus;
- compares locations of top-ranked local minima by minimum-total-distance matching;
- requires every configured threshold to pass for a configured number of consecutive comparisons.

These metrics remain heuristic. Their validity depends on CV ranges, grid resolution, block length, sampling coverage, and the physical system. Matching local minima can be unstable when minima are nearly degenerate or a basin appears/disappears; review the surfaces and independent observables instead of treating the output as an automatic proof of convergence.

## Units and output metadata

The source workflows mix Å and nm between trajectory analysis and OpenMM CVs. This package reports distances in nm and vector angles in radians, and it writes units into configuration and FES metadata. The original 2D CV code also used angle labels inconsistently in plots/output. The generalized metadata takes CV names, units, grid ranges, and grid widths from configuration.

OpenMM's FES grid coordinates follow its API convention: `minValue + i*(maxValue-minValue)/gridWidth`, with `i` from zero to `gridWidth - 1`. This avoids treating the upper boundary as an included grid point when it is not.

## OpenMM forces and system-specific terms

The original study scripts add copper-specific harmonic bonds and atom-index-based positional restraints. These are intentionally not copied into a generic runner because the numeric atom indices and force constants depend on the exact prepared system and force-field parameterization. The general runner supports configured pair distances and a four-atom vector angle; users must separately implement and validate every system-specific force before running production simulations.

The OpenMM vector-angle CV uses the ordered atoms `(p1, p2, p3, p4)` for vectors `(p1-p2)` and `(p3-p4)`. It reconstructs their dot product from pair distances and uses OpenMM's built-in `distance(p_i,p_j)` function with the force's PBC option enabled, rather than taking raw Cartesian coordinate differences. The pair-distance reconstruction still assumes the minimum-image pair distances represent a consistent local geometry, so compare it against MDAnalysis on frames near periodic boundaries and confirm the vector segments are correctly imaged.

## Runtime output and errors

The original run-script generator wrote a run script for only the first umbrella window. The generalized umbrella runner launches one window per command and records its target centers in a summary, so it can be called once per seed/window by a shell loop or scheduler array. It does not infer missing force-field XML files or assume a CUDA device exists; paths and platform configuration must be checked on the target machine.
