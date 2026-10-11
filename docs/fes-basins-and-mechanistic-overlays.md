# FES basin representatives and mechanistic overlays

## Minima detection

`detect_fes_minima` detects strict local minima in 1D/2D/3D FES data, collapses adjacent cells from flat minima, sorts by energy and applies grid-index-space minimum separation. `min_separation_grid` is measured in Euclidean grid cells, not physical CV units. A completely flat surface intentionally yields no minima.

## Frame representatives

`select_basin_representatives` maps CV coordinates to grid indices using the axis ranges and OpenMM metadata convention `min + i*(max-min)/grid_width`. A representative must lie within `basin_radius_grid` and satisfy `max_sample_distance_grid`. Each frame identity is used at most once. Unrepresented minima are retained in the output rather than assigned an arbitrarily distant frame.

Optional PDB export requires MDAnalysis and the same topology/trajectory source mapping used for CV generation. The current config schema maps one trajectory ID to one trajectory file; where data have multiple file segments that reuse local frame numbers, create a source-specific mapping before extraction. PDB files contain the complete current Universe, not a chemically validated or automatically re-imaged/whole structure.

## Mechanistic overlays

`plot-fes-overlay` draws sample points on top of a 2D FES and colors them by a numeric observable. It performs no reweighting and does not compute probabilities from a biased trajectory. Supply correctly reweighted samples or clearly label the plot as a projection of sampled observations; FES axis metadata and units must match the point table.

Minima and structure selection are descriptive helpers. They do not identify transition states or establish Cu-S bond formation on their own.
