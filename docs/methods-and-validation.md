# Methods and validation checklist

## Current automated tests

Automated tests use small synthetic CV tables and NumPy arrays. They validate deterministic mechanics, not physical correctness for a specific copper-trafficking system.

Tested by unit tests:

- strict validation of CV atom counts, names, bounds, widths, and units;
- rectangular grid construction and upper-bound bin handling;
- occupied-window frame selection;
- explicit empty-window policies and nearest-seed distance cutoffs;
- frame-reuse accounting;
- FES minimum normalization, plateau-aware local-minimum identification, and shape checks;
- independent maximum-delta and RMSE thresholds plus consecutive-block convergence logic;
- FES metadata consistency and OpenMM-style axis coordinate convention;
- command-line parser registration.

Not validated in this build environment:

- runtime import or execution of OpenMM, because OpenMM is not installed in the build environment;
- force-field template matching or custom copper coordination parameters;
- CUDA/OpenCL execution and GPU-specific settings;
- PBC correctness of the OpenMM vector-angle expression for production systems;
- correctness of chain/residue indexing against the user's exact production PDB/PSF;
- numerical FES results against original simulation outputs;
- overlap, effective sample sizes, replica consistency, reweighting, or thermodynamic convergence.

## Before production simulations

1. Install the repository in the target analysis/simulation environment and run all tests.
2. Verify the same atom references independently in the prepared topology and in MDAnalysis/OpenMM.
3. Compute a short CV table from 10–100 frames and compare distances/angles with an independent viewer or analysis implementation.
4. For PBC-enabled CVs, test frames where one or more atom pairs cross a periodic boundary.
5. Create the OpenMM System and evaluate energies with the chosen platform before adding metadynamics; verify force-field residue templates and all required Cu(I) terms.
6. Add project-specific bond/position restraints only after mapping and unit-checking the intended atoms and force constants.
7. Run a short metadynamics pilot and inspect the deposition cadence, CV ranges, output time, FES metadata, and restart/bias files.
8. For umbrella sampling, confirm every target center and seed frame, minimize/equilibrate consistently, check neighboring-window overlap, and use WHAM/MBAR with decorrelation/error analysis before estimating free energies.
9. Treat the convergence tracker as a diagnostic trigger. Validate stable sampling with additional blocks/replicas and compare physically meaningful distributions.
10. Archive the exact config JSON, force-field input versions, software versions, random seeds, platform properties, and topology hashes with each production dataset.

## Suggested checks for this research system

- Trajectory IDs should follow the intended mapping `T4`, `T7`, `T8`, `T26`, not incidental file ordering.
- A1–X1: resid 56 CA in segment/chain A to resid 55 CA in segment/chain B.
- A2–X2: resid 63 CA in A to resid 62 CA in B.
- B2–Y2: resid 26 CA in A to resid 25 CA in B.
- B–Y vector angle: use ordered atoms `(A:26 CA, A:18 CA, B:25 CA, B:18 CA)` for vectors `(B2-B1)` and `(Y2-Y1)`.
- Distance outputs should be in nm for OpenMM compatibility; original MDAnalysis coordinates are in Å.
- For a 2D umbrella grid, target windows span `d_A1X1=1.6–3.0 nm`, `d_A2X2=0.2–1.0 nm`, at `0.05 nm` spacing. Empty windows should be flagged and reviewed before use.
