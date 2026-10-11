# Contributing

## Development setup

```bash
python -m pip install -e '.[dev,trajectory,plot]'
python -m pytest -q
```

OpenMM is optional for analysis-only changes. If you change `openmm_runner.py`, compile the module and run short pilot simulations against a small, validated OpenMM system before claiming runtime validation. Do not put production trajectories, private force-field files, or generated multi-gigabyte outputs into commits.

## Scientific changes

A pull request that changes a CV definition, units, grid convention, PBC treatment, or convergence metric should include a short methods note and tests against known values. Any change to the heuristic convergence test should explain its expected false-positive/false-negative behavior.

## Outputs and reproducibility

Keep trajectory identity plus local frame index in generated tables. Include CV names and units in column names or metadata. New configuration keys should be documented in README examples and validated before launching a long calculation.
