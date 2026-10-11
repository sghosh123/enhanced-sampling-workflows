# OpenMM run templates

The JSON configurations in `../../configs/` are editable templates—not ready-to-run production input files. Their PDB and custom force-field paths are placeholders. Replace them with files for a validated molecular system and verify all atom identifiers against the precise topology.

## Metadynamics

```bash
esw run-metadynamics ../../configs/metadynamics_3cv.example.json
```

## One umbrella window

```bash
esw run-umbrella ../../configs/umbrella_distances.example.json \
  --centers d_A1X1=2.2,d_A2X2=0.6 --label x2.20_y0.60 \
  --pdb /absolute/path/to/seed_frame.pdb
```

## Important model-specific work that remains

The original project scripts included Cu(I)-related harmonic bonds and a fixed atom-index positional restraint list. This generic runner intentionally does not copy those terms because their correctness depends on the exact parameterized system, topology order, and units. Add them in a system-specific extension and test energies/forces before conducting a production run. Confirm chain/residue references, PBC behavior, cutoff/box compatibility, restart behavior, and output CVs on a short pilot. Use a validated OpenMM build and verify the chosen platform on the target machine.
