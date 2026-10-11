"""Generate a tiny synthetic FES bundle and compare two sampling snapshots.

Run from the repository root after installing the optional plotting extra:
    python examples/synthetic_fes_demo.py --outdir demo_output
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from enhanced_sampling_workflows.convergence import compare_fes
from enhanced_sampling_workflows.fes import build_fes_metadata, plot_fes, save_fes_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="demo_output")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(1.5, 3.0, 60, endpoint=False)
    y = np.linspace(0.2, 1.0, 40, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    fes_1 = 8.0 * (xx - 2.15) ** 2 + 11.0 * (yy - 0.52) ** 2
    fes_1 += 2.0 * np.sin(6.0 * xx) * np.cos(5.0 * yy)
    fes_2 = fes_1 + 0.15 * np.sin(9.0 * xx) * np.sin(7.0 * yy)
    cvs = [
        {"name": "d_A1X1", "kind": "distance", "unit": "nm", "min_value": 1.5, "max_value": 3.0},
        {"name": "d_A2X2", "kind": "distance", "unit": "nm", "min_value": 0.2, "max_value": 1.0},
    ]
    metadata = build_fes_metadata(fes_1, cvs, extra={"demonstration_only": True})
    p1, _ = save_fes_bundle(fes_1, outdir / "fes_snapshot_1", metadata)
    metadata_2 = build_fes_metadata(fes_2, cvs, extra={"demonstration_only": True})
    p2, _ = save_fes_bundle(fes_2, outdir / "fes_snapshot_2", metadata_2)
    plot_fes(fes_2, metadata_2, outdir / "fes_snapshot_2.png", title="Synthetic FES (not simulation data)")
    comparison = compare_fes(np.load(p1), np.load(p2), minima_count=1)
    print("Synthetic-only example; no molecular simulation data were used.")
    print(comparison.to_dict())
    print(f"Wrote example outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
