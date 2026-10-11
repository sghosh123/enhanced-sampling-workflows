"""Command-line interface for enhanced-sampling workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_json, validate_trajectory_config
from .basins import detect_fes_minima, select_basin_representatives, plot_fes_overlay
from .convergence import compare_fes
from .fes import load_fes_bundle, plot_fes, plot_fes_slices
from .trajectory import compute_cv_table
from .windows import assign_grid_windows


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    sep = "\t" if source.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(source, sep=sep)


def _write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if target.suffix.lower() in {".tsv", ".tab"} else ","
    frame.to_csv(target, sep=sep, index=False)
    return target


def _parse_centers(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    result = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError("Each center must be written as NAME=VALUE.")
        name, raw = item.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise argparse.ArgumentTypeError(f"Invalid or repeated center name: {name!r}")
        try:
            result[name] = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Center value must be numeric: {raw!r}") from exc
    return result


def _export_seeds(windows_path: str, config_path: str, outdir: str, overwrite: bool) -> Path:
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise RuntimeError(
            "Seed-PDB export requires MDAnalysis. Install with 'pip install "
            "enhanced-sampling-workflows[trajectory]'."
        ) from exc
    windows = _read_table(windows_path)
    config = load_json(config_path)
    trajectories, _cvs = validate_trajectory_config(config)
    id_map = {str(item["id"]): item for item in trajectories}
    required = {"window_id", "trajectory_id", "frame", "status"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Window table is missing columns: {sorted(missing)}")
    output_root = Path(outdir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    universes = {}
    written = 0
    try:
        for row in windows.itertuples(index=False):
            if str(row.status) == "unseeded" or pd.isna(row.frame) or pd.isna(row.trajectory_id):
                continue
            trajectory_id = str(row.trajectory_id)
            if trajectory_id not in id_map:
                raise ValueError(f"Window references unknown trajectory id {trajectory_id!r}.")
            if trajectory_id not in universes:
                item = id_map[trajectory_id]
                universes[trajectory_id] = mda.Universe(item["topology"], item["trajectory"])
            universe = universes[trajectory_id]
            frame = int(row.frame)
            if frame < 0 or frame >= len(universe.trajectory):
                raise ValueError(
                    f"Frame {frame} is outside trajectory {trajectory_id!r} "
                    f"(n_frames={len(universe.trajectory)})."
                )
            window_dir = output_root / str(row.window_id)
            window_dir.mkdir(parents=True, exist_ok=True)
            pdb_path = window_dir / "seed_frame.pdb"
            if pdb_path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing seed file: {pdb_path}")
            universe.trajectory[frame]
            universe.atoms.write(str(pdb_path))
            written += 1
    finally:
        universes.clear()
    print(f"Exported {written} seed PDB files to {output_root}")
    return output_root



def _extract_basin_pdbs(representatives: pd.DataFrame, trajectory_config_path: str, outdir: str, overwrite: bool) -> int:
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise RuntimeError("PDB export requires MDAnalysis. Install with 'pip install enhanced-sampling-workflows[trajectory]'.") from exc
    config = load_json(trajectory_config_path)
    trajectories, _ = validate_trajectory_config(config)
    by_id = {str(item["id"]): item for item in trajectories}
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    cache = {}
    written = 0
    try:
        for row in representatives.itertuples(index=False):
            if row.representative_status != "matched":
                continue
            tid = str(row.representative_trajectory_id)
            if tid not in by_id:
                raise ValueError(f"Representative references unknown trajectory id {tid!r} in trajectory config.")
            item = by_id[tid]
            if tid not in cache:
                cache[tid] = mda.Universe(item["topology"], item["trajectory"])
            universe = cache[tid]
            frame = int(row.representative_frame)
            if frame < 0 or frame >= len(universe.trajectory):
                raise ValueError(f"Representative frame {frame} exceeds trajectory {tid!r} with {len(universe.trajectory)} frames.")
            target_dir = root / str(row.basin_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "representative.pdb"
            if target.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite to replace it.")
            universe.trajectory[frame]
            universe.atoms.write(str(target))
            written += 1
    finally:
        cache.clear()
    return written


def command_extract_basin_frames(args) -> int:
    if args.export_pdb and not args.trajectory_config:
        raise ValueError("--trajectory-config is required together with --export-pdb.")
    values, metadata = load_fes_bundle(args.fes, args.metadata)
    table = _read_table(args.cv_table)
    representatives = select_basin_representatives(
        table, values, metadata, cv_columns=args.cv_columns,
        trajectory_column=args.trajectory_column, segment_column=args.segment_column,
        frame_column=args.frame_column, min_separation_grid=args.min_separation_grid,
        max_minima=args.max_minima, basin_radius_grid=args.basin_radius_grid,
        max_sample_distance_grid=args.max_sample_distance_grid,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    representatives.to_csv(outdir / "basin_representatives.tsv", sep="\t", index=False)
    if args.export_pdb:
        count = _extract_basin_pdbs(representatives, args.trajectory_config, str(outdir / "structures"), args.overwrite)
        print(f"Exported {count} representative PDB files.")
    matched = int((representatives.get("representative_status", pd.Series(dtype=str)) == "matched").sum())
    print(f"Found {len(representatives)} separated minima; representative frames matched for {matched}; table: {(outdir / 'basin_representatives.tsv').resolve()}")
    return 0


def command_plot_fes_overlay(args) -> int:
    values, metadata = load_fes_bundle(args.fes, args.metadata)
    table = _read_table(args.cv_table)
    output = plot_fes_overlay(
        values, metadata, table, cv_columns=args.cv_columns,
        color_column=args.color_by, output_path=args.output,
        max_points=args.max_points, title=args.title,
    )
    print(f"FES overlay written to {output.resolve()}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esw",
        description="Reusable utilities for trajectory CVs, umbrella windows, OpenMM sampling, and FES diagnostics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cvs = sub.add_parser("compute-cvs", help="Compute CVs from configured MDAnalysis trajectories.")
    cvs.add_argument("config", help="Trajectory/CV JSON configuration.")
    cvs.add_argument("--output", default="cv_table.csv", help="Output CSV or TSV based on file extension.")

    windows = sub.add_parser("assign-windows", help="Assign CV-table frames to a rectangular 2D umbrella grid.")
    windows.add_argument("cv_table", help="CSV/TSV with trajectory_id, frame, and CV columns.")
    windows.add_argument("--cv1", required=True)
    windows.add_argument("--cv2", required=True)
    windows.add_argument("--cv1-range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"))
    windows.add_argument("--cv2-range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"))
    windows.add_argument("--spacing", type=float, required=True, help="Grid spacing in the CV units (typically nm).")
    windows.add_argument("--empty-policy", choices=("skip", "nearest", "error"), default="skip")
    windows.add_argument("--max-seed-distance", type=float, help="Maximum nearest-seed offset in grid-spacing units.")
    windows.add_argument("--trajectory-column", default="trajectory_id")
    windows.add_argument("--frame-column", default="frame")
    windows.add_argument("--output", default="US_windows_summary.tsv")

    seeds = sub.add_parser("export-seeds", help="Export assigned trajectory frames as seed PDBs.")
    seeds.add_argument("windows_table")
    seeds.add_argument("trajectory_config")
    seeds.add_argument("--outdir", default="US_windows")
    seeds.add_argument("--overwrite", action="store_true")

    compare = sub.add_parser("compare-fes", help="Compare two same-shaped FES arrays.")
    compare.add_argument("fes_a")
    compare.add_argument("fes_b")
    compare.add_argument("--minima-count", type=int, default=3)
    compare.add_argument("--output", help="Optional JSON destination.")

    plot = sub.add_parser("plot-fes", help="Plot a 1D or 2D FES bundle.")
    plot.add_argument("fes", help="Path to .npy FES (matching .json metadata must exist unless --metadata is given).")
    plot.add_argument("--metadata")
    plot.add_argument("--output", default="fes.png")
    plot.add_argument("--title")

    slices = sub.add_parser("plot-fes-slices", help="Plot 2D slices from a 3D FES bundle.")
    slices.add_argument("fes")
    slices.add_argument("--metadata")
    slices.add_argument("--output-dir", default="fes_slices")
    slices.add_argument("--slice-axis", type=int, default=2)
    slices.add_argument("--number-of-slices", type=int, default=8)

    basins = sub.add_parser("extract-basin-frames", help="Detect separated FES minima and select nearby sampled frames")
    basins.add_argument("fes", help="FES .npy file with matching JSON metadata")
    basins.add_argument("cv_table", help="CV table with one row per sampled frame")
    basins.add_argument("--metadata", help="Optional explicit FES metadata JSON path")
    basins.add_argument("--cv-columns", nargs="+", required=True, help="CV table columns in FES axis order")
    basins.add_argument("--trajectory-column", default="trajectory_id")
    basins.add_argument("--segment-column", default="segment_id")
    basins.add_argument("--frame-column", default="frame")
    basins.add_argument("--min-separation-grid", type=float, default=2.0)
    basins.add_argument("--max-minima", type=int, default=20)
    basins.add_argument("--basin-radius-grid", type=float, default=1.5)
    basins.add_argument("--max-sample-distance-grid", type=float, default=2.5)
    basins.add_argument("--outdir", default="fes_basin_representatives")
    basins.add_argument("--export-pdb", action="store_true", help="Also export whole-system PDBs for matched frames")
    basins.add_argument("--trajectory-config", help="Trajectory configuration used for PDB export")
    basins.add_argument("--overwrite", action="store_true")
    basins.set_defaults(func=command_extract_basin_frames)

    overlay = sub.add_parser("plot-fes-overlay", help="Overlay CV samples colored by a mechanistic observable on a 2D FES")
    overlay.add_argument("fes", help="2D FES .npy file with matching JSON metadata")
    overlay.add_argument("cv_table", help="CV table containing sampled CVs and the observable")
    overlay.add_argument("--metadata", help="Optional explicit FES metadata JSON path")
    overlay.add_argument("--cv-columns", nargs=2, required=True, metavar=("CV_X", "CV_Y"))
    overlay.add_argument("--color-by", required=True, help="Numeric column, e.g. CuS_distance_A or salt_bridge_count")
    overlay.add_argument("--max-points", type=int, default=12000)
    overlay.add_argument("--output", default="fes_mechanism_overlay.png")
    overlay.add_argument("--title")
    overlay.set_defaults(func=command_plot_fes_overlay)

    meta = sub.add_parser("run-metadynamics", help="Run an OpenMM well-tempered metadynamics configuration.")
    meta.add_argument("config")
    meta.add_argument("--pdb", help="Optional override for system.pdb.")

    umbrella = sub.add_parser("run-umbrella", help="Run one OpenMM harmonic umbrella window for distance CVs.")
    umbrella.add_argument("config")
    umbrella.add_argument("--centers", help="Comma-separated CV targets, e.g. d_A1X1=2.2,d_A2X2=0.6 (nm).")
    umbrella.add_argument("--label", default="window")
    umbrella.add_argument("--pdb", help="Optional override for system.pdb (e.g. the chosen window seed PDB).")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compute-cvs":
            config = load_json(args.config)
            table = compute_cv_table(config)
            output = _write_table(table, args.output)
            print(f"Computed {len(table)} frames; wrote {output}")
        elif args.command == "assign-windows":
            table = _read_table(args.cv_table)
            assigned = assign_grid_windows(
                table, cv1=args.cv1, cv2=args.cv2,
                cv1_range=tuple(args.cv1_range), cv2_range=tuple(args.cv2_range),
                spacing=args.spacing, empty_policy=args.empty_policy,
                max_seed_distance=args.max_seed_distance,
                trajectory_column=args.trajectory_column, frame_column=args.frame_column,
            )
            output = _write_table(assigned, args.output)
            counts = assigned["status"].value_counts().to_dict()
            print(f"Wrote {len(assigned)} window records to {output}; statuses={counts}")
            dropped = assigned.attrs.get("dropped_nonfinite_rows", 0)
            if dropped:
                print(f"Warning: ignored {dropped} rows with non-finite CV values.", file=sys.stderr)
            out_of_range = assigned.attrs.get("out_of_range_rows", 0)
            if out_of_range:
                print(f"Warning: {out_of_range} finite rows fall outside the requested CV grid.", file=sys.stderr)
        elif args.command == "export-seeds":
            _export_seeds(args.windows_table, args.trajectory_config, args.outdir, args.overwrite)
        elif args.command == "compare-fes":
            comparison = compare_fes(np.load(args.fes_a, allow_pickle=False),
                                     np.load(args.fes_b, allow_pickle=False),
                                     minima_count=args.minima_count)
            output = json.dumps(comparison.to_dict(), indent=2, allow_nan=False)
            print(output)
            if args.output:
                target = Path(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(output + "\n", encoding="utf-8")
        elif args.command == "plot-fes":
            values, metadata = load_fes_bundle(args.fes, args.metadata)
            output = plot_fes(values, metadata, args.output, title=args.title)
            print(f"Wrote {output}")
        elif args.command == "plot-fes-slices":
            values, metadata = load_fes_bundle(args.fes, args.metadata)
            outputs = plot_fes_slices(values, metadata, args.output_dir,
                                      slice_axis=args.slice_axis,
                                      number_of_slices=args.number_of_slices)
            print(f"Wrote {len(outputs)} slices to {Path(args.output_dir).resolve()}")
        elif args.command == "extract-basin-frames":
            command_extract_basin_frames(args)
        elif args.command == "plot-fes-overlay":
            command_plot_fes_overlay(args)
        elif args.command == "run-metadynamics":
            from .openmm_runner import run_metadynamics
            output = run_metadynamics(args.config, pdb_override=args.pdb)
            print(f"Metadynamics output: {output}")
        elif args.command == "run-umbrella":
            from .openmm_runner import run_umbrella_window
            centers = _parse_centers(args.centers)
            output = run_umbrella_window(args.config, centers, label=args.label, pdb_override=args.pdb)
            print(f"Umbrella output: {output}")
        else:  # pragma: no cover
            parser.error(f"Unsupported command {args.command!r}")
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, argparse.ArgumentTypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0
