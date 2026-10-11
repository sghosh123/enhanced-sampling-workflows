import numpy as np
import pandas as pd
import pytest

from enhanced_sampling_workflows.basins import detect_fes_minima, select_basin_representatives, plot_fes_overlay
from enhanced_sampling_workflows.fes import build_fes_metadata, save_fes_bundle


def _surface_and_metadata():
    fes = np.full((7, 7), 10.0)
    fes[1, 1] = 0.0
    fes[5, 5] = 1.0
    cvs = [
        {"name": "x", "kind": "distance", "unit": "nm", "min_value": 0.0, "max_value": 7.0},
        {"name": "y", "kind": "distance", "unit": "nm", "min_value": 0.0, "max_value": 7.0},
    ]
    return fes, build_fes_metadata(fes, cvs)


def test_detect_fes_minima_collapses_local_minima_and_sorts_by_energy():
    fes, meta = _surface_and_metadata()
    minima = detect_fes_minima(fes, meta, min_separation_grid=2.0)
    assert len(minima) == 2
    assert minima.iloc[0]["free_energy_kJ_mol"] == 0.0
    assert minima.iloc[1]["free_energy_kJ_mol"] == 1.0
    assert minima.iloc[0]["x_index"] == 1
    assert minima.iloc[1]["y_index"] == 5


def test_representatives_are_near_a_minimum_and_never_globally_guessed():
    fes, meta = _surface_and_metadata()
    table = pd.DataFrame({
        "trajectory_id": ["T1", "T2"], "segment_id": ["0", "0"], "frame": [10, 20],
        "x": [1.1, 5.1], "y": [0.9, 5.0], "CuS_distance_A": [2.9, 3.6],
    })
    reps = select_basin_representatives(table, fes, meta, cv_columns=["x", "y"], basin_radius_grid=1.5,
                                        max_sample_distance_grid=2.0)
    assert list(reps["representative_status"]) == ["matched", "matched"]
    assert list(reps["representative_frame"]) == [10, 20]


def test_unrepresented_minimum_is_not_assigned_a_distant_frame():
    fes, meta = _surface_and_metadata()
    table = pd.DataFrame({"trajectory_id": ["T1"], "segment_id": ["0"], "frame": [7], "x": [2.0], "y": [2.0]})
    reps = select_basin_representatives(table, fes, meta, cv_columns=["x", "y"], basin_radius_grid=0.5,
                                        max_sample_distance_grid=0.5)
    assert (reps["representative_status"] == "unrepresented").all()
    assert reps["representative_frame"].isna().all()


def test_plot_fes_mechanistic_overlay(tmp_path):
    pytest.importorskip("matplotlib")
    fes, meta = _surface_and_metadata()
    table = pd.DataFrame({"x": [1, 2, 4, 5], "y": [1, 2, 4, 5], "CuS_distance_A": [2.5, 3.0, 4.5, 3.4]})
    path = plot_fes_overlay(fes, meta, table, cv_columns=["x", "y"], color_column="CuS_distance_A",
                            output_path=tmp_path / "overlay.png")
    assert path.exists() and path.stat().st_size > 1000


def test_extract_basin_frames_cli_writes_representative_table(tmp_path):
    fes, meta = _surface_and_metadata()
    fes_path, _ = save_fes_bundle(fes, tmp_path / "fes", meta)
    table_path = tmp_path / "cvs.csv"
    pd.DataFrame({"trajectory_id": ["T1", "T2"], "segment_id": ["0", "0"], "frame": [0, 1],
                  "x": [1.0, 5.0], "y": [1.0, 5.0]}).to_csv(table_path, index=False)
    from enhanced_sampling_workflows.cli import main
    outdir = tmp_path / "basins"
    assert main(["extract-basin-frames", str(fes_path), str(table_path), "--cv-columns", "x", "y", "--outdir", str(outdir)]) == 0
    table = pd.read_csv(outdir / "basin_representatives.tsv", sep="\t")
    assert len(table) == 2
    assert (table["representative_status"] == "matched").all()


def test_plot_fes_overlay_cli(tmp_path):
    pytest.importorskip("matplotlib")
    fes, meta = _surface_and_metadata()
    fes_path, _ = save_fes_bundle(fes, tmp_path / "fes", meta)
    table_path = tmp_path / "samples.csv"
    pd.DataFrame({"x": [1, 2, 5], "y": [1, 2, 5], "salt_bridge_count": [2, 0, 1]}).to_csv(table_path, index=False)
    from enhanced_sampling_workflows.cli import main
    output = tmp_path / "overlay.png"
    assert main(["plot-fes-overlay", str(fes_path), str(table_path), "--cv-columns", "x", "y",
                 "--color-by", "salt_bridge_count", "--output", str(output)]) == 0
    assert output.exists() and output.stat().st_size > 1000


def test_extract_basin_frames_exports_pdb_for_matched_frames(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    universe = mda.Universe.empty(2, n_residues=2, n_segments=2,
                                  atom_resindex=np.array([0, 1]), residue_segindex=np.array([0, 1]), trajectory=True)
    universe.add_TopologyAttr("names", ["CA", "CA"])
    universe.add_TopologyAttr("types", ["C", "C"])
    universe.add_TopologyAttr("elements", ["C", "C"])
    universe.add_TopologyAttr("resids", [1, 2])
    universe.add_TopologyAttr("resnames", ["ALA", "GLY"])
    universe.add_TopologyAttr("segids", ["A", "B"])
    universe.add_TopologyAttr("chainIDs", ["A", "B"])
    universe.atoms.positions = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    universe.dimensions = np.array([10, 10, 10, 90, 90, 90], dtype=np.float32)
    topology = tmp_path / "toy.pdb"
    trajectory = tmp_path / "toy.xtc"
    universe.atoms.write(str(topology))
    with mda.Writer(str(trajectory), n_atoms=2) as writer:
        writer.write(universe.atoms)
        universe.atoms.positions = np.array([[2, 1, 1], [3, 2, 2]], dtype=np.float32)
        writer.write(universe.atoms)

    fes, meta = _surface_and_metadata()
    # One minimum and its sampled representative; only request a single minimum to keep
    # the source structure mapping and expected output unambiguous.
    fes[5, 5] = 11.0
    fes_path, _ = save_fes_bundle(fes, tmp_path / "fes_export", meta)
    cv_table = tmp_path / "cv_export.csv"
    pd.DataFrame({"trajectory_id": ["T1"], "segment_id": ["0"], "frame": [0], "x": [1.0], "y": [1.0]}).to_csv(cv_table, index=False)
    config = tmp_path / "trajectory_sources.json"
    config.write_text(__import__("json").dumps({
        "trajectories": [{"id": "T1", "topology": str(topology), "trajectory": str(trajectory)}],
        "collective_variables": [{"name": "x", "kind": "distance", "atoms": [
            {"segid": "A", "resid": 1, "name": "CA"}, {"segid": "B", "resid": 2, "name": "CA"}],
            "min_value": 0.0, "max_value": 7.0, "bias_width": 0.1, "unit": "nm"}],
    }))
    from enhanced_sampling_workflows.cli import main
    outdir = tmp_path / "exported"
    assert main(["extract-basin-frames", str(fes_path), str(cv_table), "--cv-columns", "x", "y",
                 "--max-minima", "1", "--outdir", str(outdir), "--export-pdb", "--trajectory-config", str(config)]) == 0
    assert (outdir / "structures" / "basin_001" / "representative.pdb").exists()
