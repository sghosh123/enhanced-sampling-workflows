import warnings

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis")

from enhanced_sampling_workflows.trajectory import compute_cv_table


def make_tiny_system(tmp_path):
    # Four residues in each of two segments make a tiny PDB/XTC pair suitable for
    # unit tests; coordinates are in Å on input and include a valid periodic box.
    atom_resindex = np.arange(8)
    residue_segindex = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    universe = mda.Universe.empty(
        8, n_residues=8, atom_resindex=atom_resindex, n_segments=2,
        residue_segindex=residue_segindex, trajectory=True,
    )
    universe.add_TopologyAttr("name", ["CA"] * 8)
    universe.add_TopologyAttr("type", ["C"] * 8)
    universe.add_TopologyAttr("elements", ["C"] * 8)
    universe.add_TopologyAttr("resname", ["ALA"] * 8)
    universe.add_TopologyAttr("resid", [56, 63, 26, 18, 55, 62, 25, 18])
    universe.add_TopologyAttr("segid", ["A", "B"])
    universe.add_TopologyAttr("chainID", ["A", "A", "A", "A", "B", "B", "B", "B"])
    universe.atoms.positions = np.array([
        [0, 0, 0], [0, 10, 0], [1, 0, 0], [0, 0, 0],
        [10, 0, 0], [0, 20, 0], [0, 1, 0], [0, 0, 0],
    ], dtype=np.float32)
    universe.dimensions = np.array([100, 100, 100, 90, 90, 90], dtype=np.float32)
    topology = tmp_path / "toy.pdb"
    trajectory = tmp_path / "toy.xtc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe.atoms.write(str(topology))
    with mda.Writer(str(trajectory), n_atoms=8) as writer:
        writer.write(universe.atoms)
    return topology, trajectory


def test_compute_distance_and_vector_angle_units_and_frame_identity(tmp_path):
    topology, trajectory = make_tiny_system(tmp_path)
    config = {
        "_config_dir": str(tmp_path),
        "trajectories": [{"id": "T7", "topology": str(topology), "trajectory": str(trajectory)}],
        "collective_variables": [
            {
                "name": "d_A1X1", "kind": "distance",
                "atoms": [{"segid": "A", "resid": 56, "name": "CA"},
                          {"segid": "B", "resid": 55, "name": "CA"}],
                "min_value": 0.5, "max_value": 2.0, "bias_width": 0.05, "unit": "nm", "pbc": True,
            },
            {
                "name": "angle_BY", "kind": "vector_angle",
                "atoms": [{"segid": "A", "resid": 26, "name": "CA"},
                          {"segid": "A", "resid": 18, "name": "CA"},
                          {"segid": "B", "resid": 25, "name": "CA"},
                          {"segid": "B", "resid": 18, "name": "CA"}],
                "min_value": 0.1, "max_value": 3.0, "bias_width": 0.05, "unit": "rad", "pbc": True,
            },
        ],
    }
    table = compute_cv_table(config)
    assert len(table) == 1
    assert table.loc[0, "trajectory_id"] == "T7"
    assert table.loc[0, "frame"] == 0
    assert table.loc[0, "d_A1X1"] == pytest.approx(1.0, abs=1e-4)
    assert table.loc[0, "angle_BY"] == pytest.approx(np.pi / 2, abs=1e-4)
