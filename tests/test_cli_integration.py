import json

import numpy as np
import pandas as pd

from enhanced_sampling_workflows.cli import main


def test_assign_windows_cli_writes_summary(tmp_path, capsys):
    source = tmp_path / "cvs.csv"
    target = tmp_path / "windows.tsv"
    pd.DataFrame({
        "trajectory_id": ["T4", "T7"], "frame": [10, 20],
        "x": [0.2, 0.8], "y": [0.2, 0.8],
    }).to_csv(source, index=False)
    code = main([
        "assign-windows", str(source), "--cv1", "x", "--cv2", "y",
        "--cv1-range", "0", "1", "--cv2-range", "0", "1",
        "--spacing", "0.5", "--empty-policy", "nearest", "--max-seed-distance", "2",
        "--output", str(target),
    ])
    assert code == 0
    assert target.exists()
    result = pd.read_csv(target, sep="\t")
    assert len(result) == 4
    assert "seed_distance_grid_units" in result
    assert "statuses=" in capsys.readouterr().out


def test_compare_fes_cli_outputs_json(tmp_path, capsys):
    a = np.array([[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]])
    pa, pb = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(pa, a)
    np.save(pb, a + 4.0)
    output = tmp_path / "comparison.json"
    code = main(["compare-fes", str(pa), str(pb), "--minima-count", "1", "--output", str(output)])
    assert code == 0
    payload = json.loads(output.read_text())
    assert payload["max_abs_delta_kj_mol"] == 0.0
    assert json.loads(capsys.readouterr().out)["matched_minima"] == 1
