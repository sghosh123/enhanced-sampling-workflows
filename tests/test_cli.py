from enhanced_sampling_workflows.cli import build_parser, _parse_centers
import argparse
import pytest


def test_cli_registers_core_commands():
    parser = build_parser()
    for command in ("compute-cvs", "assign-windows", "export-seeds", "compare-fes",
                    "plot-fes", "plot-fes-slices", "run-metadynamics", "run-umbrella"):
        # Parsing --help exits normally; the parser's subcommand action exposes choices.
        choices = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction)).choices
        assert command in choices


def test_parse_centers():
    assert _parse_centers("d1=2.2,d2=0.6") == {"d1": 2.2, "d2": 0.6}
    assert _parse_centers(None) == {}


def test_parse_centers_rejects_duplicate():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_centers("d1=2.2,d1=2.3")
