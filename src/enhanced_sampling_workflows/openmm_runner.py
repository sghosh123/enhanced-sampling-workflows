"""Optional OpenMM runners for distance/angle metadynamics and distance umbrellas.

OpenMM is imported lazily, so trajectory analysis and FES diagnostics work without
an OpenMM installation. The runners intentionally do not add project-specific copper
bond restraints or positional restraints: those must be specified and verified for
an individual molecular system before production use.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConfigurationError, load_json, resolve_config_path, validate_collective_variables
from .convergence import ConvergenceTracker
from .fes import build_fes_metadata, save_fes_bundle


# Use pair-distance identities rather than raw coordinate differences so OpenMM's
# built-in distance(p_i,p_j) can apply minimum-image displacements when PBC is on.
# For v1=p1-p2 and v2=p3-p4, v1·v2 = 1/2*(d14^2+d23^2-d13^2-d24^2).
VECTOR_ANGLE_EXPRESSION = (
    "acos(min(1,max(-1,(distance(p1,p4)^2+distance(p2,p3)^2-"
    "distance(p1,p3)^2-distance(p2,p4)^2)/"
    "(2*distance(p1,p2)*distance(p3,p4)+1e-12))))"
)


def _openmm_imports():
    try:
        import openmm as mm
        import openmm.app as app
        from openmm import unit
        from openmm.app import BiasVariable, Metadynamics
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OpenMM runners require OpenMM >= 8.0. Install it in the simulation environment with "
            "'python -m pip install openmm' or the matching Conda package."
        ) from exc
    return mm, app, unit, BiasVariable, Metadynamics


def _find_openmm_atom_index(topology, atom_ref: dict[str, Any]) -> int:
    if "index" in atom_ref:
        idx = int(atom_ref["index"])
        atoms = list(topology.atoms())
        if idx < 0 or idx >= len(atoms):
            raise ConfigurationError(f"Atom index {idx} is outside the topology (0..{len(atoms)-1}).")
        return idx
    if "selection" in atom_ref:
        raise ConfigurationError(
            "OpenMM atom references use chain/resid/name (or explicit zero-based index); "
            "MDAnalysis selection strings are not portable to OpenMM."
        )
    chain_id = atom_ref.get("chain", atom_ref.get("chain_id", atom_ref.get("segid")))
    if chain_id is None or "resid" not in atom_ref or "name" not in atom_ref:
        raise ConfigurationError("OpenMM atom reference requires chain, resid, and name (or index).")
    matches = [
        atom.index for atom in topology.atoms()
        if str(atom.residue.chain.id) == str(chain_id)
        and str(atom.residue.id) == str(atom_ref["resid"])
        and str(atom.name) == str(atom_ref["name"])
    ]
    if len(matches) != 1:
        raise ConfigurationError(
            f"OpenMM atom lookup chain={chain_id!r}, resid={atom_ref['resid']!r}, "
            f"name={atom_ref['name']!r} matched {len(matches)} atoms; expected exactly one."
        )
    return int(matches[0])


def _build_cv_force(mm, cv: dict[str, Any], atom_indices: list[int]):
    if cv["kind"] == "distance":
        force = mm.CustomBondForce("r")
        force.addBond(atom_indices[0], atom_indices[1])
        force.setUsesPeriodicBoundaryConditions(bool(cv.get("pbc", True)))
    elif cv["kind"] == "vector_angle":
        force = mm.CustomCompoundBondForce(4, VECTOR_ANGLE_EXPRESSION)
        force.addBond(atom_indices)
        force.setUsesPeriodicBoundaryConditions(bool(cv.get("pbc", True)))
    else:  # validated earlier, retained defensively
        raise ConfigurationError(f"Unsupported OpenMM CV kind: {cv['kind']!r}")
    return force


def _resolve_forcefield_entry(entry: str, config_dir: str) -> str:
    """Resolve custom XML paths while allowing a small set of OpenMM built-ins."""
    candidate = resolve_config_path(entry, config_dir)
    if candidate.exists():
        return str(candidate)
    builtins = {
        "charmm36.xml", "charmm36/water.xml", "amber14-all.xml", "amber14/tip3p.xml",
        "amber19-all.xml", "amber19/tip3p.xml", "amber99sb.xml", "amber99sbildn.xml",
        "amber03.xml", "amber96.xml", "amoeba2018.xml", "amoeba2009.xml",
    }
    if entry in builtins:
        return entry
    raise FileNotFoundError(
        f"Force-field XML {entry!r} was not found relative to the configuration directory "
        f"({candidate}). Use a valid relative/absolute path or a supported OpenMM built-in filename."
    )


def _load_openmm_base(config: dict[str, Any], config_path: str | Path, *, pdb_override: str | None = None):
    mm, app, unit, BiasVariable, Metadynamics = _openmm_imports()
    config_dir = str(Path(config_path).expanduser().resolve().parent)
    system_cfg = config.get("system", {})
    pdb_value = pdb_override or system_cfg.get("pdb") or config.get("pdb")
    if not pdb_value:
        raise ConfigurationError("OpenMM configuration requires system.pdb.")
    pdb_path = Path(pdb_override).expanduser().resolve() if pdb_override else resolve_config_path(pdb_value, config_dir)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {pdb_path}")
    pdb = app.PDBFile(str(pdb_path))
    if system_cfg.get("box_vectors_nm"):
        box_lengths = [float(value) for value in system_cfg["box_vectors_nm"]]
        if len(box_lengths) != 3 or any(value <= 0 or not math.isfinite(value) for value in box_lengths):
            raise ConfigurationError("system.box_vectors_nm must contain three positive finite lengths.")
        pdb.topology.setUnitCellDimensions(mm.Vec3(*box_lengths) * unit.nanometer)

    forcefield_files = system_cfg.get("forcefield_xml", config.get("forcefield_xml"))
    if not isinstance(forcefield_files, list) or not forcefield_files:
        raise ConfigurationError("system.forcefield_xml must be a non-empty list of OpenMM XML files.")
    forcefield_files = [_resolve_forcefield_entry(str(item), config_dir) for item in forcefield_files]
    forcefield = app.ForceField(*forcefield_files)
    method_name = str(system_cfg.get("nonbonded_method", "PME"))
    method_map = {
        "PME": app.PME,
        "NoCutoff": app.NoCutoff,
        "CutoffPeriodic": app.CutoffPeriodic,
        "CutoffNonPeriodic": app.CutoffNonPeriodic,
    }
    if method_name not in method_map:
        raise ConfigurationError(f"Unsupported nonbonded_method {method_name!r}.")
    if method_name in {"PME", "CutoffPeriodic"}:
        dimensions = pdb.topology.getUnitCellDimensions()
        if dimensions is None or any(float(value.value_in_unit(unit.nanometer) if hasattr(value, "value_in_unit") else value) <= 0 for value in dimensions):
            raise ConfigurationError(
                f"nonbonded_method={method_name} requires periodic box dimensions in the PDB or system.box_vectors_nm."
            )
    cutoff_value = float(system_cfg.get("nonbonded_cutoff_angstrom", 12.0))
    if not math.isfinite(cutoff_value) or cutoff_value <= 0:
        raise ConfigurationError("system.nonbonded_cutoff_angstrom must be a positive finite value.")
    cutoff = cutoff_value * unit.angstrom
    constraints_name = system_cfg.get("constraints", "HBonds")
    constraint_map = {"None": None, "HBonds": app.HBonds, "AllBonds": app.AllBonds}
    if constraints_name not in constraint_map:
        raise ConfigurationError("system.constraints must be 'None', 'HBonds', or 'AllBonds'.")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=method_map[method_name],
        nonbondedCutoff=cutoff,
        constraints=constraint_map[constraints_name],
        rigidWater=bool(system_cfg.get("rigid_water", True)),
        removeCMMotion=bool(system_cfg.get("remove_cm_motion", True)),
        ewaldErrorTolerance=float(system_cfg.get("ewald_error_tolerance", 5e-4)),
    )
    cvs = validate_collective_variables(config.get("collective_variables"))
    if any(cv.get("pbc", True) for cv in cvs):
        dimensions = pdb.topology.getUnitCellDimensions()
        if dimensions is None or any(
            float(value.value_in_unit(unit.nanometer) if hasattr(value, "value_in_unit") else value) <= 0
            for value in dimensions
        ):
            raise ConfigurationError(
                "At least one CV requests PBC corrections, but the PDB has no valid periodic box. "
                "Provide system.box_vectors_nm or set pbc=false only when appropriate."
            )
    cv_forces: dict[str, Any] = {}
    for cv in cvs:
        indices = [_find_openmm_atom_index(pdb.topology, atom) for atom in cv["atoms"]]
        cv_forces[cv["name"]] = _build_cv_force(mm, cv, indices)
    return mm, app, unit, BiasVariable, Metadynamics, pdb, system, cvs, cv_forces, pdb_path


def _make_integrator_and_simulation(mm, app, unit, pdb, system, config, output_dir: Path):
    integrator_cfg = config.get("integrator", {})
    sim_cfg = config.get("simulation", {})
    temperature = float(integrator_cfg.get("temperature_k", 310.0))
    friction = float(integrator_cfg.get("friction_per_ps", 1.0)) / unit.picosecond
    timestep_fs = float(integrator_cfg.get("timestep_fs", 2.0))
    if temperature <= 0 or friction <= 0 or timestep_fs <= 0:
        raise ConfigurationError("Temperature, friction, and timestep must be positive.")
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, friction, timestep_fs * unit.femtoseconds
    )
    platform_name = str(sim_cfg.get("platform", "CPU"))
    properties = {str(k): str(v) for k, v in sim_cfg.get("platform_properties", {}).items()}
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
    except Exception as exc:
        raise RuntimeError(
            f"OpenMM platform {platform_name!r} is unavailable. Try 'CPU', or verify the GPU/OpenMM installation."
        ) from exc
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    if bool(sim_cfg.get("minimize", True)):
        simulation.minimizeEnergy(maxIterations=int(sim_cfg.get("minimize_max_iterations", 0)))
    simulation.context.setVelocitiesToTemperature(float(sim_cfg.get("initial_velocity_temperature_k", temperature)) * unit.kelvin)
    equilibration_steps = int(sim_cfg.get("equilibration_steps", 0))
    if equilibration_steps < 0:
        raise ConfigurationError("simulation.equilibration_steps must be non-negative.")
    if equilibration_steps:
        simulation.step(equilibration_steps)

    # Add production reporters only after the configured equilibration is complete.
    dcd_interval = int(sim_cfg.get("dcd_interval_steps", 0))
    if dcd_interval > 0:
        simulation.reporters.append(app.DCDReporter(str(output_dir / "production.dcd"), dcd_interval))
    state_interval = int(sim_cfg.get("state_interval_steps", 0))
    if state_interval > 0:
        simulation.reporters.append(app.StateDataReporter(
            str(output_dir / "state.tsv"), state_interval,
            step=True, time=True, temperature=True, potentialEnergy=True, speed=True,
            separator="\t",
        ))
    return simulation, temperature


def _validate_metadynamics_settings(settings: dict[str, Any]) -> dict[str, Any]:
    required_defaults = {
        "bias_factor": 15.0,
        "hill_height_kj_mol": 1.5,
        "hill_frequency_steps": 100,
        "save_frequency_steps": 1000,
        "steps_per_block": 25000,
        "total_steps": 1000000,
        "bias_directory": "meta_bias",
        "output_directory": "metadynamics_output",
    }
    result = {**required_defaults, **settings}
    for key in ("hill_frequency_steps", "save_frequency_steps", "steps_per_block", "total_steps"):
        result[key] = int(result[key])
        if result[key] <= 0:
            raise ConfigurationError(f"metadynamics.{key} must be a positive integer.")
    if result["save_frequency_steps"] % result["hill_frequency_steps"] != 0:
        raise ConfigurationError("save_frequency_steps must be a multiple of hill_frequency_steps (OpenMM requirement).")
    if result["steps_per_block"] % result["hill_frequency_steps"] != 0:
        raise ConfigurationError("steps_per_block must be a multiple of hill_frequency_steps.")
    if result["total_steps"] % result["hill_frequency_steps"] != 0:
        raise ConfigurationError("total_steps must be divisible by hill_frequency_steps.")
    if result["bias_factor"] <= 1 or result["hill_height_kj_mol"] <= 0:
        raise ConfigurationError("bias_factor must be > 1 and hill_height_kj_mol must be positive.")
    if "temperature_k" in result and (not math.isfinite(float(result["temperature_k"])) or float(result["temperature_k"]) <= 0):
        raise ConfigurationError("metadynamics.temperature_k must be a positive finite value.")
    convergence = result.get("convergence")
    if convergence is not None:
        if not isinstance(convergence, dict):
            raise ConfigurationError("metadynamics.convergence must be an object or null.")
        for key in ("max_delta_f_kj_mol", "max_rmse_kj_mol", "max_minima_shift_grid"):
            if float(convergence.get(key, 0)) < 0:
                raise ConfigurationError(f"convergence.{key} must be non-negative.")
    return result


def _create_fresh_output_dir(path: Path) -> None:
    """Create an output directory, refusing to mix results from separate runs."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new output_directory/label or archive the old run first. "
            "This runner does not implicitly restart/reuse bias files."
        )
    path.mkdir(parents=True, exist_ok=True)


def run_metadynamics(config_path: str | Path, *, pdb_override: str | None = None) -> Path:
    """Run OpenMM well-tempered metadynamics as configured and save FES snapshots."""
    config = load_json(config_path)
    settings = _validate_metadynamics_settings(config.get("metadynamics", {}))
    mm, app, unit, BiasVariable, Metadynamics, pdb, system, cvs, cv_forces, pdb_path = _load_openmm_base(
        config, config_path, pdb_override=pdb_override
    )
    physical_temperature = float(config.get("integrator", {}).get("temperature_k", 310.0))
    meta_temperature = float(settings.get("temperature_k", physical_temperature))
    if not math.isclose(meta_temperature, physical_temperature, rel_tol=0.0, abs_tol=1e-8):
        raise ConfigurationError(
            "metadynamics.temperature_k should match integrator.temperature_k for this workflow."
        )
    config_dir = Path(config_path).expanduser().resolve().parent
    output_dir = resolve_config_path(settings["output_directory"], config_dir)
    _create_fresh_output_dir(output_dir)
    bias_dir = resolve_config_path(settings["bias_directory"], output_dir)
    _create_fresh_output_dir(bias_dir)

    variables = [BiasVariable(
        force=cv_forces[cv["name"]],
        minValue=cv["min_value"], maxValue=cv["max_value"],
        biasWidth=cv["bias_width"], periodic=cv["periodic"], gridWidth=cv.get("grid_width"),
    ) for cv in cvs]
    meta = Metadynamics(
        system=system, variables=variables,
        temperature=meta_temperature * unit.kelvin,
        biasFactor=float(settings["bias_factor"]),
        height=float(settings["hill_height_kj_mol"]) * unit.kilojoules_per_mole,
        frequency=int(settings["hill_frequency_steps"]),
        saveFrequency=int(settings["save_frequency_steps"]),
        biasDir=str(bias_dir),
    )
    simulation, _temperature = _make_integrator_and_simulation(mm, app, unit, pdb, system, config, output_dir)
    convergence_cfg = settings.get("convergence")
    tracker = None
    if convergence_cfg is not None:
        tracker = ConvergenceTracker(
            max_delta_f=float(convergence_cfg.get("max_delta_f_kj_mol", 5.0)),
            max_rmse=float(convergence_cfg.get("max_rmse_kj_mol", 2.5)),
            max_minima_shift_grid=float(convergence_cfg.get("max_minima_shift_grid", 1.0)),
            minima_count=int(convergence_cfg.get("minima_count", 3)),
            consecutive_needed=int(convergence_cfg.get("consecutive_blocks", 4)),
        )

    history_path = output_dir / "cv_history.csv"
    history_fields = ["step", "time_ps", *[cv["name"] for cv in cvs], "fes_file", "converged", "stable_count"]
    steps_done = 0
    block_number = 0
    convergence_record: dict[str, Any] | None = None
    final_fes = None
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        while steps_done < settings["total_steps"]:
            block_steps = min(settings["steps_per_block"], settings["total_steps"] - steps_done)
            if block_steps % settings["hill_frequency_steps"] != 0:
                raise ConfigurationError(
                    "total_steps must be divisible by hill_frequency_steps so the final block ends on a hill step."
                )
            meta.step(simulation, block_steps)
            steps_done += block_steps
            block_number += 1
            cv_values = [float(v) for v in meta.getCollectiveVariables(simulation)]
            time_ps = float(simulation.context.getState().getTime().value_in_unit(unit.picosecond))
            fes = np.asarray(meta.getFreeEnergy(), dtype=float)
            final_fes = fes
            extra = {
                "source_pdb": str(pdb_path), "step": steps_done, "time_ps": time_ps,
                "integrator_timestep_fs": float(config.get("integrator", {}).get("timestep_fs", 2.0)),
                "metadynamics": {k: v for k, v in settings.items() if k != "convergence"},
            }
            metadata = build_fes_metadata(fes, cvs, extra=extra)
            fes_path, _ = save_fes_bundle(fes, output_dir / f"fes_block_{block_number:05d}", metadata)
            result = {
                "step": steps_done,
                "time_ps": time_ps,
                **{cv["name"]: value for cv, value in zip(cvs, cv_values)},
                "fes_file": fes_path.name,
                "converged": False,
                "stable_count": 0,
            }
            if tracker is not None:
                convergence_record = tracker.observe(fes, step=steps_done)
                result["converged"] = bool(convergence_record["converged"])
                result["stable_count"] = int(convergence_record["stable_count"])
                with (output_dir / "convergence_history.jsonl").open("a", encoding="utf-8") as ch:
                    ch.write(json.dumps(convergence_record, allow_nan=False) + "\n")
            writer.writerow(result)
            handle.flush()
            if tracker is not None and convergence_record and convergence_record["converged"]:
                break

    if final_fes is not None:
        final_metadata = build_fes_metadata(final_fes, cvs, extra={
            "source_pdb": str(pdb_path), "steps_completed": steps_done,
            "time_ps": float(simulation.context.getState().getTime().value_in_unit(unit.picosecond)),
            "convergence_status": convergence_record,
        })
        save_fes_bundle(final_fes, output_dir / "fes_final", final_metadata)
    summary = {
        "config_path": str(Path(config_path).resolve()),
        "source_pdb": str(pdb_path),
        "steps_completed": steps_done,
        "blocks_completed": block_number,
        "convergence_enabled": tracker is not None,
        "convergence_reached": bool(convergence_record and convergence_record.get("converged")),
        "convergence_record": convergence_record,
        "warning": "Convergence thresholds are heuristics, not proof of thermodynamic convergence.",
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return output_dir


def run_umbrella_window(
    config_path: str | Path,
    centers_nm: dict[str, float],
    *,
    label: str = "umbrella_window",
    pdb_override: str | None = None,
) -> Path:
    """Run one OpenMM harmonic umbrella window for one or more distance CVs.

    The current generalized umbrella runner supports distance CVs only, with target
    distances in nm and a shared spring constant in kJ/mol/nm^2.
    """
    config = load_json(config_path)
    mm, app, unit, _BiasVariable, _Metadynamics, pdb, system, cvs, cv_forces, pdb_path = _load_openmm_base(
        config, config_path, pdb_override=pdb_override
    )
    umbrella = config.get("umbrella", {})
    if not label or label in {".", ".."} or Path(label).name != label:
        raise ConfigurationError("label must be a simple directory name without path separators.")
    cv_map = {cv["name"]: cv for cv in cvs}
    if not centers_nm:
        centers_nm = umbrella.get("centers_nm", {})
    if not isinstance(centers_nm, dict) or not centers_nm:
        raise ConfigurationError("Provide centers_nm for at least one CV, either by CLI or in config. ")
    if not set(centers_nm).issubset(cv_map):
        raise ConfigurationError(f"Unknown umbrella CV name(s): {sorted(set(centers_nm) - set(cv_map))}")
    for name, center in centers_nm.items():
        if cv_map[name]["kind"] != "distance":
            raise ConfigurationError("Umbrella runner currently supports distance CVs only; angle restraints need separate units.")
        if not math.isfinite(float(center)):
            raise ConfigurationError(f"Umbrella center for {name!r} must be finite.")
        if float(center) < cv_map[name]["min_value"] or float(center) > cv_map[name]["max_value"]:
            raise ConfigurationError(
                f"Umbrella center for {name!r} lies outside its configured CV range "
                f"[{cv_map[name]['min_value']}, {cv_map[name]['max_value']}] nm."
            )
    k = float(umbrella.get("force_constant_kj_mol_nm2", 2000.0))
    if not math.isfinite(k) or k <= 0:
        raise ConfigurationError("umbrella.force_constant_kj_mol_nm2 must be positive.")
    ordered_names = list(centers_nm.keys())
    terms = [f"(q{i}-{float(centers_nm[name]):.12g})^2" for i, name in enumerate(ordered_names)]
    bias_force = mm.CustomCVForce("0.5*k*(" + "+".join(terms) + ")")
    bias_force.addGlobalParameter("k", k)
    for i, name in enumerate(ordered_names):
        bias_force.addCollectiveVariable(f"q{i}", cv_forces[name])
    system.addForce(bias_force)

    config_dir = Path(config_path).expanduser().resolve().parent
    output_value = umbrella.get("output_directory", "umbrella_output")
    base_out = resolve_config_path(output_value, config_dir)
    output_dir = base_out / label
    _create_fresh_output_dir(output_dir)
    simulation, _temperature = _make_integrator_and_simulation(mm, app, unit, pdb, system, config, output_dir)

    production_steps = int(umbrella.get("production_steps", 1000000))
    block_steps = int(umbrella.get("steps_per_block", 5000))
    if production_steps <= 0 or block_steps <= 0:
        raise ConfigurationError("umbrella production_steps and steps_per_block must be positive.")
    history = output_dir / "cv_history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        fields = ["step", "time_ps", *ordered_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        steps_done = 0
        while steps_done < production_steps:
            count = min(block_steps, production_steps - steps_done)
            simulation.step(count)
            steps_done += count
            values = [float(v) for v in bias_force.getCollectiveVariableValues(simulation.context)]
            time_ps = float(simulation.context.getState().getTime().value_in_unit(unit.picosecond))
            writer.writerow({"step": steps_done, "time_ps": time_ps,
                             **{name: value for name, value in zip(ordered_names, values)}})
            handle.flush()
    summary = {
        "label": label, "source_pdb": str(pdb_path), "centers_nm": {k: float(v) for k, v in centers_nm.items()},
        "force_constant_kj_mol_nm2": k, "production_steps": production_steps,
        "steps_completed": steps_done,
        "note": "Project-specific restraints and equilibration choices must be reviewed for the target system.",
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return output_dir
