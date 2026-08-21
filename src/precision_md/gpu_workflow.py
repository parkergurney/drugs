from __future__ import annotations

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from .model import MaceEvaluator
from .policies import precision_context
from .trajectories import (Checkpoint, basin_boundary_distance, conformer_state,
                           pair_distance_rmsd, restore_checkpoint, serialize_checkpoint,
                           unsafe_label, wrapped_angle_difference)


def _dependencies():
    try:
        from ase import units
        from ase.build import molecule
        from ase.calculators.calculator import Calculator, all_changes
        from ase.md.langevin import Langevin
        from ase.md.verlet import VelocityVerlet
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    except ImportError as e: raise RuntimeError("install precision-md[ml] for trajectory workflows") from e
    return units, molecule, Calculator, all_changes, Langevin, VelocityVerlet, MaxwellBoltzmannDistribution, Stationary


def _policy_calculator(base, policy, device):
    _, _, Calculator, all_changes, *_ = _dependencies()
    class PolicyCalculator(Calculator):
        implemented_properties = ["energy", "forces"]
        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            with precision_context(policy, device.split(":")[0]):
                base.calculate(atoms, properties, system_changes)
            self.results = {k: np.asarray(v).copy() if hasattr(v, "shape") else v for k, v in base.results.items()}
    return PolicyCalculator()


def _butane(angle):
    _, molecule, *_ = _dependencies()
    atoms = molecule("C4H10")
    carbon = [a.index for a in atoms if a.symbol == "C"]
    if len(carbon) != 4: raise RuntimeError("ASE C4H10 template does not contain four carbons")
    mask = np.zeros(len(atoms), bool); mask[carbon[3]] = True
    # Move atoms bonded to the terminal carbon together with it.
    for i in range(len(atoms)):
        if i != carbon[2] and np.linalg.norm(atoms.positions[i] - atoms.positions[carbon[3]]) < 1.3: mask[i] = True
    atoms.set_dihedral(*carbon, angle, mask=mask)
    return atoms, carbon


def prepare_trajectory(config):
    units, _, _, _, Langevin, VelocityVerlet, MaxwellBoltzmannDistribution, Stationary = _dependencies()
    evaluator = MaceEvaluator("small", "cuda"); out = config.output_dir; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed); checkpoint_id = 0; timings = []
    for label, angle in (("trans", 180.), ("gauche", 60.)):
        atoms, carbon = _butane(angle); atoms.calc = _policy_calculator(evaluator.calculator, "fp32", "cuda")
        MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature_k, rng=rng); Stationary(atoms)
        eq = Langevin(atoms, config.timestep_fs * units.fs, temperature_K=config.temperature_k,
                      friction=0.01 / units.fs, rng=rng)
        eq.run(round(config.equilibration_ps * 1000 / config.timestep_fs))
        dyn = VelocityVerlet(atoms, config.timestep_fs * units.fs)
        total = round(config.production_ps * 1000 / config.timestep_fs)
        per_start = config.checkpoints // 2; targets = set(np.linspace(0, total - 1, per_start, dtype=int))
        for step in range(total):
            dyn.run(1)
            if step in targets:
                cp = Checkpoint(checkpoint_id, atoms.positions.copy(), atoms.get_velocities().copy(), atoms.get_masses(),
                                (step + 1) * config.timestep_fs, {"seed": config.seed, "start": label, "carbon": carbon})
                elapsed = serialize_checkpoint(cp, out / f"checkpoint-{checkpoint_id:03d}.npz")
                timings.append({"checkpoint_id": checkpoint_id, "serialize_seconds": elapsed}); checkpoint_id += 1
    pd.DataFrame(timings).to_parquet(out / "checkpoint_timings.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(evaluator.manifest() | {"seed": config.seed}, indent=2) + "\n")


def _run_segment(cp, evaluator, policy, steps, timestep_fs):
    units, _, _, _, _, VelocityVerlet, *_ = _dependencies()
    from ase import Atoms
    atoms = Atoms("C4H10", positions=cp.positions, masses=cp.masses); atoms.set_velocities(cp.velocities)
    atoms.calc = _policy_calculator(evaluator.calculator, policy, "cuda")
    initial_energy = atoms.get_potential_energy(); start = time.perf_counter()
    VelocityVerlet(atoms, timestep_fs * units.fs).run(steps)
    elapsed = time.perf_counter() - start; energy = atoms.get_potential_energy(); forces = atoms.get_forces()
    carbon = cp.metadata["carbon"]; dihedral = atoms.get_dihedral(*carbon)
    return {"positions": atoms.positions.copy(), "velocities": atoms.get_velocities().copy(), "forces": forces,
            "initial_energy_ev": initial_energy, "endpoint_energy_ev": energy, "dihedral_deg": dihedral,
            "conformer": conformer_state(dihedral), "seconds": elapsed,
            "minimum_distance": float(np.min(atoms.get_all_distances()[np.triu_indices(len(atoms), 1)]))}


def fork_segments(config):
    if not config.selected_policy: raise RuntimeError("selected_policy must be set after Gate 1 passes")
    evaluator = MaceEvaluator("small", "cuda"); rows = []
    for path in sorted(config.output_dir.glob("checkpoint-*.npz")):
        cp, restore_seconds = restore_checkpoint(path)
        for length in config.block_lengths:
            ref = _run_segment(cp.clone(), evaluator, "fp32", length, config.timestep_fs)
            fast = _run_segment(cp.clone(), evaluator, config.selected_policy, length, config.timestep_fs)
            audit_start = time.perf_counter()
            force_delta = np.linalg.norm(ref["forces"] - fast["forces"], axis=1)
            audit_seconds = time.perf_counter() - audit_start
            unsafe, reasons = unsafe_label(ref, fast); force_norm = np.linalg.norm(ref["forces"], axis=1)
            rows.append({"checkpoint_id": cp.checkpoint_id, "block_length": length, "unsafe": unsafe,
                         "unsafe_reasons": reasons, "fp32_seconds": ref["seconds"], "fast_seconds": fast["seconds"],
                         "audit_seconds": audit_seconds, "restore_seconds": restore_seconds, "checkpoint_seconds": 0.,
                         "max_force_disagreement": force_delta.max(), "mean_force_disagreement": force_delta.mean(),
                         "p95_force_disagreement": np.quantile(force_delta, .95),
                         "relative_force_disagreement": force_delta.max() / max(force_norm.max(), 1e-12),
                         "energy_disagreement_per_atom": abs(ref["endpoint_energy_ev"] - fast["endpoint_energy_ev"]) / len(cp.masses),
                         "maximum_force_norm": force_norm.max(), "minimum_distance": fast["minimum_distance"],
                         "basin_boundary_distance": basin_boundary_distance(fast["dihedral_deg"])})
    pd.DataFrame(rows).to_parquet(config.output_dir / "forks.parquet", index=False)
