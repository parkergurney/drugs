from __future__ import annotations

import hashlib, importlib.metadata, platform, time
import numpy as np
from .policies import Policy, precision_context
from .mace_adapter import install_fp32_result_adapter, result_adapter_diagnostics
from .types import (AtomicBatch, BatchedEvaluationResult, EvaluationResult,
                    PreparedMaceBatch)


class MaceEvaluator:
    def __init__(self, model="small", device="cuda"):
        self.device = device
        try:
            import torch
            from mace.calculators import mace_off
        except ImportError as e:
            raise RuntimeError("install precision-md[ml] to evaluate MACE") from e
        self.torch = torch
        self.calculator = mace_off(model=model, device=device, default_dtype="float32")
        install_fp32_result_adapter(self.calculator)
        self.model_hash = self._hash_model()

    def _hash_model(self):
        h = hashlib.sha256()
        for value in self.calculator.models[0].state_dict().values():
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _attach_fresh_calculator(self, atoms):
        """Attach MACE after clearing ASE results from any previous policy/frame."""
        self.calculator.reset()
        atoms.calc = self.calculator

    def evaluate(self, batch: AtomicBatch, policy: Policy) -> EvaluationResult:
        torch = self.torch
        result = EvaluationResult(policy=policy, atom_count=batch.atom_count)
        result.metadata = self.manifest()
        try:
            from ase import Atoms
            if len(batch.positions) != len(batch.atomic_numbers):
                raise ValueError("positions and atomic_numbers must have equal length")
            atoms = Atoms(numbers=batch.atomic_numbers, positions=batch.positions,
                          cell=batch.cells, pbc=batch.pbc)
            self._attach_fresh_calculator(atoms)
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
                start_event.record()
            start = time.perf_counter()
            with precision_context(policy, self.device.split(":")[0]):
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
            if self.device.startswith("cuda"):
                end_event.record(); torch.cuda.synchronize()
                result.cuda_seconds = start_event.elapsed_time(end_event) / 1000
                result.allocated_bytes = torch.cuda.memory_allocated()
                result.peak_bytes = torch.cuda.max_memory_allocated()
            result.wall_seconds = time.perf_counter() - start
            result.energy = np.atleast_1d(energy).astype(float)
            result.forces = np.asarray(forces, dtype=float)
            result.finite = bool(np.isfinite(result.energy).all() and np.isfinite(result.forces).all())
        except (RuntimeError, NotImplementedError, TypeError) as e:
            # Any BF16-path failure is unsupported: never obscure it by retrying in FP32.
            result.unsupported = policy == "bf16_amp"
            result.error = f"{type(e).__name__}: {e}"
        result.metadata["result_adapter"] = result_adapter_diagnostics(self.calculator)
        return result

    def prepare_batch(self, batches: list[AtomicBatch]) -> PreparedMaceBatch:
        """Construct one genuine disconnected MACE/PyG graph batch."""
        if not batches:
            raise ValueError("cannot prepare an empty batch")
        if len(self.calculator.models) != 1:
            raise RuntimeError("Gate 1 batching currently requires exactly one MACE model")
        from ase import Atoms
        from mace import data as mace_data
        from mace.tools import torch_geometric, torch_tools

        calculator = self.calculator
        calculator.arrays_keys.update({calculator.charges_key: "charges"})
        keyspec = mace_data.KeySpecification(
            info_keys=calculator.info_keys, arrays_keys=calculator.arrays_keys
        )
        graphs, frame_ids, atom_counts = [], [], []
        with torch_tools.default_dtype(calculator.default_dtype):
            for index, batch in enumerate(batches):
                if len(batch.positions) != len(batch.atomic_numbers):
                    raise ValueError("positions and atomic_numbers must have equal length")
                atoms = Atoms(numbers=batch.atomic_numbers, positions=batch.positions,
                              cell=batch.cells, pbc=batch.pbc)
                config = mace_data.config_from_atoms(
                    atoms, key_specification=keyspec, head_name=calculator.head
                )
                graphs.append(mace_data.AtomicData.from_config(
                    config, z_table=calculator.z_table, cutoff=calculator.r_max,
                    heads=calculator.available_heads,
                ))
                frame_ids.append(batch.frame_ids[0] if batch.frame_ids else str(index))
                atom_counts.append(batch.atom_count)
        graph = torch_geometric.Batch.from_data_list(graphs).to(self.device)
        if int(graph.num_graphs) != len(batches):
            raise RuntimeError("MACE graph batching did not preserve the frame count")
        return PreparedMaceBatch(
            graph=graph,
            frame_ids=tuple(frame_ids),
            atom_counts=tuple(atom_counts),
            edge_count=int(graph["edge_index"].shape[1]),
        )

    def evaluate_prepared_batch(
        self, prepared: PreparedMaceBatch, policy: Policy
    ) -> BatchedEvaluationResult:
        """Evaluate a genuine disconnected graph batch in one model call."""
        torch = self.torch
        result = BatchedEvaluationResult(
            policy=policy, frame_ids=prepared.frame_ids,
            atom_counts=prepared.atom_counts, edge_count=prepared.edge_count,
            metadata=self.manifest(),
        )
        try:
            model = self.calculator.models[0]
            graph = prepared.graph.clone()
            model_dtype = next(model.parameters()).dtype
            for key in graph.keys:
                value = graph[key]
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    graph[key] = value.to(dtype=model_dtype)
            batch_dict = graph.to_dict()
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
                start_event.record()
            start = time.perf_counter()
            with precision_context(policy, self.device.split(":")[0]):
                out = model(
                    batch_dict, compute_stress=False, training=False,
                    compute_edge_forces=False, compute_atomic_stresses=False,
                )
                energies_tensor = out["energy"]
                forces_tensor = out["forces"]
                source_dtypes = {
                    "energy": str(energies_tensor.dtype),
                    "forces": str(forces_tensor.dtype),
                }
                energies = energies_tensor.detach().float().cpu().numpy()
                forces_flat = forces_tensor.detach().float().cpu().numpy()
            if self.device.startswith("cuda"):
                end_event.record()
                torch.cuda.synchronize()
                result.cuda_seconds = start_event.elapsed_time(end_event) / 1000
                result.allocated_bytes = torch.cuda.memory_allocated()
                result.peak_bytes = torch.cuda.max_memory_allocated()
            result.wall_seconds = time.perf_counter() - start
            energy_factor = self.calculator.energy_units_to_eV
            force_factor = energy_factor / self.calculator.length_units_to_A
            result.energies = np.asarray(energies, dtype=float).reshape(-1) * energy_factor
            offsets = np.cumsum((0,) + prepared.atom_counts)
            result.forces = tuple(
                np.asarray(forces_flat[offsets[i]:offsets[i + 1]], dtype=float) * force_factor
                for i in range(len(prepared.atom_counts))
            )
            result.finite = bool(
                len(result.energies) == len(prepared.frame_ids)
                and np.isfinite(result.energies).all()
                and all(np.isfinite(force).all() for force in result.forces)
            )
            result.metadata["batch_execution"] = {
                "genuine_disconnected_graph_batch": True,
                "num_graphs": len(prepared.frame_ids),
                "source_dtypes": source_dtypes,
                "output_dtype": "numpy.float32",
                "fp32_replay": False,
                "compute_stress": False,
            }
        except (RuntimeError, NotImplementedError, TypeError) as error:
            result.unsupported = policy == "bf16_amp"
            result.error = f"{type(error).__name__}: {error}"
        return result

    def manifest(self):
        torch = self.torch
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
        versions = {}
        for package in ("torch", "mace-torch", "ase", "numpy"):
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        return {"model_hash": self.model_hash, "versions": versions, "gpu": gpu,
                "cuda": torch.version.cuda, "python": platform.python_version()}
