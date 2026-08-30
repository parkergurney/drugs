from __future__ import annotations

import importlib.metadata
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mace_adapter import SUPPORTED_MACE_VERSION


class TraceSetupError(RuntimeError):
    """Raised when D1 cannot install its required eager-module probes."""


def _tensor_items(value: Any, path: str = "value"):
    """Yield tensors from nested MACE inputs and outputs with stable paths."""
    try:
        import torch
    except ImportError:
        return
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _tensor_items(value[key], f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_items(item, f"{path}.{index}")


def _summary(tensor) -> dict:
    array = tensor.detach().double().cpu().numpy()
    finite = np.isfinite(array)
    values = array[finite]
    return {
        "shape": json_shape(array.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "element_count": int(array.size),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(array.size - finite.sum()),
        "minimum": float(values.min()) if values.size else np.nan,
        "maximum": float(values.max()) if values.size else np.nan,
        "mean": float(values.mean()) if values.size else np.nan,
        "rms": float(np.sqrt(np.mean(values * values))) if values.size else np.nan,
    }


def json_shape(shape) -> str:
    return "[" + ",".join(str(int(value)) for value in shape) + "]"


def _operation_group(boundary: str, module_class: str) -> str:
    text = f"{boundary} {module_class}".lower()
    if boundary.startswith("geometry."):
        return "geometry"
    if "radial" in text or "bessel" in text or "cutoff" in text:
        return "radial_and_cutoff"
    if "spherical" in text:
        return "spherical_harmonics"
    if "scatter_sum" in text:
        return "scatter_reduction"
    if "scatter_add" in text:
        return "scatter_reduction"
    if "tensorproduct" in text or "contraction" in text or "products." in text:
        return "equivariant_contraction"
    if "linear" in text or "readouts." in text or "node_embedding" in text:
        return "linear_projection"
    if "scale_shift" in text:
        return "normalization_and_rescaling"
    if "interaction_energy" in text or "total_energy" in text or "atomic_energies" in text:
        return "energy_accumulation"
    if "forces" in text:
        return "force_gradient"
    if boundary.startswith("aten."):
        if any(name in text for name in ("mm", "matmul", "einsum")):
            return "dense_contraction"
        if any(name in text for name in ("norm", "sqrt", "div", "reciprocal")):
            return "normalization_and_rescaling"
        if any(name in text for name in ("sum", "add")):
            return "reduction_or_accumulation"
        return "aten_arithmetic"
    if "result_conversion" in text:
        return "result_conversion"
    if "result_transfer" in text:
        return "device_transfer"
    if "interactions." in text:
        return "message_passing"
    return "other"


@dataclass
class TraceCollector(AbstractContextManager):
    """Non-mutating module-boundary observer for the locked MACE model."""

    model: Any
    policy: str
    frame_id: str
    repeat: int = 0
    trace_level: str = "coarse"
    rows: list[dict] = field(default_factory=list)
    tensors: dict[tuple, Any] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list)
    _order: int = 0
    _scatter_originals: list[tuple[Any, Any]] = field(default_factory=list)
    _scatter_counts: dict[str, int] = field(default_factory=dict)
    skipped_script_modules: list[str] = field(default_factory=list)

    def __post_init__(self):
        installed = importlib.metadata.version("mace-torch")
        if installed != SUPPORTED_MACE_VERSION:
            raise RuntimeError(
                f"D1 probe supports mace-torch {SUPPORTED_MACE_VERSION}, found {installed}"
            )
        if type(self.model).__name__ != "ScaleShiftMACE":
            raise RuntimeError(
                f"D1 probe requires ScaleShiftMACE, found {type(self.model).__name__}"
            )
        for attribute in (
            "atomic_energies_fn", "node_embedding", "spherical_harmonics",
            "radial_embedding", "interactions", "products", "readouts", "scale_shift",
        ):
            if not hasattr(self.model, attribute):
                raise RuntimeError(f"D1 probe model is missing {attribute}")

    @staticmethod
    def _wanted(name: str) -> bool:
        roots = (
            "atomic_energies_fn", "node_embedding", "spherical_harmonics",
            "radial_embedding", "interactions.", "products.", "readouts.", "scale_shift",
        )
        return any(name == root.rstrip(".") or name.startswith(root) for root in roots)

    def _record(self, module_name: str, module, role: str, value: Any):
        import torch

        for path, tensor in _tensor_items(value, role):
            if not tensor.is_floating_point():
                continue
            key = (self._order, module_name, role, path)
            with torch._C._DisableTorchDispatch():
                summary = _summary(tensor)
                stored = tensor.detach().double().cpu()
            row = {
                "frame_id": self.frame_id,
                "policy": self.policy,
                "repeat": self.repeat,
                "trace_level": self.trace_level,
                "boundary_order": self._order,
                "boundary": module_name,
                "module_class": type(module).__name__,
                "operation_group": _operation_group(
                    module_name, type(module).__name__
                ),
                "role": role,
                "tensor_path": path,
                "accumulation_dtype": "unknown",
                "accumulation_evidence": "not observable at module boundary",
            } | summary
            self.rows.append(row)
            self.tensors[key] = stored
            self._order += 1

    def __enter__(self):
        import torch

        for name, module in self.model.named_modules():
            if not name or not self._wanted(name):
                continue
            # PyTorch deliberately rejects ordinary forward hooks on compiled
            # ScriptModules. Observe those operations through the nearest eager
            # parent boundary and the targeted TorchDispatch probe instead.
            if isinstance(module, torch.jit.ScriptModule):
                self.skipped_script_modules.append(name)
                continue
            installed = []
            try:
                installed.append(module.register_forward_pre_hook(
                    lambda mod, args, module_name=name: self._record(
                        module_name, mod, "input", args
                    )
                ))
                installed.append(module.register_forward_hook(
                    lambda mod, args, output, module_name=name: self._record(
                        module_name, mod, "output", output
                    )
                ))
            except Exception as exc:
                for handle in reversed(installed):
                    handle.remove()
                self.__exit__(None, None, None)
                raise TraceSetupError(
                    f"failed to install D1 hooks on {name} "
                    f"({type(module).__name__}): {exc}"
                ) from exc
            self._handles.extend(installed)
        # MACE scatter reductions are functions rather than Modules, so regular
        # forward hooks cannot see them. Wrap the two imported references only
        # for the lifetime of this trace and always restore them in __exit__.
        from mace.modules import blocks, models
        for namespace_name, namespace in (("blocks", blocks), ("models", models)):
            original = namespace.scatter_sum
            self._scatter_originals.append((namespace, original))

            def traced_scatter(*args, _original=original, _name=namespace_name, **kwargs):
                source = kwargs.get("src", args[0] if args else None)
                count = self._scatter_counts.get(_name, 0)
                self._scatter_counts[_name] = count + 1
                boundary = f"scatter_sum.{_name}.{count}"
                if source is not None:
                    self._record(boundary, self.model, "input", source)
                output = _original(*args, **kwargs)
                before = len(self.rows)
                self._record(boundary, self.model, "output", output)
                for row in self.rows[before:]:
                    row["accumulation_dtype"] = str(output.dtype)
                    row["accumulation_evidence"] = "torch.scatter_add output buffer dtype"
                return output

            namespace.scatter_sum = traced_scatter
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for namespace, original in reversed(self._scatter_originals):
            namespace.scatter_sum = original
        self._scatter_originals.clear()
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        return False


def targeted_operation_mode(collector: TraceCollector):
    """Return a scoped PyTorch dispatcher for implicated D1 cases only."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    fragments = (
        "mm", "matmul", "einsum", "scatter_add", "linalg_vector_norm",
        "div", "mul", "add", "sub", "sum", "pow", "sqrt", "reciprocal",
        "sin", "cos", "exp", "stack", "cat",
    )

    class D1OperationMode(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            name = str(func)
            wanted = any(fragment in name for fragment in fragments)
            boundary = f"aten.{name}.{self.count}"
            if wanted:
                self.count += 1
                with torch._C._DisableTorchDispatch():
                    collector._record(boundary, collector.model, "input", (args, kwargs))
            output = func(*args, **kwargs)
            if wanted:
                with torch._C._DisableTorchDispatch():
                    collector._record(boundary, collector.model, "output", output)
            return output

    return D1OperationMode()


def add_geometry_trace(collector: TraceCollector, graph) -> None:
    """Capture the exact pre-model geometry operations used by MACE."""
    import torch

    positions = graph["positions"]
    edge_index = graph["edge_index"]
    vectors = positions[edge_index[1]] - positions[edge_index[0]] + graph["shifts"]
    lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    sentinel = collector.model
    collector._record("geometry.positions", sentinel, "output", positions)
    collector._record("geometry.neighbor_vectors", sentinel, "output", vectors)
    collector._record("geometry.distances", sentinel, "output", lengths)


def compare_trace_to_fp32(
    rows: list[dict], tensors: dict[tuple, Any], reference, floor: float = 1e-12
) -> None:
    """Attach comparisons using stable module/role/path call occurrence."""
    reference_rows, reference_tensors = reference
    def signature(row):
        return (row["boundary"], row["role"], row["tensor_path"])
    reference_by_signature: dict[tuple, list[tuple[dict, Any]]] = {}
    for row, tensor in zip(reference_rows, reference_tensors.values(), strict=False):
        reference_by_signature.setdefault(signature(row), []).append((row, tensor))
    seen: dict[tuple, int] = {}
    for row, tensor in zip(rows, tensors.values(), strict=False):
        sig = signature(row)
        occurrence = seen.get(sig, 0)
        seen[sig] = occurrence + 1
        choices = reference_by_signature.get(sig, [])
        row["max_abs_difference_from_fp32"] = np.nan
        row["relative_rms_from_fp32"] = np.nan
        if occurrence >= len(choices):
            continue
        reference_tensor = choices[occurrence][1]
        if tuple(reference_tensor.shape) != tuple(tensor.shape):
            continue
        difference = tensor.numpy() - reference_tensor.numpy()
        finite = np.isfinite(difference)
        if not finite.any():
            continue
        difference = difference[finite]
        reference_values = reference_tensor.numpy()
        reference_values = reference_values[np.isfinite(reference_values)]
        row["max_abs_difference_from_fp32"] = float(np.max(np.abs(difference)))
        denominator = max(
            float(np.sqrt(np.mean(reference_values * reference_values)))
            if reference_values.size else 0.0,
            floor,
        )
        row["relative_rms_from_fp32"] = float(
            np.sqrt(np.mean(difference * difference)) / denominator
        )
