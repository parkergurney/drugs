from __future__ import annotations

import importlib.metadata
from types import MethodType

SUPPORTED_MACE_VERSION = "0.3.16"


def install_fp32_result_adapter(calculator, *, check_version: bool = True):
    """Make MACE's completed floating results FP32 before NumPy conversion.

    MACE 0.3.16 allocates its ensemble/result buffers with the model output
    dtype. Under BF16 autocast those buffers remain BF16 and its later direct
    ``.numpy()`` conversion fails. This adapter disables autocast only for the
    atomic baseline/result-buffer construction and promotes completed floating
    result buffers to FP32. It does not rerun the model or alter model weights.
    """
    if check_version:
        installed = importlib.metadata.version("mace-torch")
        if installed != SUPPORTED_MACE_VERSION:
            raise RuntimeError(
                f"BF16 result adapter supports mace-torch {SUPPORTED_MACE_VERSION}, "
                f"found {installed}"
            )
    if getattr(calculator, "_precision_md_result_adapter", False):
        return calculator

    import torch

    original = calculator._create_result_tensors

    def create_result_tensors(self, num_models, num_atoms, batch, out):
        device_type = str(self.device).split(":", maxsplit=1)[0]
        # The atomic baseline is an output-side calculation. Running it outside
        # autocast also prevents its direct NumPy conversion from seeing BF16.
        with torch.autocast(device_type=device_type, enabled=False):
            result_tensors, node_e0 = original(num_models, num_atoms, batch, out)

        source_dtypes = {
            key: str(value.dtype)
            for key, value in out.items()
            if torch.is_tensor(value) and torch.is_floating_point(value)
        }
        cast_keys = []
        low_precision_dtypes = (torch.bfloat16, torch.float16)
        for key, value in tuple(result_tensors.items()):
            if torch.is_floating_point(value) and value.dtype in low_precision_dtypes:
                result_tensors[key] = value.float()
                cast_keys.append(key)

        self._precision_md_source_dtypes = source_dtypes
        self._precision_md_output_cast_keys = tuple(sorted(cast_keys))
        self._precision_md_fp32_replay = False
        return result_tensors, node_e0

    calculator._create_result_tensors = MethodType(create_result_tensors, calculator)
    calculator._precision_md_result_adapter = True
    calculator._precision_md_source_dtypes = {}
    calculator._precision_md_output_cast_keys = ()
    calculator._precision_md_fp32_replay = False
    return calculator


def result_adapter_diagnostics(calculator):
    return {
        "enabled": bool(getattr(calculator, "_precision_md_result_adapter", False)),
        "source_dtypes": dict(getattr(calculator, "_precision_md_source_dtypes", {})),
        "output_cast_keys": list(getattr(calculator, "_precision_md_output_cast_keys", ())),
        "low_precision_output_cast_dtype": "torch.float32",
        "fp32_replay": bool(getattr(calculator, "_precision_md_fp32_replay", False)),
        "supported_mace_version": SUPPORTED_MACE_VERSION,
    }
