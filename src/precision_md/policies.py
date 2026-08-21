from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

Policy = Literal["fp32", "tf32", "bf16_amp"]


@contextmanager
def precision_context(policy: Policy, device: str = "cuda"):
    """Set and restore all global precision flags; never silently downgrade."""
    import torch
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = policy == "tf32"
        torch.backends.cudnn.allow_tf32 = policy == "tf32"
        if policy == "bf16_amp":
            if device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("complete CUDA BF16 autocast path is unsupported")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                yield
        else:
            with torch.autocast(device_type=device.split(":")[0], enabled=False):
                yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
