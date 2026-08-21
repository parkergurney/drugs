import pytest

torch = pytest.importorskip("torch")

from precision_md.mace_adapter import (install_fp32_result_adapter,
                                       result_adapter_diagnostics)


class FakeCalculator:
    device = "cpu"

    def _create_result_tensors(self, num_models, num_atoms, batch, out):
        del batch
        return {
            "energy": torch.zeros(num_models, dtype=out["energy"].dtype),
            "forces": torch.zeros(num_models, num_atoms, 3, dtype=out["forces"].dtype),
        }, None


def test_bf16_results_are_promoted_without_replay():
    calculator = install_fp32_result_adapter(FakeCalculator(), check_version=False)
    out = {
        "energy": torch.ones((), dtype=torch.bfloat16),
        "forces": torch.ones(2, 3, dtype=torch.bfloat16),
    }

    tensors, _ = calculator._create_result_tensors(1, 2, None, out)
    diagnostics = result_adapter_diagnostics(calculator)

    assert tensors["energy"].dtype == torch.float32
    assert tensors["forces"].dtype == torch.float32
    assert diagnostics["source_dtypes"] == {
        "energy": "torch.bfloat16",
        "forces": "torch.bfloat16",
    }
    assert diagnostics["output_cast_keys"] == ["energy", "forces"]
    assert diagnostics["fp32_replay"] is False


def test_fp32_results_are_not_reported_as_casts():
    calculator = install_fp32_result_adapter(FakeCalculator(), check_version=False)
    out = {
        "energy": torch.ones((), dtype=torch.float32),
        "forces": torch.ones(2, 3, dtype=torch.float32),
    }

    tensors, _ = calculator._create_result_tensors(1, 2, None, out)

    assert all(value.dtype == torch.float32 for value in tensors.values())
    assert result_adapter_diagnostics(calculator)["output_cast_keys"] == []


def test_fp64_results_are_preserved():
    calculator = install_fp32_result_adapter(FakeCalculator(), check_version=False)
    out = {
        "energy": torch.ones((), dtype=torch.float64),
        "forces": torch.ones(2, 3, dtype=torch.float64),
    }

    tensors, _ = calculator._create_result_tensors(1, 2, None, out)

    assert all(value.dtype == torch.float64 for value in tensors.values())
    assert result_adapter_diagnostics(calculator)["output_cast_keys"] == []
