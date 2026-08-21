from pathlib import Path


def test_batch_validation_script_is_committed():
    script = Path(__file__).parents[1] / "scripts" / "validate-gate1-batching.py"

    assert script.exists()
    assert "combined FP32 batch" in script.read_text()
