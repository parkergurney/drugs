from __future__ import annotations

from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field, model_validator


class Gate1Config(BaseModel):
    dataset_id: str = "gate1"
    seed: int = 20260819
    output_dir: Path = Path("results/gate1")
    dataset_dir: Path = Path("data/rmd17")
    molecules: list[str] = ["ethanol", "malonaldehyde", "aspirin"]
    ordinary_counts: list[int] = [34, 33, 33]
    candidate_pool: int = 3000
    high_force_count: int = 100
    high_force_cap_per_molecule: int = 40
    close_contact_count: int = 100
    close_contact_distances: list[float] = [0.8, 1.0, 1.2]
    policies: list[Literal["fp32", "tf32", "bf16_amp"]] = ["fp32", "tf32", "bf16_amp"]
    batch_sizes: list[int] = [1, 8, 32]
    warmups: int = 20
    timed_iterations: int = 100
    device: str = "cuda"
    model: str = "small"
    exclude_selection: Path | None = None

    @model_validator(mode="after")
    def totals(self):
        if sum(self.ordinary_counts) != 100 or self.high_force_count != 100 or self.close_contact_count != 100:
            raise ValueError("Gate 1 requires exactly 100 frames in each stratum")
        if len(self.molecules) != len(self.ordinary_counts):
            raise ValueError("ordinary_counts must match molecules")
        return self


class Gate2Config(BaseModel):
    seed: int = 20260819
    output_dir: Path = Path("results/gate2")
    selected_policy: Literal["tf32", "bf16_amp"] | None = None
    temperature_k: float = 500.0
    equilibration_ps: float = 2.0
    production_ps: float = 5.0
    timestep_fs: float = 0.5
    checkpoints: int = 100
    block_lengths: list[int] = [10, 50, 100]
    train_checkpoints: int = 60


class D1Config(BaseModel):
    """Frozen inputs and measurement settings for the exploratory D1 study."""

    experiment_id: str = "D1-A40-P1-LOCALIZATION"
    dataset_id: str = "p1"
    dataset: Path = Path("artifacts/datasets/p1")
    c1_analysis: Path = Path("artifacts/analysis/c1-reproduction")
    output_dir: Path = Path("artifacts/diagnostics/d1-p1")
    device: str = "cuda"
    model: str = "small"
    expected_frames_sha256: str
    expected_evaluations_sha256: str
    expected_model_hash: str
    diagnostic_policies: list[Literal["fp64", "fp32", "tf32", "bf16_amp"]] = [
        "fp64", "fp32", "tf32", "bf16_amp"
    ]
    timing_policies: list[Literal["fp32", "tf32", "bf16_amp"]] = [
        "fp32", "tf32", "bf16_amp"
    ]
    selection_quantiles: list[float] = [0.0, 0.5, 0.95, 1.0]
    diagnostic_thresholds: dict[str, float] = {
        "tf32": 0.01,
        "bf16_amp": 0.08,
    }
    relative_rms_floor: float = 1e-12
    finite_difference_steps_a: list[float] = [1e-3, 3e-4, 1e-4]
    batch_sizes: list[int] = [1, 8, 32]
    warmups: int = 20
    timed_iterations: int = 100
    trace_repeats: int = 1
    seed: int = 2026082801

    @model_validator(mode="after")
    def d1_invariants(self):
        if self.diagnostic_policies[0] != "fp64" or "fp32" not in self.diagnostic_policies:
            raise ValueError("D1 diagnostics require FP64 first and an FP32 reference")
        if self.timing_policies != ["fp32", "tf32", "bf16_amp"]:
            raise ValueError("D1 timing policies are frozen as FP32, TF32, and BF16 AMP")
        if self.selection_quantiles != [0.0, 0.5, 0.95, 1.0]:
            raise ValueError("D1 selection quantiles are frozen at 0, .5, .95, and 1")
        if self.batch_sizes != [1, 8, 32]:
            raise ValueError("D1 batch sizes are frozen at 1, 8, and 32")
        hashes = (
            self.expected_frames_sha256,
            self.expected_evaluations_sha256,
            self.expected_model_hash,
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in hashes
        ):
            raise ValueError("D1 expected hashes must be SHA-256 hex digests")
        return self


def load_config(path: str | Path, cls):
    with Path(path).open("r", encoding="utf-8") as f:
        return cls.model_validate(yaml.safe_load(f))
