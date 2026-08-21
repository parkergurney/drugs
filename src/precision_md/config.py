from __future__ import annotations

from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field, model_validator


class Gate1Config(BaseModel):
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


def load_config(path: str | Path, cls):
    with Path(path).open("r", encoding="utf-8") as f:
        return cls.model_validate(yaml.safe_load(f))
