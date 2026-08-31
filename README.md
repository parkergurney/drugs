# PreciseMD

[![CI](https://github.com/parkergurney/PreciseMD/actions/workflows/ci.yml/badge.svg)](https://github.com/parkergurney/PreciseMD/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PreciseMD benchmarks the performance and numerical reliability of
reduced-precision machine-learning interatomic potentials. The current study
evaluates MACE-OFF23-small under FP32, TF32, and BF16 autocast on an NVIDIA
A40.

## Results

| Policy | Performance | Numerical behavior | Outcome |
|---|---|---|---|
| FP32 | Baseline | Stable on ordinary and high-force diagnostic frames | Retained |
| TF32 | No measurable end-to-end speedup | Finite on all 27 D1 frames; sensitive on adversarial close contacts | No deployment benefit |
| BF16 autocast | 9–12% slower for prepared-model inference | Six localized nonfinite close-contact cases | Rejected as a blanket policy |

BF16 component timing localized the slowdown to energy-forward and
force-gradient execution rather than graph construction or transfers. The
first large traced discrepancy appeared at an addition/accumulation boundary.
The first nonfinite values appeared later in dense matrix multiplications.
These are first observed boundaries, not proof of root cause.

The result is limited to the tested model, software stack, molecular systems,
and A40 GPU. The 27 D1 frames were selected for failure localization and do not
provide a population failure-rate estimate.

## Study structure

- **P1:** 300-frame pilot across ordinary, high-force, and constructed
  close-contact configurations.
- **C1:** five independent A40 benchmark processes with process-first
  hierarchical confidence intervals.
- **D1:** three independent component-timing processes and one instrumented
  failure-localization process.
- **A1:** planned operator-level FP32-protection and kernel ablations.

The [confirmatory protocol](studies/confirmatory-c1/protocol.md) defines the
experimental design and decision rules. Completed records are available for
[P1](studies/pilot-p1-a40/README.md) and
[D1](studies/d1-p1-a40/README.md).

## Features

- Isolated output directories for independent GPU processes.
- Immutable datasets with SHA-256 validation.
- Model, configuration, environment, and Git provenance.
- Process-first hierarchical bootstrap analysis.
- Continuous GPU telemetry.
- Energy, force, finite-difference, equivariance, and batching diagnostics.
- MACE module-boundary and PyTorch operation tracing.
- Resumable workflows that preserve nonfinite outputs and failed runs.

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/) are the reference environment.

```bash
git clone https://github.com/parkergurney/PreciseMD.git
cd PreciseMD
uv sync --extra test
uv run pytest
uv run precisemd --help
```

The historical `precision-md` command remains available for compatibility with
frozen experiment manifests.

Install the MACE/CUDA dependencies with:

```bash
uv sync --extra ml --extra test
```

## Data

rMD17 inputs are not redistributed. Place the required archives at:

```text
data/rmd17/ethanol.npz
data/rmd17/malonaldehyde.npz
data/rmd17/aspirin.npz
```

Each archive must contain `R` and either `z` or `nuclear_charges`.

## GPU workflows

```bash
# Five-process C1 reproduction and combined analysis
scripts/run-a40-c1-reproduction.sh

# D1 timing, tracing, analysis, validation, and packaging
scripts/run-a40-d1.sh
```

The scripts validate the Git tag, GPU model, input hashes, model hash,
telemetry, trial isolation, and final checksums. Matching completed work is
resumable; inconsistent artifacts are not overwritten.

Lower-level commands include:

```bash
uv run precisemd prepare-data --config configs/gate1.yaml
uv run precisemd benchmark --config configs/gate1.yaml --allow-gpu-benchmark
uv run precisemd analyze-trials --trials results/gate1 --output results/c1-analysis
uv run precisemd validate-d1 --config configs/d1-p1.yaml
```

## Repository layout

```text
configs/       Experiment configurations
paper/         Working manuscript
scripts/       Guarded GPU workflows
src/           Python package
studies/       Protocol and completed study records
tests/         Unit and workflow tests
REFERENCES.md  Literature index
```

Raw datasets, model checkpoints, generated outputs, and complete result bundles
are excluded from Git. Study manifests record their checksums and provenance.

## Diagnostic correction

The first D1 diagnostic process produced empty traces because PyTorch forward
hooks are unsupported on MACE TorchScript submodules. The unaffected timing
processes were retained. The failed trace was archived, the instrumentation was
corrected under a new commit and tag, and only the diagnostic process and
derived analysis were rerun. Empty traces now fail during collection.

## Next step

A1 will test selective FP32 protection for accumulation, contraction, geometry,
reduction, and force-gradient operations. Compilation and optimized
equivariant kernels will be evaluated as separate interventions.

PreciseMD is released under the [MIT License](LICENSE). Citation metadata is in
[CITATION.cff](CITATION.cff). Supporting literature is listed in
[REFERENCES.md](REFERENCES.md).
