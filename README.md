# precision-md

`precision-md` is a reproducible study of when reduced-precision inference is
both faster and numerically reliable for pretrained machine-learning
interatomic potentials. The current implementation evaluates MACE-OFF23-small
under FP32, TF32, and BF16-autocast policies on stratified molecular
configurations.

The completed NVIDIA A40 pilot found no viable reduced-precision policy for the
tested model and workload. TF32 was effectively speed-neutral, while BF16 was
slower and produced six nonfinite evaluations. This is a bounded pilot result,
not a claim about all MLIPs or hardware.

## Repository layout

```text
configs/       Versioned experiment configurations
paper/         Manuscript source
scripts/       Reproducible GPU entry points and validation utilities
src/           Installable precision_md package
studies/       Frozen study records and prospective protocols
tests/         CPU unit and workflow tests
REFERENCES.md  Literature source index
```

Raw datasets, model checkpoints, generated results, downloaded papers, and
complete result bundles are intentionally excluded from Git. Frozen studies
commit compact metadata, summaries, checksums, and retrieval information.

## Installation

Python 3.11 is the reference environment. Install the CPU test environment:

```bash
uv sync --extra test
uv run pytest
```

For CUDA/MACE experiments:

```bash
uv sync --extra ml --extra test
uv run precision-md --help
```

MACE checkpoints and rMD17 archives must be supplied locally. Expected rMD17
files are `data/rmd17/{ethanol,malonaldehyde,aspirin}.npz`, each containing `R`
and either `z` or `nuclear_charges`.

## Running Gate 1

GPU benchmarking is deliberately guarded against accidental CPU execution:

```bash
uv run precision-md prepare-data --config configs/gate1.yaml
uv run precision-md benchmark --config configs/gate1.yaml --allow-gpu-benchmark
uv run precision-md analyze --results results
uv run precision-md render-report --results results --output report/decision.md
```

The convenience entry point `scripts/run-gpu-gate1.sh` performs the same
sequence. Record the Git commit, configuration hash, model hash, environment,
GPU telemetry, random seed, command, and output checksums for every scientific
run.

## Research provenance

Pilot P1 is documented in
[`studies/pilot-p1-a40/`](studies/pilot-p1-a40/README.md). The exact code state
that produced it was legacy commit `bc48cb5`. The pre-reset history containing
that commit is preserved separately with the pilot artifacts; its bundle
checksum is recorded in the study. The study directory was added afterward to
record the result and does not claim to be the executing commit.

The next experiment is prospectively defined in
[`studies/confirmatory-c1/protocol.md`](studies/confirmatory-c1/protocol.md).
The A100 configuration is confirmation-only and should not be interpreted as a
substitute for the independent A40 reproduction.

## Reproducibility policy

- Treat numerical failures, nonfinite values, and out-of-memory events as
  outcomes rather than exclusions.
- Preserve pilot, exploratory, and confirmatory labels.
- Freeze primary metrics, margins, exclusions, and stopping rules before
  examining confirmatory outcomes.
- Report ordinary, high-force, and deliberately adversarial close-contact
  configurations separately.
- Use FP64 only as a diagnostic where supported; FP32 is the operational
  baseline, not physical ground truth.
- Store large immutable artifacts in a release or research-data repository and
  identify them here by checksum and persistent URL or DOI.

## Citation

Citation metadata will be added when the manuscript is submitted. Literature
used to motivate and design the study is listed in [REFERENCES.md](REFERENCES.md).
