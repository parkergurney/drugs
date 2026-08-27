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

## Running independent trials

Prepare and checksum the 300-frame dataset once. Then point every fresh
benchmark process at that immutable file, give it a unique run ID, and vary
only the timing seed used to randomize policy order:

```bash
uv run precision-md benchmark --config configs/gate1.yaml \
  --frames results/gate1/frames.npz --run-id c1-run-01 \
  --timing-seed 2026081901 --allow-gpu-benchmark
uv run precision-md benchmark --config configs/gate1.yaml \
  --frames results/gate1/frames.npz --run-id c1-run-02 \
  --timing-seed 2026081902 --allow-gpu-benchmark
```

With the committed Gate 1 configuration these write isolated results beneath
`results/gate1/<run-id>/`. A benchmark refuses to overwrite an existing trial.
Run additional processes in the same way, then combine them with the
process-first hierarchical bootstrap:

```bash
uv run precision-md analyze-trials \
  --trials results/gate1 --output results/c1-analysis
```

Every trial manifest records its run ID, timing seed, frozen-frame path and
SHA-256, experiment/config/Git provenance, and model hash. Multi-trial analysis
rejects mixed frame, model, configuration, dataset, or experiment hashes. When
`--run-id` is used, the benchmark copies the canonical `frames.npz` into the
trial and verifies its hash.

Create a portable immutable dataset bundle with:

```bash
uv run precision-md freeze-dataset \
  --source data/frozen/p1 --output artifacts/datasets/p1 \
  --dataset-id p1 --provenance studies/pilot-p1-a40/manifest.json
uv run precision-md validate-dataset \
  --dataset artifacts/datasets/p1 --dataset-id p1
```

Each bundle contains `frames.npz`, `selection.json`,
`candidate_scores.parquet`, `manifest.json`, and `SHA256SUMS`. Freezing is
atomic and idempotent for matching content and refuses to replace a different
dataset.

On an A40, `scripts/run-a40-c1-reproduction.sh` performs the guarded complete
workflow: it imports P1, prepares and freezes the disjoint C1 dataset once,
packages C1 immediately, runs all five isolated P1 reproduction processes
sequentially, samples GPU telemetry once per second, analyzes the trials, and
packages the result tree. Complete verified trials are resumable; partial or
inconsistent trials are never overwritten. It refuses a dirty tracked
worktree, a non-A40 GPU, altered inputs, or inconsistent artifacts.

```text
artifacts/
  datasets/{p1,c1}/
  trials/c1-reproduction/c1-run-*/
  analysis/c1-reproduction/
  system/
  bundles/
```

## Preparing the sealed C1 dataset

Recover the exact P1 `frames.npz`, `selection.json`, and
`candidate_scores.parquet` into `data/frozen/p1/` before GPU work. The C1
configuration selects a new dataset while excluding every molecular source
frame represented in P1:

```bash
uv run precision-md prepare-data --config configs/c1-dataset.yaml
```

This writes the resumable staging directory
`results/datasets/c1-confirmatory/`, including a dataset manifest
with source-data, selection, frame, model, and exclusion checksums. Preparation
fails on any P1 source-frame overlap. C1 preparation uses FP32 scoring only;
do not benchmark this dataset under TF32 or BF16 until the confirmatory energy
and force margins are frozen in a dated protocol amendment. Freeze the
completed staging directory with:

```bash
uv run precision-md freeze-dataset \
  --source results/datasets/c1-confirmatory \
  --output artifacts/datasets/c1 --dataset-id c1-confirmatory
```

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
