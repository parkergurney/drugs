# D1 — P1 failure localization on NVIDIA A40

## Status

Complete exploratory mechanism study. D1 explains the performance and
numerical behavior observed in P1 and the independent C1 reproduction on the
frozen P1 frames. It is not a confirmatory accuracy test and does not change
C1's conclusions.

## Provenance

- Experiment ID: `D1-A40-P1-LOCALIZATION`
- Timing commit and tag: `0257594121043ecd35e72c8f91f0d3befbf844eb`,
  `d1-p1-preflight-v1`
- Corrected diagnostic/analysis commit and tag:
  `e2707b3adc29774d61142f79c75b673a21d5f4c1`,
  `d1-p1-tracer-fix-v2`
- Configuration SHA-256:
  `8cd269786419937aaac3a5efb6bdbd478699fcdb459390986f76982d141e96af`
- P1 frames SHA-256:
  `f7e759b6f0050b82eae88ff99416a2d43f50eac9e2e944a7524e80eaff40a28d`
- C1 evaluations SHA-256:
  `da741dc69df6052f83c2036282af52bb1b403eb85c4ee4fb3aa8f1ac55cedaa8`
- Model SHA-256:
  `9bd176f569bb26925f5d8ae7779e01babafaed42bece49f34cb1f561925a8149`

The full final bundle and the pre-fix raw bundle are intentionally excluded
from Git. Their locally verified hashes are recorded in `manifest.json`.

## Environment

- Provider: RunPod
- GPU: NVIDIA A40, 46,068 MiB
- Driver: 570.195.03
- PyTorch CUDA runtime: 12.1
- Python: 3.11.10
- PyTorch: 2.4.1
- MACE: 0.3.16
- ASE: 3.29.0
- NumPy: 2.4.6
- Platform: Linux 6.8.0-64-generic x86_64, glibc 2.35

## Design

D1 deterministically selected 27 diagnostic frames from the frozen P1 dataset
and combined C1 results. The selection included every reduced-policy nonfinite
frame plus frozen finite-error quantiles by policy and stratum. One process
traced FP64, FP32, TF32, and BF16 AMP. Three independent, uninstrumented
processes measured FP32, TF32, and BF16 AMP at batch sizes 1, 8, and 32, using
20 warm-ups and 100 measured iterations.

Module-boundary discrepancy flags were frozen at 0.01 relative RMS for TF32
and 0.08 for BF16 AMP. These are exploratory localization thresholds, not
scientific acceptance margins.

## Instrumentation correction

The first diagnostic attempt failed uniformly before model evaluation because
ordinary PyTorch hooks cannot be registered on MACE TorchScript submodules.
Every operator-trace table was empty and every policy reported the same setup
exception. No localization outcome existed to inspect.

The three completed timing processes were preserved. The correction skipped
unsupported ScriptModule hooks, recorded skipped modules, retained eager
boundary and targeted operation tracing, and made empty traces an immediate
error. Only the invalid diagnostic process and derived analysis were rerun.
The original failure is retained in the checksummed pre-fix raw archive. The
prospectively selected frames, thresholds, timing code, and timing results were
not changed.

## Results

All six localized nonfinite cases occurred under BF16 AMP on selected
close-contact frames. The first observed nonfinite boundaries were dense
`aten.mm` contractions. Eight BF16 cases first crossed the frozen discrepancy
threshold at an earlier `aten.add.Tensor` accumulation boundary. TF32 produced
no localized nonfinite case or frozen trace-threshold crossing.

Neither reduced policy accelerated the complete workload. TF32 remained near
parity with FP32, with every reported end-to-end confidence interval including
1.0. BF16 AMP was consistently slower:

| Scope | Batch | TF32 FP32/policy (95% CI) | BF16 FP32/policy (95% CI) |
|---|---:|---:|---:|
| Prepared model | 1 | 1.004 (0.966–1.041) | 0.895 (0.860–0.931) |
| Prepared model | 8 | 1.008 (0.972–1.048) | 0.915 (0.854–0.985) |
| Prepared model | 32 | 0.968 (0.896–1.034) | 0.901 (0.859–0.943) |
| Coordinate to result | 1 | 1.007 (0.977–1.039) | 0.962 (0.917–1.009) |
| Coordinate to result | 8 | 0.995 (0.967–1.020) | 0.955 (0.930–0.983) |
| Coordinate to result | 32 | 0.998 (0.951–1.045) | 0.897 (0.845–0.943) |

Component timing localized the BF16 performance deficit to energy-forward and
force-gradient execution rather than graph construction or transfers. At
batch 32, BF16's FP32/policy ratio was 0.828 for energy forward and 0.891 for
the force gradient.

## Interpretation

D1 supports a narrower thesis than uniform low-precision acceleration:
generic BF16 autocast is both slower and numerically unsafe for this locked
MACE/A40 workload, while TF32 is safer but provides no useful speedup. The next
stage should therefore test operator-aware FP32 protection and optimized
equivariant kernels one factor at a time.

The first observed `aten.mm` nonfinite is not automatically the root cause.
The earlier accumulation discrepancy may feed unsafe values into the dense
contraction. That causal distinction belongs to A1 ablations.

## Limits

The 27 frames were selected to localize difficult behavior, so six failures in
27 cases is not a population failure-rate estimate. Close-contact geometries
are deliberately adversarial and showed sensitivity even in FP32 relative to
FP64. Trace boundaries came from one instrumented process; the three-process
replication applies to timing, not localization. Operation sequence numbers
are trace-local identifiers rather than stable semantic MACE layer names.

## Artifacts

`manifest.json` is the compact machine-readable study record. The immutable
Parquet tables, telemetry, per-run manifests, report, and checksum manifest are
stored in `precision-md-d1-a40-results.tar.gz`. Add a persistent archive URL or
DOI to `manifest.json` before publication.
