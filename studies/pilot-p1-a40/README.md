# Pilot P1 — MACE-OFF23(S) on NVIDIA A40

## Status

Frozen pilot record. This experiment motivated the confirmatory protocol and
must not be counted as an independent confirmatory replicate.

## Provenance

- Completion time: 2026-08-19 19:04:36 UTC
- Legacy repository commit: `bc48cb550aa58c4588ca8822c2076d83fc0b182f`
- Pre-reset history bundle SHA-256:
  `26456048103ac761a9460c4a8ab200a579de619ce5032a4322833106bb0d187a`
- Result archive: `precision-md-a40-gate1-stop.tar.gz`
- Archive SHA-256 observed locally: `f487fb4bb969478c96a897e77acf612867f2ff65a8a737e32b9a04e5a1122419`
- Internal checksum manifest: `precision-md-results-SHA256SUMS.txt`
- Integrity status: archive hash recorded; full extraction and verification of
  every internal file remains a confirmatory-preparation task.

This directory was added after execution to archive the scientific record. The
legacy commit above identifies the code and configuration that produced the
measurements and remains recoverable from the checksummed history bundle. The
commit containing this directory records the results but did not itself produce
them.

## Environment

- GPU: NVIDIA A40
- CUDA runtime reported by PyTorch: 12.1
- Python: 3.11.13
- PyTorch: 2.4.1
- MACE: 0.3.16
- ASE: 3.29.0
- NumPy: 2.4.6
- Platform: Linux 6.8.0-134-generic x86_64, glibc 2.39
- Model: MACE-OFF23 small
- Model SHA-256: `9bd176f569bb26925f5d8ae7779e01babafaed42bece49f34cb1f561925a8149`
- cuEquivariance: unavailable; acceleration disabled

## Design

The benchmark used rMD17 configurations of ethanol, malonaldehyde, and aspirin.
The 300-frame accuracy set contained:

- 100 deterministically sampled ordinary frames;
- 100 high-force frames selected from a 3,000-frame candidate pool, capped at
  40 frames per molecule;
- 100 constructed close-contact frames at target distances of 0.8, 1.0, and
  1.2 angstrom.

The policies were FP32, TF32, and BF16 autocast. Timing used genuine prepared
disconnected graph batches of 1, 8, and 32, with 20 warm-ups and 100 measured
iterations. The timed scope was the model call plus output transfer. The random
seed was `20260819`.

The preregistered Gate 1 requirements for a reduced-precision policy were:

- 300/300 finite evaluations;
- speedup at least 1.20 relative to FP32;
- lower bound of the paired 95% bootstrap interval greater than 1.0;
- batch-size-one slowdown no greater than 1.05;
- a discrepancy from FP32 above the configured numerical-noise floor.

## Results

| Policy | Finite | Best speedup | 95% bootstrap interval | Batch-1 ratio | Pass |
|---|---:|---:|---:|---:|---:|
| TF32 | 300/300 | 0.9955 | 0.9886–1.0030 | 1.0088 | No |
| BF16 AMP | 294/300 | 0.8195 | 0.8056–0.8345 | 1.2182 | No |

TF32 had a maximum force discrepancy of 19,075.54 eV/angstrom, a mean of
116.09 eV/angstrom, and a 95th percentile of 3.56 eV/angstrom. Its mean energy
difference from FP32 was -1.94 eV, with a standard deviation of 23.91 eV.

BF16 AMP had a maximum force discrepancy of 455.83 eV/angstrom, a mean of
5.59 eV/angstrom, and a 95th percentile of 12.15 eV/angstrom. Its mean energy
difference from FP32 was 80.18 eV, with a standard deviation of 55.55 eV. Six
frames produced nonfinite results.

A separate BF16 preflight frame had a 90.50 eV absolute-energy difference but
only a 0.0172 eV/angstrom maximum force difference. This motivates separating
composition-dependent energy offsets from configuration-dependent energy
differences.

The graph-batching invariance preflight passed its stated criteria. Across its
nine frames, the maximum FP32 single-versus-batch energy difference was zero
and maximum relative force difference was approximately `9.27e-7`.

## Pilot interpretation

Neither blanket TF32 nor blanket BF16 AMP was a viable fast policy for this
model, workload, software stack, and A40 GPU. TF32 was statistically
indistinguishable from FP32 in speed, while BF16 was slower and not universally
finite. Gate 1 therefore produced `STOP`, and Gate 2 was not run.

## Limits on interpretation

P1 does not establish that reduced precision is unsuitable for all MLIPs. It
does not isolate the operation causing each nonfinite value or large error. The
close-contact stratum is deliberately adversarial and must be reported
separately from ordinary configurations. FP32 was an operational reference, not
a physical ground truth. Only one model, GPU architecture, software stack, and
independent benchmark process were tested.

## Stored and external artifacts

`manifest.json` and `summary.json` are compact normalized records derived from
the archived machine-readable outputs. `checksums.txt` identifies the full
local result bundle and key original files. The full bundle is intentionally
excluded from ordinary Git history; it should be attached to a versioned
release or deposited in a research-data repository before publication.
