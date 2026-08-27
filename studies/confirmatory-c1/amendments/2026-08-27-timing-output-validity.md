# C1 protocol amendment: timing versus output validity

- Date: 2026-08-27 UTC
- Experiment: C1-A40-P1-REPRODUCTION
- Status: post-observation analysis amendment
- GPU trials affected: none; all five processes were already complete

## Observation before amendment

The first combined-analysis attempt stopped with `unpaired finite timings for
bf16_amp, batch 32`. Inspection then established that all 500 BF16 batch-32
measurements had positive finite wall-clock durations but nonfinite model
outputs. The original analyzer filtered timing rows using model-output
finiteness, incorrectly conflating two separate outcomes.

## Amendment

Performance analysis pairs completed measurements by process, batch size, and
iteration. A timing is usable when its wall-clock duration is finite and
positive, regardless of whether the evaluated energy and forces were finite.
Structural pairing failures or invalid wall-clock values remain analysis
errors.

Model-output finiteness is reported separately for every policy, batch size,
and process. A policy with any nonfinite required output fails the finite-
execution criterion even when a timing ratio can be estimated. No timing,
evaluation, process, or numerical failure is discarded by this amendment.

## Rationale

The performance estimand is the cost of the attempted model execution. A model
call that completes in measurable time but returns a nonfinite scientific
result still has a valid execution cost and an invalid numerical outcome.
Keeping those axes separate preserves both findings and prevents numerical
failure from selectively removing performance samples.
