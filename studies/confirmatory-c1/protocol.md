# Confirmatory protocol: reduced-precision MLIP inference

## Version and status

- Protocol version: 0.3
- Status: C1 complete; D1 instrumentation correction frozen after setup failure
- Pilot used to design protocol: P1
- Outcomes inspected: P1 and the five-process C1 reproduction on P1
- Sealed C1 reduced-precision outputs inspected: none
- Confirmatory selection seed: `2026082601`
- Confirmatory frame exclusion: all source frames represented in the frozen P1
  selection, including sources used to construct close-contact frames

Any change after freezing must be recorded in a dated amendment before the
affected result is analyzed.

### 2026-08-30 D1 instrumentation correction

The first D1 instrumented process failed uniformly before model evaluation
because ordinary PyTorch forward hooks are unsupported on MACE TorchScript
submodules. All 27 cases produced empty operator traces and explicit
`register_forward_pre_hook is not supported on ScriptModules` errors for all
four diagnostic policies. No operator-localization result existed to inspect.

The three uninstrumented timing processes completed before this analysis
failure and remain frozen. They will not be rerun. The correction skips hooks
on ScriptModules, records their names, observes their nearest eager-module
boundaries, retains the prospectively specified targeted TorchDispatch probe,
and makes empty or malformed traces an immediate diagnostic error. Only the
invalid `d1-trace-01` process and derived analysis will be rerun under tag
`d1-p1-tracer-fix-v2`. The original failed process is preserved in the
checksummed pre-fix raw archive.

The confirmatory dataset may be constructed using geometry and FP32 force
scores only. Its TF32 and BF16 outputs must not be generated or inspected until
the primary energy and force margins below are frozen in a dated amendment.
Dataset construction records the P1 exclusion-file checksum and must report
zero source-frame overlap.

## Primary research question

Under what combinations of numerical policy, MLIP operation, molecular
configuration, workload scale, and GPU architecture does reduced-precision
energy-and-force inference provide a reliable speedup over FP32?

## Confirmatory hypotheses

- **H1 — performance:** At least one tested reduced-precision policy achieves
  an end-to-end paired speedup of at least 1.20 over FP32, with the lower bound
  of its 95% confidence interval greater than 1.0.
- **H2 — finite execution:** A usable policy returns finite energies and forces
  for every confirmatory frame in its declared operating domain.
- **H3 — numerical reliability:** A usable policy satisfies preregistered
  relative-energy and force-error margins in that domain.
- **H4 — precision placement:** Selectively retaining sensitive operations in
  FP32 improves the joint speed–reliability outcome over blanket BF16 autocast.
- **H5 — portability:** The direction and magnitude of the result may change
  with model architecture, workload scale, optimized kernels, or GPU
  architecture.

H5 is an estimation question; failure to transfer will be reported rather than
treated as an excluded result.

## Experimental stages and gates

### C1: independent A40 reproduction

Run at least five independent benchmark processes from a clean locked
environment, using the P1 model, frames, policies, batch sizes, and timing
scope. Counterbalance policy order between processes. Capture GPU telemetry and
confirm no competing workload.

Gate C1 passes when the sign and practical interpretation of both P1 speed
results reproduce and numerical-failure counts are characterized. Exact point
estimates need not match P1.

### D1: failure localization

D1 is exploratory mechanism work on the frozen P1 dataset and combined
five-process C1 reproduction. It does not evaluate the separately prepared C1
confirmatory dataset. The diagnostic population includes every
reduced-policy nonfinite frame and, for each reduced policy and stratum, the
nearest finite frame to the minimum, median, 95th percentile, and maximum
process-aggregated maximum force error. Ties are resolved by `frame_id`, and
duplicate selections are not replaced.

For every selected ordinary, high-force, close-contact, and nonfinite frame:

1. compare FP64 diagnostics, FP32, TF32, and BF16;
2. record the first operation producing a nonfinite or extreme discrepancy;
3. decompose total energy into atomic/reference and interaction terms;
4. compare raw, per-atom, and composition-centered relative energies;
5. perform finite-difference energy–force consistency checks;
6. separate forward, force-gradient, conversion, transfer, and graph costs.

Module-boundary relative RMS is defined as
`RMS(policy - FP32) / max(RMS(FP32), 1e-12)`. Exploratory localization flags
are 0.01 for TF32 and 0.08 for BF16 AMP. These flags identify operations for
investigation; they are not scientific accuracy margins or policy-acceptance
criteria. Finite-difference steps are `1e-3`, `3e-4`, and `1e-4` angstrom.

D1 timing uses the original P1 representative batch construction at batch
sizes 1, 8, and 32, with 20 warmups and 100 measured iterations in each of
three fresh A40 processes. Instrumented traces are never used as timing
measurements. FP64 is diagnostic only. Unsupported operations, nonfinite
values, exceptions, and batching-dependent behavior are retained as outcomes.

Gate D1 passes when the dominant performance and numerical failure mechanisms
are supported by direct measurements rather than inferred from final outputs.
Artifact validation establishes evidence completeness but does not
automatically make a causal scientific gate decision.

### A1: operator-level ablations

Change one factor at a time. Candidate FP32-protected regions include geometry,
radial bases, normalization, tensor products, reductions, energy accumulation,
and force differentiation. Also test compilation and officially supported
optimized kernels as separate factors.

Only policies that are completely finite on ordinary and high-force frames and
meet the numerical margins advance.

### G1: generalization

Test two or three representative MLIP architectures, more than one workload
scale, and at least one second GPU architecture. Use matched systems, timing
definitions, and statistics. Do not describe a condition as general unless it
replicates beyond the original model and A40.

### M1: molecular-dynamics validation

Only promising policies advance to replicated MD. Compare NVE energy drift,
temperature, structural distributions, and selected torsional or state
populations using equivalence margins fixed before the trajectories are
analyzed. Pointwise trajectory identity is not required.

### Optional controller stage

An adaptive audit/replay controller will be attempted only if a candidate fast
policy is at least 1.20 times faster before controller overhead, is usually
safe, and has failures that available signals can detect. Otherwise the thesis
ends with the benchmark and mechanism study.

## Experimental controls

- Use identical frames and batch composition for paired policy comparisons.
- Synchronize CUDA immediately outside every timed region.
- Use warm-up iterations specified in advance.
- Counterbalance policy order across independent processes.
- Record cold-start and steady-state measurements separately.
- Record GPU clocks, temperature, utilization, power, and concurrent processes.
- Keep accuracy evaluation separate from timing when instrumentation changes
  execution.
- Preserve all finite and nonfinite observations.
- Report ordinary, high-force, and close-contact strata separately and jointly.

## Confirmatory sample plan

- C1: at least five independent processes per policy and batch configuration.
- D1: one resumable instrumented diagnostic process and three independent,
  uninstrumented A40 timing processes.
- Within process: retain the P1 20 warm-ups and 100 paired measured iterations
  unless a timing-autocorrelation pilot justifies a frozen amendment.
- Accuracy: reuse the frozen 300 frames for direct reproduction, then evaluate
  a separately selected confirmatory set using a new recorded seed.
- MD: determine trajectory count through a pilot variance estimate, then freeze
  the count and equivalence margins before confirmatory trajectories.

## Exclusions and missing data

A run may be excluded only for a documented external failure such as GPU
preemption, a concurrent unrelated workload, corrupted input, or environment
mismatch. Numerical exceptions, nonfinite model outputs, out-of-memory events,
and precision-policy failures are outcomes, not exclusions. Report all
exclusions with experiment IDs and reasons.

## Analysis policy

- Primary performance result: ratio of paired mean steady-state wall times,
  FP32 divided by candidate time.
- Confidence interval: paired hierarchical bootstrap, resampling independent
  processes first and iterations within processes second.
- Report raw distributions and process-level estimates.
- Analyze strata separately before pooled summaries.
- Do not remove numerical outliers merely because they are extreme.
- Distinguish exploratory localization from confirmatory hypothesis tests.
- Apply the metric definitions below.

## Metric definitions

For paired observations indexed by `i`, define steady-state speedup as
`mean(t_fp32,i) / mean(t_policy,i)`. The primary timed scope is prepared
graph-batch model evaluation, force calculation, and required output transfer.
Graph construction and end-to-end MD-step time are reported separately. A
practically useful policy requires speedup of at least 1.20 and a 95% interval
lower bound greater than 1.0.

An evaluation is finite only if its total energy and every force component are
finite. Exceptions, unsupported operations, out-of-memory events, and
nonfinite outputs are recorded separately and are outcomes rather than
exclusions.

Energy analysis reports raw total-energy difference, absolute difference per
atom, within-composition centered difference, and pairwise conformational
energy-difference error. For same-composition frames `i` and `j`, the latter is
`(E_policy,i - E_policy,j) - (E_reference,i - E_reference,j)`. Both raw and
centered results are retained. The primary energy margin remains `TBD` and must
be frozen before confirmatory accuracy data are examined.

Force analysis reports maximum absolute component error, maximum atomic vector
error, component RMSE, full-vector relative RMSE, and the median, 95th
percentile, and maximum by stratum. Componentwise relative errors near zero are
not used. The primary force margin remains `TBD` pending an independently
justified scientific threshold.

Selected frames undergo finite-difference energy–force consistency checks and
single-versus-genuine-batch invariance checks. Policies advancing to MD report
NVE energy drift, nonfinite and termination rates, temperature distributions,
selected structural or state-population distributions, and effective
throughput. MD equivalence margins and trajectory counts will be frozen after
an independent variance pilot and before confirmatory trajectory analysis.

## Success and stopping rules

A policy is useful only if it simultaneously passes performance, finite-output,
and numerical-reliability criteria in a declared domain. A fast but inaccurate
policy and an accurate but slower policy both fail the joint criterion.

Stop development of the adaptive controller if no policy passes after the
predeclared A1 ablations. A negative result remains a thesis result: it defines
the tested boundary conditions under which reduced precision is not beneficial.

## Experiment registry

| ID | Status | Purpose | Confirmatory? |
|---|---|---|---|
| P1 | Complete | Initial MACE-OFF23(S)/A40 feasibility | No |
| C1 | Complete | Independent A40 reproduction | Yes |
| D1 | Ready | Numerical and timing localization | Exploratory |
| A1 | Planned | FP32-protected operation ablations | Mixed |
| G1 | Planned | Cross-model and cross-hardware generalization | Yes |
| M1 | Conditional | MD observable equivalence | Yes |
| R1 | Conditional | Audit/replay controller | Yes |

For each run, record the experiment ID, UTC time, protocol version, exploratory
or confirmatory status, Git commit and worktree status, exact command,
configuration/input/model/output hashes, seed, complete software and hardware
environment, GPU telemetry, policy order, timing scope, synchronization method,
warnings, deviations, factual results, contemporary interpretation, and any
inclusion decision. Protocol amendments must state their date, affected
experiments, whether outcomes had already been inspected, and justification.
