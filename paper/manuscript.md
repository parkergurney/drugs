# Beyond FP32 Benchmarking: Performance and Numerical Reliability of Reduced-Precision Pretrained Interatomic Potentials

## Abstract

Pretrained machine-learning interatomic potentials (MLIPs) are increasingly
evaluated by their predictive accuracy, simulation speed, memory consumption,
and short-run stability. Numerical precision is rarely treated as an explicit
benchmark dimension, even though reduced-precision tensor operations can alter
both computational throughput and the energy gradients used as molecular
forces. We introduce a reproducible framework for measuring the performance,
numerical reliability, and simulation-level consequences of TF32 and BF16
inference relative to FP32. An initial feasibility experiment with
MACE-OFF23(S) on an NVIDIA A40 found no useful blanket reduced-precision policy:
TF32 achieved a speed ratio of 0.996 (95% bootstrap interval 0.989–1.003), while
BF16 autocast achieved 0.820 (0.806–0.834) and returned finite results for only
294 of 300 stress-stratified configurations. These pilot results do not imply a
universal limitation of reduced precision. They motivate an operator-level,
cross-workload, and cross-hardware study of the conditions required for MLIP
inference to be both faster and scientifically reliable.

## 1. Introduction

Machine-learning interatomic potentials approximate quantum-chemical energy
surfaces at a fraction of the cost of direct electronic-structure calculations.
Rapid growth in pretrained molecular models has shifted part of the practical
challenge from constructing a potential to choosing one whose accuracy,
throughput, memory requirements, and stability match a target application.

Eastman, Pretti, and Markland addressed this selection problem through a uniform
comparison of fifteen pretrained MLIPs. Their study evaluated predictive
accuracy, simulation speed, GPU memory use, and the ability to produce stable
short simulations. It also showed that architecture can influence speed and
memory as strongly as parameter count, making direct measurement more useful
than simple model-size proxies.

That benchmark deliberately performed all calculations in FP32, stating that
lower-precision modes such as TF32 were insufficiently accurate for many
applications. This is a defensible operational choice, but it leaves several
questions unresolved. The magnitude and structure of reduced-precision errors
were not quantified; speed gains were not measured; precision-sensitive model
operations were not localized; and portability across molecular workload and
GPU architecture was not assessed.

Meanwhile, precision-aware equivariant architectures have reported substantial
benefits from mixed-precision execution when geometrical preprocessing,
normalization, or reductions remain in FP32. Those results need not transfer to
blanket autocast of an existing pretrained potential. Equivariant tensor
products, energy accumulation, differentiation of energy into forces, workload
size, and accelerator-specific kernels can each affect the joint accuracy and
performance outcome.

This work therefore treats numerical precision as a first-class benchmark
variable. Its objective is not to show that reduced precision is always useful,
but to determine the conditions under which it is useful and to identify the
mechanisms by which it fails. We contribute: (1) a reproducible paired protocol
for measuring reduced-precision energy-and-force inference; (2) a
configuration-stratified taxonomy of numerical failures; (3) operator-level
precision-placement ablations; and, conditionally, (4) cross-model,
cross-hardware, and molecular-dynamics validation of promising policies.

## 2. Related work

### 2.1 Benchmarking pretrained MLIPs

Existing MLIP benchmarks differ in datasets, error definitions, hardware, and
timing scope, which complicates comparisons between published models. Eastman
et al. reduced this ambiguity by evaluating many molecular MLIPs under a common
protocol. They used the SPICE test set for conformational energy accuracy and
ASE simulations on an NVIDIA H100 for speed and memory measurements. Their
study is the methodological foundation for the present work, which extends the
benchmark space with execution precision and numerical-failure analysis.

### 2.2 Reduced precision in equivariant potentials

Reduced precision can improve tensor-core utilization and reduce memory
traffic, but benefits depend on whether a workload contains sufficiently large
eligible operations. Numerical risk is also nonuniform: geometric bases,
normalization, reductions, and energy gradients can amplify rounding differently
from dense matrix multiplication. DPA4 provides an important contrasting case,
reporting mixed-precision benefits from a design that retains selected
operations in FP32. The present study asks whether comparable benefits can be
obtained safely in pretrained models not originally co-designed for blanket
reduced-precision inference.

### 2.3 Numerical reliability in molecular dynamics

Single-point energy and force error does not by itself determine simulation
quality. Constant composition-dependent energy offsets may be irrelevant to
forces, while small systematic force errors can alter ensemble observables.
Pointwise trajectories also diverge chaotically even when they sample equivalent
distributions. A scientifically useful precision assessment must therefore
separate raw energy offsets, relative energies, forces, finite execution, and
simulation-level observables.

## 3. Research questions

- **RQ1:** Do TF32 or BF16 policies accelerate end-to-end MLIP energy-and-force
  inference relative to FP32?
- **RQ2:** How do their numerical errors and nonfinite failures depend on model
  operation and molecular configuration?
- **RQ3:** Can selective FP32 protection recover reliability while preserving a
  practical speedup over full FP32?
- **RQ4:** How do conclusions transfer across workload size, model architecture,
  optimized kernels, and GPU architecture?
- **RQ5:** When single-point criteria pass, are selected molecular-dynamics
  observables equivalent within preregistered margins?

## 4. Methods

### 4.1 Study structure

The study separates an observed pilot from prospectively specified
confirmatory experiments. Pilot P1 is retained unchanged and is not used as an
independent confirmatory replicate. Subsequent experiments proceed through
reproduction, failure localization, precision-placement ablation,
generalization, and conditional MD validation. The full decision rules are in
the confirmatory protocol accompanying the artifact.

### 4.2 Initial model and configurations

The initial model is MACE-OFF23(S). The P1 accuracy set contains 300 rMD17
configurations from ethanol, malonaldehyde, and aspirin: 100 ordinary frames,
100 high-force frames selected from a candidate pool, and 100 constructed
close-contact stress tests. Results are reported by stratum so that deliberately
adversarial configurations do not obscure the intended operating domain.

### 4.3 Precision policies

The baseline policies are FP32, TF32 matrix multiplication, and CUDA BF16
autocast. Diagnostic experiments add FP64 where supported. Operator-level
ablations will retain candidate sensitive regions in FP32 while allowing
eligible operations to use reduced precision. Compilation and optimized kernels
are separate experimental factors rather than implicit configuration changes.

### 4.4 Numerical outcomes

Every evaluation records finite status, raw total energy, forces, and error
against FP32 and, where supported, FP64. Energy analysis includes raw,
per-atom, composition-centered, and pairwise conformational differences. Force
analysis includes component error, atomic vector error, RMSE, and robust
distribution summaries. Selected configurations undergo finite-difference
energy–force consistency checks and intermediate-operation instrumentation.

### 4.5 Performance outcomes

Policies are compared on identical prepared graph batches. CUDA is synchronized
around the timed region. Cold-start and steady-state costs are distinguished,
and graph construction, forward evaluation, force differentiation, conversion,
and transfer costs are measured separately. Confirmatory comparisons use at
least five independent processes with counterbalanced policy order and paired
hierarchical bootstrap intervals.

### 4.6 Reproducibility

Each run records its experiment ID, protocol version, command, Git status and
commit, configuration and input hashes, model hash, seed, environment, GPU
telemetry, warnings, output hash, and protocol deviations. Numerical failures
and out-of-memory events are outcomes rather than exclusions.

### 4.7 Conditional MD validation

Only policies passing finite-output, numerical, and performance requirements
advance to replicated MD. Comparisons will use energy drift and selected
ensemble observables with margins frozen before confirmatory trajectories are
analyzed. Pointwise trajectory agreement is not required.

## 5. Pilot feasibility result

P1 was completed on 19 August 2026 using an NVIDIA A40, PyTorch 2.4.1,
MACE 0.3.16, CUDA 12.1, and genuine disconnected graph batching. It used 20
warm-up and 100 measured iterations at batch sizes 1, 8, and 32.

Neither reduced-precision policy passed Gate 1. TF32 returned finite values for
all 300 frames but achieved a best speed ratio of 0.9955, with a 95% paired
bootstrap interval of 0.9886–1.0030. BF16 autocast returned finite values for
294 frames and achieved a speed ratio of 0.8195, with an interval of
0.8056–0.8345. Thus, TF32 was indistinguishable from FP32 in speed and BF16 was
slower.

Both policies also produced configuration-dependent discrepancies. TF32 had a
maximum force difference of 19,075.54 eV/angstrom, while BF16 had a maximum of
455.83 eV/angstrom. BF16's mean raw energy difference was 80.18 eV. In a
separate preflight frame, however, a 90.50 eV energy shift coincided with a
maximum force difference of only 0.0172 eV/angstrom, suggesting that raw total
energy error may mix composition-dependent offsets with changes to the local
energy surface. These observations motivate, but do not substitute for, the
planned failure-localization experiments.

## 6. Planned diagnostic and generalization study

The next stage will reproduce P1 in independent processes, decompose timing
costs, compare with FP64 diagnostics, localize the first nonfinite operations,
and separate atomic/reference-energy offsets from interaction energies. Guided
by those measurements, controlled ablations will protect individual operation
classes in FP32. Passing policies will then be evaluated across additional
models, workload scales, and a second GPU architecture.

## 7. Threats to validity

### Internal validity

GPU timing is sensitive to asynchronous execution, initialization, thermal and
clock state, policy order, and concurrent workloads. The confirmatory protocol
therefore synchronizes timed regions, counterbalances order, records telemetry,
and uses independent processes. Instrumentation can itself change execution and
will be disabled for primary timing measurements.

### Construct validity

FP32 is an operational baseline rather than physical truth. FP64 diagnostics
and energy–force consistency tests reduce, but do not eliminate, this concern.
Raw absolute-energy error can exaggerate dynamical consequences when it is
dominated by a constant composition-dependent offset, so raw and relative
energies are both reported. Conversely, acceptable average errors can hide rare
catastrophic configurations, so nonfinite rates and tails are retained.

### External validity

P1 used one small pretrained model, three small molecular species, one A40, and
one software stack. Close-contact frames are useful stress tests but may not
represent configurations reached in ordinary MD. Claims will therefore be
limited to tested domains until replicated across model and GPU architectures.

### Simulation validity

Short single-point benchmarks cannot establish equilibrium-observable
equivalence, while chaotic divergence makes pointwise trajectory comparison
inappropriate. Only policies passing earlier gates undergo replicated MD with
prospectively chosen observables and equivalence margins.

## 8. Current conclusion

The pilot establishes a narrow result: blanket TF32 and BF16 autocast did not
provide safe acceleration for MACE-OFF23(S) under the tested A40 workload. The
broader thesis remains an open empirical question: reliable reduced-precision
MLIP inference may require operator-aware precision placement and may depend on
architecture, workload, software kernels, and hardware.
