# Automatic Fusion Precision Test Report

> Related Document: [Automatic Fusion Precision Consistency Explanation](./autofuse_precision_consistency.md)
>
> This report provides empirical data support for the four core arguments of the main document. Test results flow back to main document §3 and §6.1.

---

## 1. Overview

This report splits into 8 independent tasks (including one shared infrastructure) for parallel advancement. Each task is self-contained with purpose, configuration, steps, expected results, and deliverables. You can independently assign them to different responsible persons.

## 2. Task Assignment Table

| ID | Test Name | Corresponding Argument | Owner | Workload | Priority | Prerequisite Dependency | Status |
|------|---------|---------|--------|--------|--------|---------|------|
| S   | Precision Test Toolkit (bit-diff / ULP / fp64 reference) | Shared Infrastructure | TODO | 1d | P0 (Blocking Item) | None | Not Started |
| A   | Automatic Fusion Self Binary Determinism | Main Document §3.3 | TODO | 0.5d | P0 | None | Not Started |
| C1  | Tensor × Scalar Instruction Selection Difference | Main Document §1.2 (2) | TODO | 0.5d | P0 | None | Not Started |
| C2  | Reduction Accumulation Order Difference | Main Document §1.2 (3) | TODO | 0.5d | P0 | Tiling Control Interface | Not Started |
| C3  | rsqrt Algorithm Iteration Difference | Main Document §1.2 (4) | TODO | 0.5d | P0 | Iteration Count Control Interface | Not Started |
| D   | Pure Data Movement Operator Consistency | Main Document §2 | TODO | 0.5d | P1 | S Complete | Not Started |
| B   | Eager vs AutoFuse Accumulated Error | Main Document §3.2 | TODO | 2–3d | P1 | S Complete | Not Started |
| E   | End-to-End Training Metric Alignment | Main Document §3.2 Real Projection | TODO | 5–7d | P2 | S Complete, NPU Resource | Not Started |

## 3. Parallel Advancement Recommendations

- **Day 0 Start Separately**: S (Infrastructure, 1 day, blocks other tasks depending on it).
- **Day 0 Start in Parallel**: A, C1, C2, C3 do not depend on S. Can start simultaneously with S.
- **Day 1 Start in Parallel**: B, D, E start after S completes.
- **Minimum Human Configuration**: 2 people ≈ 1 week; recommend 4–5 people ≈ 3–4 days.
- **Shortest Path**: After all P0 complete, the main document can already supplement key data. P1/P2 results strengthen §6.1 and end-to-end persuasiveness.

---

## 4. Task S — Precision Test Toolkit

### Purpose
Provide shared numerical comparison tools for A/B/D/E. Avoid reinventing the wheel. The toolkit itself does not produce arguments but is a prerequisite dependency for other tasks.

### Deliverable List
1. `bit_equal(a, b) -> bool`: Element-wise bit-wise comparison. Returns whether completely consistent.
2. `ulp_diff(a, b) -> ndarray`: Element-wise ULP difference.
3. `error_stats(a, ref) -> dict`: Statistics of max / mean / P50 / P99 absolute error, relative error, ULP error.
4. `fp64_reference(graph, inputs) -> tensor`: Recalculates subgraph using numpy / torch CPU fp64 as ground truth baseline.
5. Packaging form: Internal pip package + pytest fixture entry.

### Steps
1. Form a group to align toolkit interfaces and repository location.
2. First produce `bit_equal` and `ulp_diff` to meet A/D immediate use; complete other functions within two days.
3. Unit test coverage: fp16 / bf16 / fp32 three levels; normal values, subnormal numbers, inf / nan full coverage.
4. Publish to internal pypi or integrate in source code form.

### Deliverables
Toolkit repository link, interface documentation, unit test report.

---

## 5. Task A — Automatic Fusion Self Binary Determinism

### Purpose
Verify main document §3.3: When compilation-time inputs and runtime inputs are unchanged, output from multiple runs is binary consistent, and compilation artifacts themselves are hash stable.

### Test Configuration
- Subgraph: `matmul → add → layernorm → gelu` (Transformer FFN first half).
- Input shape: `[8, 2048, 4096]` (fp16).
- Hardware: Single card, fixed model.

### Steps
1. Build subgraph, enable automatic fusion, collect fusion kernel binary hash (record as `H_k0`).
2. Use fixed seed to generate a set of input tensors, save to disk.
3. Execute 1000 times consecutively in the same process, save each output MD5.
4. Clear JIT cache, cold start new process, recompile, execute 1000 times again.
5. Summary:
   - First round 1000 output MD5 set size should be 1.
   - Second round kernel hash should equal `H_k0`, output MD5 should equal first round.

### Expected Results
- Run output hash: All consistent.
- Compilation artifact hash: Unchanged after cold start.
- If inconsistency occurs: Identify source (random dropout, non-deterministic reduce, NUMA drift, and so on). Record and report to framework team.

### Deliverables
Table: `Run Count / Unique Hash Count / Max ULP Diff`, expect `1000 / 1 / 0`.

---

## 6. Task C1 — Tensor × Scalar Instruction Selection Difference

### Purpose
Verify main document §1.2 (2): Same mathematical semantics using different hardware instructions produces least significant bit differences.

### Test Configuration
- Input A: fp16 tensor shape `[1024, 1024]`, fixed seed generates `U(-1, 1)`.
- Scalar: `0.7`.
- Path A: Use scalar multiplication instruction (`muls` or equivalent Python notation `x * 0.7`, let framework automatically lower to muls).
- Path B: First broadcast scalar to `[1024, 1024]` tensor, then use vector multiplication instruction (`x * scalar_tensor.expand_as(x)`).

### Steps
1. Write two minimal reproduction scripts.
2. Use profiler / op trace to verify the two paths actually generate different instructions.
3. Execute each path once with the same input, save output.
4. Use tool S to calculate both ULP difference distributions.
5. Report: ULP = 0 / ULP = 1 / ULP ≥ 2 percentages.

### Expected Results
The two path outputs are **not** completely consistent. Most elements have ULP ≤ 1, very few higher. The data itself is evidence of "instruction selection introducing least significant bit differences".

### Feasibility Risk
Framework may automatically select instructions. Need profiler to confirm two paths actually generate different instructions. Otherwise test significance is lost. If uncontrollable, change to "observe instruction changes in one path" and disclose honestly.

### Deliverables
Script + profiler screenshot + ULP difference histogram.

---

## 7. Task C2 — Reduction Accumulation Order Difference

### Purpose
Verify main document §1.2 (3): Floating-point addition does not satisfy associativity. Different tiling sizes introduce least significant bit differences.

### Test Configuration
- Input: fp16 tensor shape `[1, 16384]`, fixed seed.
- Operation: `sum(dim=-1)`.
- Path A: Tiling 512.
- Path B: Tiling 1024.
- Path C: Tiling 2048.
- Reference: fp64 CPU precision calculation.

### Steps
1. Generate three tiling versions through framework tiling configuration (compilation flag or kernel-level hint).
2. Run each version 1000 times with same input, verify self determinism (tool S's `bit_equal`).
3. Pairwise compare ULP differences between three versions.
4. Compare each version with fp64 reference, see which is closer.

### Expected Results
- Each version's own 1000 runs are completely consistent (self determinism).
- ULP ≥ 1 differences exist between different tiling versions.
- All three fall within random error range of fp64 baseline.

### Feasibility Risk
Framework does not expose tiling control externally. Need framework team cooperation to provide a debug flag. Without this interface, this test needs to be shelved or changed to "observe relationship between framework self-selected tiling strategy and output under different shapes".

### Deliverables
Three version output ULP difference matrix, comparison table with fp64.

---

## 8. Task C3 — rsqrt Algorithm Iteration Difference

### Purpose
Verify main document §1.2 (4): Implementation differences in iteration approximation operators directly reflect as least significant bit precision differences.

### Test Configuration
- Input: fp16 tensor shape `[1024, 1024]`, value range `U(0.01, 100)`, fixed seed.
- Operation: `rsqrt`.
- Path A: Framework default implementation.
- Path B: Adjust iteration count in same framework (for example, 2 rounds vs 3 rounds Newton iteration).
- Reference: fp64 precision calculation.

### Steps
1. Contact framework team to get iteration count configuration interface (environment variable or compilation switch).
2. Run both paths with same input once, save output.
3. Use tool S to calculate ULP difference distributions for both A / B relative to fp64 reference.
4. If smooth, can add another example: `div(x, y)`, same mechanism.

### Expected Results
- A and B outputs are not completely consistent.
- Path with more iterations has smaller ULP difference.

### Feasibility Risk
Same as C2: If iteration count cannot be controlled, change to "framework implementation A vs same semantics numpy fp32 implementation" comparison. Persuasiveness slightly lower but still demonstrates algorithm-level differences.

### Deliverables
Two path ULP difference histograms + table of iteration count vs least significant bit difference relationship.

---

## 9. Task D — Pure Data Movement Operator Consistency

### Purpose
Verify main document §2: Data movement operators without mathematical operations can achieve binary consistency between automatic fusion and Eager.

### Test Configuration
- Operator list: `transpose` / `reshape` / `slice` / `concat` / `split` / `broadcast`.
- Combined subgraphs (need to trigger fusion):
  - `transpose → concat`
  - `slice → broadcast → concat`
  - `split → transpose → concat`
- Input: fp16 / bf16 / fp32 one set each, shape covers typical business values.

### Steps
1. Run Eager and AutoFuse separately for each subgraph.
2. Verify using tool S's `bit_equal`.
3. Enumerate dtype × shape combinations, fill in check table.
4. When counterexamples occur (for example, broadcast with implicit type promotion), record and explain reason.

### Expected Results
All pure data movement subgraphs are bit-wise consistent. Broadcast with implicit computation may have differences. Mark separately.

### Deliverables
Check table + counterexample list.

---

## 10. Task B — Eager vs AutoFuse Accumulated Error Comparison

### Purpose
Support main document §3.2's most core quantitative conclusion: Automatic fusion accumulated error ≤ Eager mode.

### Test Configuration
- Operator chains (covering different computation patterns):
  1. Pure elementwise: `mul → add → gelu → mul`
  2. With reduction: `layernorm`, `softmax`
  3. With matmul: `matmul → add → layernorm` (FFN first half)
  4. Optional: `attention`'s `qk → softmax → pv` fragment
- Input: 1000 sets of random inputs for each chain, fixed seed pool.
- Three-way execution:
  - fp64 CPU reference (ground truth)
  - fp16 Eager (operator-by-operator execution, boundary write to disk)
  - fp16 AutoFuse

### Steps
1. Use tool S's `fp64_reference` to generate ground truth.
2. Run Eager and AutoFuse separately, collect outputs.
3. For both paths, calculate max absolute error, relative error, ULP error P50 / P99 / Max.
4. Draw histograms and CDF of 1000 sets of errors.
5. Compare operator chain by operator chain: Whether AutoFuse's P99 / Max ≤ Eager's P99 / Max.
6. When counterexamples occur:
   - Identify specific operator and input distribution.
   - Flow results back to main document §6.1 as empirical evidence for "need to disable fusion by operator type".

### Expected Results
- Most operator chains have AutoFuse P99 / Max error ≤ Eager.
- Individual counterexamples need to flow separately to main document §6.1.

### Deliverables
- One error distribution comparison chart per operator chain + one summary table row.
- Counterexample list (directly hand to §6.1 author).

---

## 11. Task E — End-to-End Training Metric Alignment

### Purpose
Support main document §6 opening "focus on numerical correctness can enable directly" recommendation. Use downstream model experiments to address training stability concerns.

### Test Configuration
Model candidates (in ascending cost order):
1. GPT-2 small (124M) — minimum cost, 1k steps approximately several hours.
2. LLaMA-7B LoRA fine-tuning — medium cost.
3. Small-scale pretrain (such as 300M model 5k steps) — most persuasive but most expensive.

Starting recommendation: First run candidate 1. If conclusion is clear, skip subsequent. For each model fix: seed, dataset, batch, lr schedule, optimizer.

### Steps
1. Build reproducible training script (Eager and AutoFuse two switches).
2. Run same script with both switches once, sample every 50 steps.
3. Compare metrics:
   - loss curve (train / eval).
   - grad norm.
   - If LM: PPL.
4. Two curves should coincide within normal random fluctuation. If significantly deviate:
   - Record deviation point, deviation amount.
   - Flow back to main document §6.1.

### Expected Results
Candidate 1 completion yields preliminary conclusion. Two curves should basically coincide.

### Feasibility Risk
Need NPU / GPU resources and dataset coordination. If resources are tight, at least complete candidate 1 first.

### Deliverables
loss / eval metric comparison curve charts + final metric difference table.

---

## 12. Post-Completion Flowback Actions

After test execution completes:
1. Embed the two key conclusion tables from A / B back into main document §3 as links.
2. If B / E produce counterexamples, drive main document §6.1 (@wangxiaotian) to implement specific switch lists.
3. Tool S is retained long-term as subsequent precision regression baseline infrastructure.