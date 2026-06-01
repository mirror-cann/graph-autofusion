# Explanation of Automatic Fusion Module Precision Consistency

## Conclusion

**The automatic fusion module cannot guarantee binary-level complete consistency with handwritten operators under Eager mode output**. However, through conservative precision strategies, it can guarantee **precision no worse than Eager mode**. This document explains its principles and applicable boundaries. It also provides processing paths when you wish to continue using automatic fusion.

---

## 1. Root Cause of Precision Differences

### 1.1 Different Code Generation Paths

- **Eager Mode**: Each operator is **pre-written by humans** and compiled into fixed binary kernels.
- **Automatic Fusion**: Uses codegen technology to generate operator code at runtime (JIT). Even **without any fusion**, for operators with the same semantics, automatically fused generated code and handwritten kernels are not the same source code.

**Different source code is the starting point of all numerical differences.**

### 1.2 How Source Code Differences Evolve into Numerical Differences

Two different source codes for the same operator introduce observable least significant bit differences in at least the following four aspects:

1. **Different Precision Promotion Strategies**
   To maintain numerical stability under fp16, operators often promote intermediate calculations to fp32 internally. Whether to promote, at which step to promote, and when to demote back—these choices may differ between handwritten implementations and codegen implementations.
2. **Different Instruction Selection**
   The same semantics can be implemented with different instructions. For example, `Tensor * Scalar`:
   - First broadcast scalar, then use standard multiplication instruction;
   - Directly use `muls` type scalar multiplication instruction.

   Both are mathematically equivalent. However, they use different multiplication units on hardware. Rounding behaviors are not completely consistent.
3. **Different Tiling Strategies and Reduction Accumulation Order**
   Floating-point addition **does not satisfy associativity**: `(a+b)+c` and `a+(b+c)` results usually differ in the least significant bit. Reduction order for `reduce`, `matmul`, `layernorm`, and other calculations is determined by tiling strategy and parallelism. The tiling strategy itself **cannot be consistent from hardware resource constraints** between automatic fusion and handwritten operators.

   To elaborate, under the same tiling size, a fusion kernel internally undertakes intermediate calculations for multiple operators. Single-step UB occupation will be higher than a single-operator kernel. Handwritten single operators, to reduce transfer overhead, often make tiling as large as possible (use full ub)—this tiling size may not fit in fusion scenarios.
4. **Different Algorithm Implementations**
   The same mathematical function can be approximated with different algorithms. For example, `rsqrt` and `div` often use multi-round iteration approximation implementations. Different iteration rounds and initial value selections lead to different least significant bit errors.

### 1.3 Summary

Different source code → produces differences in any dimension of precision promotion, instructions, accumulation order, or algorithms → final output differs at binary bit level. This is a **principle-level** limitation, not a bug that can be eliminated through engineering alignment.

---

## 2. Scenarios Where Binary Consistency Can Be Guaranteed

Precision differences originate from **computation**. For **pure data movement operators** (no mathematical operations, only data rearrangement):

- `broadcast`
- `concat` / `split`
- `transpose` / `reshape` / `slice`

Such operators can theoretically achieve binary consistency.

However, **fusion benefits for such scenarios are usually small**. Automatic fusion benefits mainly come from:

- Multiple computation operator fusion (eliminates intermediate result read/write);
- Multiple computation + single data movement fusion (computation and memory access overlap).

Pure data movement fusion accounts for a low proportion in real business. Therefore, the intersection of "scenarios that can guarantee consistency" and "scenarios worth fusing" is very narrow.

---

## 3. Guarantees Provided by Automatic Fusion

Although binary consistency with handwritten operators is not guaranteed, automatic fusion provides the following guarantees in two directions: precision and reproducibility:

### 3.1 Forced fp32 Accumulation Inside Fusion Regions

Automatic fusion uniformly promotes all intermediate calculations to fp32 inside generated fusion kernels by default. It retains original dtype only at fusion block input/output boundaries. It trades partial computational performance for higher intermediate precision.

### 3.2 Precision No Worse Than Eager Mode

Under Eager mode, each operator executes independently. Intermediate results between operators must be written to disk at tensor original dtype (such as fp16/bf16), then read by the next operator. This means:

- **Eager Mode**: fp16/bf16 truncation occurs at **each operator boundary**;
- **Automatic Fusion**: Merges a chain of operators into one kernel. **Truncates only at fusion block boundaries**, fully fp32 inside the block.

Therefore, automatic fusion accumulated rounding error upper bound ≤ Eager mode corresponding operator chain accumulated error. That is, precision is not lower than Eager mode.

### 3.3 Self-Determinism (Reproducibility)

> This section discusses **automatic fusion's own** output determinism, **not comparison with Eager mode**. For relationship with Eager, see §1, §2, §3.2.

Automatic fusion determinism can be viewed at two levels:

- **Compilation Determinism**: Same compilation-time inputs → same kernel binary. Compilation-time inputs include model structure, shape, dtype, automatic fusion version, and compilation options.
- **Execution Determinism**: Same kernel binary + same runtime inputs → same output. Runtime inputs include tensor values, chip model, and so on.

Layering both gives end-to-end guarantee: **When compilation-time inputs and runtime inputs are both unchanged, output from multiple runs is completely consistent at binary level**.

That is, the difference sources listed in §1.2 (precision promotion strategy, instruction selection, tiling/accumulation order, algorithm implementation) are **completely consistent between automatic fusion's own two compilations + runs**—they only produce divergence between "automatic fusion vs handwritten operators". Automatic fusion itself does not introduce additional uncertainty.

---

## 4. PyTorch Inductor Related Explanation

PyTorch Inductor community does not guarantee binary precision consistency. Its logic is consistent with automatic fusion. Excerpted original text follows:

### 4.1 PyTorch Documentation "Numerical Accuracy"

Involves floating-point associativity and bit-exact, original text:

> "PyTorch is not guaranteed to produce bitwise identical results for floating point computations that are mathematically identical."
>
> "floating point addition and multiplication are not associative, so the order of the operations affects the results."

Source: [https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

### 4.2 Edward Yang "Ways to use torch.compile" (2024)

The author is a PyTorch core developer. The article explains the numerical relationship between `torch.compile` and eager in the "Improve training efficiency on a small-medium scale" section:

> "Unfortunately, the compiler does not guarantee exact bitwise equivalence with eager code; we reserve the right to do things like select different matrix multiply algorithms with different numerics or eliminate unnecessary downcast/upcasts when fusing half precision compute together."

Source: [https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/)

### 4.3 PyTorch Documentation "torch.compile Troubleshooting"

Explanation about downstream compiler numerical performance in the "Accuracy Debugging" section:

> "the reason we need this is downstream compilers will codegen code whether it's Triton code or the C++ backend, the numerics from those downstream compilers can be different in subtle ways yet have dramatic impact on your training stability."

Source: [https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html)

---

## 5. Comparison Table

| Dimension | Eager Handwritten Operators | Automatic Fusion |
|------|---------------|----------|
| Source Code | Human-fixed | JIT generated |
| Binary Consistency | Baseline | **Not guaranteed** (computation type) / Can be consistent (pure data movement type) |
| Intermediate Calculation Precision | Limited by inter-operator dtype (often fp16) | fp32 inside fusion block |
| Accumulated Error Upper Bound | Baseline | **≤ Eager baseline** |
| Performance | Baseline | Significantly better than baseline |

---

## 6. Usage Recommendations: Processing Paths When Wishing to Continue Using Automatic Fusion

If users focus on **numerical correctness** (model precision, convergence, downstream business metrics), automatic fusion has no precision disadvantage and has significant performance benefits. You can enable it directly without additional processing.

If users have **strict alignment with Eager** requirements and wish to continue using automatic fusion performance benefits, this is very challenging. Refer to the following strategy combinations.

### 6.1 PyTorch-Inductor Scenario: Status and Usage Recommendations

Strict alignment with Eager mode involves graph processing before Inductor fusion and the fusion stage. Inductor does not provide native close switches for most related processes. You need to implement through patches or source code modifications.

#### Pre-Fusion Processing

Before automatic fusion, Inductor executes the following processes that may introduce numerical differences:

- **Inplace Operator Functionalization**: For example, `relu_` → `relu`. Functionalized kernel implementation may differ from Eager.
- **Decompose**: Decomposes complex operators into basic operators to expand fusion range. Algorithms differ compared to single-operator implementation.
- **FX Graph Pass**: Mainly pattern replacement fusion passes, involves kernel replacement.
- **Forward-Backward Graph Splitting**: Graph splitting strategy based on automatic recomputation. Recomputed fusion kernel may differ from Eager mode.

Recommended measures:

1. **Inplace Operator Functionalization** is a fusion prerequisite and non-functional item. Cannot be turned off. In practice, recommend implementing only Inplace versions. Reduce numerical differences from kernel differences through automatic functionalization.
2. **Decompose**: Inductor natively does not provide a close option. Need to implement through patches or source code modifications.
3. **FX Graph Pass**: Inductor does not provide an option to close all passes. Effective passes may differ in each version. Need to close one by one based on actual execution situation.
4. **Forward-Backward Graph Splitting**: Inductor uses max-flow min-cut algorithm for graph splitting. Introduces recomputation. You can replace with no-recomputation graph splitting strategy through configuring `custom_partitioner_fn`.

#### Automatic Fusion

Inductor fusion stage introduces numerical differences due to regenerating execution kernels. Involved fusion types are as follows:

| Fusion Type | Affects Binary Precision | Numerical Difference Source | User Configurable |
|----------|----------------|--------------|------------|
| Reduction Type | Yes | Different accumulation order | Requires patch or source code modification |
| Pointwise Type | Yes | Type promotion rules, instruction mapping, iteration algorithm differences | Requires patch or source code modification |
| MM/FA Template Type | Yes | Algorithm implementation differences | Requires patch or source code modification |

Usage notes:

- If pursuing strict alignment with Eager, need to close all above fusion types. Inductor does not provide corresponding configuration items. Need to implement through patches or source code modifications.
- Recommend explicitly enabling `TORCHINDUCTOR_EMULATE_PRECISION_CASTS`. Avoid numerical errors introduced by cast elimination optimization during Lowering process.

### 6.2 TensorFlow Scenario: Status and Usage Recommendations

#### General Optimization

TensorFlow scenarios default to using GE as the graph compiler backend. GE's built-in fusion passes and hardware-independent optimization passes (for example, constant folding, equivalent formula transformation) all cause kernel source code changes. Binary precision consistency cannot be guaranteed.

Recommended measures:

1. Set graph compilation optimization level to O0 (only retain functional optimizations).
2. At O0 level, functional-related and static shape-related optimizations still execute. You can open DUMP graph switch to observe actually effective passes in GE. Compare with section 1.2 to judge impact on precision. If impact exists, manually adjust scripts to avoid corresponding optimizations (higher difficulty).

#### Automatic Fusion

Automatic fusion currently supports fusion of the following operator types. Default strategy and controllability are as follows:

| Fusion Type | Default Status | Affects Binary Precision | User Configurable |
|----------|----------|----------------|------------|
| elemwise (including broadcast) | Enabled | Yes | No |
| reduce    | Disabled | Yes | Yes |
| concat    | Disabled | No | Yes |
| slice     | Disabled | No | Yes |
| gather    | Disabled | No | Yes |
| transpose | Disabled | No | Yes |

Fusion types disabled by default can be explicitly enabled through environment variables. Using concat as example:

```shell
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=concat"
```

Usage notes:

- **elemwise fusion currently cannot be disabled**. This fusion type is both the main source of binary precision differences and the main source of fusion benefits. Overall benefits will significantly reduce after disabling. If you have a disabling requirement, submit an issue for feedback.
- **Fusion types disabled by default are experimental features**. After enabling, functional anomalies or performance degradation may occur. Recommend testing on target networks before deciding whether to retain.