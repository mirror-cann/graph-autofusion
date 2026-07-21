# elementwise + elementwise 融合样例（abs + relu + exp）

## 用例功能

autofuse 融合 `abs + relu + exp` 三个 elementwise 算子。

## 执行命令

```bash
python3 test_abs_relu_exp.py
```

## 预期执行结果

脚本构造 `abs → relu → exp` 计算图，在 NPU 上执行 100 步推理，无报错即表示融合成功。三个算子被融合为一个 `AscBackend` 类型的融合算子 `autofuse_pointwise_0_Abs_Relu_Exp`，在 NPU 上以单个 kernel 执行。

当前目录下的 profiling 目录下，有生成的 profiling 文件。其中 `PROF_*/mindstudio_profiler_output` 下面，可以打开 `op_summary_*.csv` 文件，看到执行的算子详情。此时仅有算子名为 `autofuse_pointwise_0_Abs_Relu_Exp` 的 kernel，表示已经将三个算子融合为一个融合算子。
