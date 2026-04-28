# autofuse 用例演示 

## 用例功能：

autofuse 融合 eq + where 两个算子。

## 执行命令

```
python3 af_eq_where.py
```

## 预期执行结果

当前目录下的 profiling 目录下，有生成的 profiling 文件。其中 PROF_000001_时间辍xx/mindstudio_profiler_output 下面，可以打开 op_summary_时间辍xx.csv 文件，看到执行的算子详情。此时仅有算子名为 autofused_eq_where_拓扑哈希 的 kernel，表示已经将 eq+where 两个算子融合为了一个融合算子。