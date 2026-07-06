# SK 适配规则手册

这是 `adapt-sk-from-global` 编码规则的快速参考，也是本 skill 对生成 SK binding 形态的本地行为契约。

## 生成内容

对每一种支持的源码形态，adapter 要么生成当前 SK binding，要么返回明确的人工处理项。对于干净的非 SK `__global__` kernel，会在原始函数之后按顺序生成：

1. **Args struct**：命名为 `<NameCamel>Args`，每个 kernel 参数对应一个字段，保持原始顺序。C 类型小于 4 字节的字段，例如 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`bool`，使用 `alignas(4)`。如果原始 kernel 是模板函数，只有字段类型依赖模板参数时 Args struct 才模板化；只影响 body 或 kernel type 的模板参数保留在 SK 函数上，但不改变 runtime 参数包布局。
2. **模板化 `__sk__` 函数**：

   ```cpp
   template<uint32_t splitidx>
   __sk__ <kernel_type> void <name>_sk(const <NameCamel>Args *args
                                       [, sk::SkSystemArgs *sysArgs]) {
       <c_type> <param> = args-><param>;   // one line per parameter
       // ... original body verbatim ...
   }
   ```

   原始 body 默认不改动。只有 body 引用了 `AscendC::GetBlockNum()` 时，才注入 `sysArgs` 参数，并把调用改写为 `sysArgs->skNumBlocks`。
3. **`SK_BIND` 语句**：

   ```cpp
   SK_BIND(<orig>, <mask>, <name>_sk<0>, <name>_sk<1>, <name>_sk<2>, <name>_sk<3>)
   ```

   `mask` 默认是 4（DCCI）。允许值是 0..7，其中 0 表示没有能力 bit，1/2/4 是 bit flag。`--num-splits` 控制绑定多少个 `<name>_sk<N>` 符号，范围 1..4。

原始 `__global__` 函数必须保持不变。

## 多算子聚合渲染

单算子适配仍然为每个 asset 写一个 aclgraph-canonical tree：

```text
operator-sk-adapted/
  csrc/<op>.asc
  csrc/pybind11.asc
  op_extension/__init__.py
  op_extension/_torch_library.py
  setup.py
```

`aggregate-sk-adapted` 消费多个这样的输出，并渲染一个聚合 tree，包含所有 `csrc/<op>.asc`、一个 `pybind11.asc`、一个 `_torch_library.py` 和一个 `setup.py`。聚合内 entry 名必须唯一。生成的 pybind 层为每个算子暴露面向用户的 bind target entry：

| 函数 | 作用 |
|---|---|
| `run_<op>` | 通过 `torch.library` 注册的 SK-facing bind target；differential validation 在 baseline 和 SK context 下复用同一个入口 |

聚合 `setup.py` 保持 Python import 包名为 `op_extension`，同时用用户指定的 distribution name 和 version 生成 wheel 文件名。

## 输入形态

| 输入形态 | 适配行为 |
|---|---|
| `none` | 生成 Args struct、模板化 `__sk__` 和 `SK_BIND`。 |
| 可修复 `none` | 在临时副本上执行 codegen 拥有的预适配自动修复，再生成当前 SK binding。 |
| `current-sk-bind` | 按字节复制源码，并标记为 `already_current`。 |
| `partial` / `unknown` | 不猜测，输出 `codegen.unknown-sk-form` 等人工处理项。 |

## Kernel 类型映射

| 原始 qualifier | SK qualifier |
|---|---|
| `__vector__` | `__vector__` |
| `__cube__` | `__cube__` |
| `__mix__(c, v)` general | `__mix__(c, v)` |
| `__mix__(1, 0)` | `__cube__`（特殊情况） |
| `__mix__(0, 1)` | `__vector__`（特殊情况） |
| bare `__aicore__` | `__aicore__` |

## 何时注入 `sysArgs`

`--with-sys-args=auto` 是默认值：只有原始 body 包含 `AscendC::GetBlockNum()` 时才注入。`--with-sys-args=always` / `=never` 可以强制选择。

注入后使用当前 API 名：`sysArgs->skNumBlocks` / `sysArgs->SkGetNumBlocks()`。历史 `skBlockNum` / `SkGetBlockNum` 在当前 CANN 头文件下会编译失败；`sk-operator-validate --rule-pack spec` 会将其标记为 `sk.sys-args-api-current`，并支持自动重命名修复。

完整规格和边界情况以脚本实现、测试和本手册共同约束。
