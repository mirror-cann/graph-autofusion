# 算子迁移指南

## 函数部分

函数主体：

1. cube/vector和函数部分分别使用 `__DAV_CUBE__`和 `__DAV_VEC__`宏进行隔离

函数名部分：

1. 原有 `__global__`标识替换为 `__spk__`标识
2. 函数名更新为其他具备判别性的函数名

函数参数部分：

1. 统一修正为使用 `__gm__ uint64_t *param`指针进行传递
2. 通过 `(GM_ADDR)param[index]`方式获取原有参数
3. 结构体参数需要通过 `GET_STRUCT_PTR`宏进行获取，注意获取到的内容为结构体指针

功能主体部分：

1. 原有逻辑保持不变
2. 注意对应的结构体部分转换为了tiling指针，无需再次取地址
3. 函数结尾不需要再添加 `pipe_barrier(PIPE_ALL);`，由框架统一处理

额外注意：

1. 【Planing】目前纯V算子为保证直调使用，暂时使用内部方案FunLevelKType进行标识，后续会有更完善的方案【后续一定会删除】
2. mix1:1和mix1:2算子需要区分mix_aic和mix_aiv两种类型，分别对应AIC和AIV的场景
3. 【Planing】目前SK算子暂时没有使用模板写法，主要目前暂不能自动获取对应SK入口的函数名，后续会完善该部分
4. 【Doing】目前8.3版本的cann包条件下不支持在sk框架中使用AscendC::printf, 后续的版本已定位并解决了这个问题

## 编译部分

1. 使用ASC编译方式进行编译
2. 需要增加ops的 `common`路径，内置了迁移所需的辅助宏

## SK框架部分

1. 【Planing】目前在迁移完成后，需要在 `sk_task.cpp`的 `g_sk_fun_map`中手动增加函数名映射，以便框架能够正确获取函数地址
2. `SkBuildTask`构建算子任务时，当前框架自动偏移了ffts，但后续的cann版本底层接口已完成偏移，无需框架完成，暂时为了兼容仍保留的该部分代码，后续会删除该部分逻辑

## 算子实例

### 算子实现&编译

*同时写2个入口函数，一个global应用于单算子，一个aicore用于sk调用*

```c++
// 默认打开了SuperKernel选项，但算子模式退出前自己加一个PIPE_ALL
template<typename DTYPE_X, typename DTYPE_GAMMA>
__global__ __aicore__ void rms_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, RMSNormTilingData tiling)
{
  {
    KernelRmsNormMergeN<DTYPE_X, DTYPE_GAMMA, RMSNormTilingData> op; 
    op.Init(x, gamma, y, rstd, &tiling);
    op.Process();
  }
  pipe_barrier(PIPE_ALL);
}

// aicore函数需要通过宏隔离一下，否则编译有问题(当前有bug，会导致编译1个aic，1个aiv)
#ifdef __DAV_VEC__
extern "C" __aicore__ void rms_norm_half_half_sk(__gm__ uint64_t *param)
{
  GM_ADDR x = (GM_ADDR)param[0];      // 参数需要额外处理一下，从单一的buffer中获取
  GM_ADDR gamma = (GM_ADDR)param[1];
  GM_ADDR y = (GM_ADDR)param[2];
  GM_ADDR rstd= (GM_ADDR)param[3];
  KernelRmsNormMergeN<half, half, __gm__ RMSNormTilingData> op; // tiling结构体需要带__gm__描述，不属于参数表了
  __gm__ RMSNormTilingData *tilingdata = (__gm__ RMSNormTilingData *)(__gm__ uint8_t *)(param + 4); 
  op.Init(x, gamma, y, rstd, tilingdata);
  op.Process();
}
#endif
```