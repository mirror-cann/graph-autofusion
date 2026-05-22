# Kernel Tool Usage Guide

## 1. Tool Purpose

This tool primarily uses CPU simulation to locate UB non-aligned access issues in kernels.

## 2. Tool File List and Description

- 1 autofuse_tiling_data.h
  > Copy content from kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_tiling_data.h

- 2 autofuse_tiling_func_common.h
  > Copy content from kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_tiling_func_common.h, and modify the following macro definitions to empty:
  ```cpp
  #define OP_LOGD(name, fmt, ...) 
  #define OP_LOGI(name, fmt, ...) 
  #define OP_LOGW(name, fmt, ...) 
  #define OP_LOGE(name, fmt, ...) 
  #define OP_EVENT(name, fmt, ...)
  ```

- 3 kernel.cpp
  > Copy content from kernel_meta_xxxxxxx/te_ascbackend_xxxx/device/autofuse_xxx_op_kernel.cpp, and add the following two lines of code at the file header:
  ```cpp
  #define REGISTER_TILING_DEFAULT(tiling)
  #define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
  ```

- 4 main.cpp
  > Main function for constructing input, gold values, tiling data, and verifying kernel calculation results against gold values. Refer to its implementation for details.

- 5 Makefile
  > Build project script

- 6 README.md
  > Usage instructions

- 7 tiling_func_asc_graph0_schedule_result0_g0.cpp
  > Copy content from kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_xxx_tiling_func_asc_graph0_schedule_result0_g0.cpp

- 8 tiling_func_schedule_group_tail.cpp
  > Copy content from kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_xxx_tiling_func_schedule_group_tail.cpp

- 9 tiling_func_solver_func.cpp
  > Copy content from kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_xxx_tiling_func_solver_func.cpp

- 10 tiling_func_tiling_def_and_tiling_const.cpp
  > Copy content from kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_xxx_tiling_func_tiling_def_and_tiling_const.cpp

## 3. Refer to main.cpp Implementation for Test Cases

main.cpp is the test case entry point. You need to refer to its implementation to construct corresponding implementation. Currently only supports dynamic shape scenarios. If static shape, can modify to dynamic shape.

## 4. Compilation

```sh
make CANN_INSTALL_PATH=/usr/local/Ascend/latest # Replace with your own CANN package path
```

## 5. Execute Tool

```sh
source /usr/local/Ascend/latest/bin/setenv.bash # Set CANN package environment variable
./test_kernel
```