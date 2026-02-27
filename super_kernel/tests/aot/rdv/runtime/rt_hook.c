/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>
#include <pthread.h>
#include <string.h>
// #include <assert.h>
#include <sys/types.h>
#include "securec.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "rt_intf.h"

static __thread void *g_rt_dso_handle;

static void *get_rt_func(const char *name)
{
  void *fun = dlsym(RTLD_NEXT, name);
  if (fun) {
    return fun;
  }
  if (!g_rt_dso_handle) {
    g_rt_dso_handle = dlopen("libruntime.so", RTLD_GLOBAL|RTLD_NOW);
  }
  return dlsym(g_rt_dso_handle, name);
}

static __thread size_t g_rt_cap_flag;
static __thread size_t g_rt_kernel_cnt;
static __thread struct rt_info *g_rt_kernel_info;
static __thread size_t g_rt_bin_hdl_cnt;
static __thread void **g_rt_bin_hdls;
static __thread size_t g_rt_elf_hdl_cnt;
static __thread void **g_rt_elf_hdls;
static __thread int g_rt_dry_run;
#define MAX_HDL_CNT 4096

static void add_key_val(void ***box, size_t *count, void *key, void *val)
{
  if (!(*box)) {
    *box = (void **)malloc(sizeof(void*) * MAX_HDL_CNT * 2);
    *count = 0;
  }
  // if (*count >= MAX_HDL_CNT) {
  //   assert(0);
  // }
  (*box)[*count * 2] = key;
  (*box)[*count * 2 + 1] = val;
  (*count)++;
}
static void *get_key_val(void **box, size_t count, const char *key)
{
  for (int i = count - 1; i >= 0; i--) {
    if (box[i * 2] == key) {
      return box[i * 2 + 1];
    }
  }
  return NULL;
}
static void add_bin_hdl(void *bin_hdl, const char *fun)
{
  if (!g_rt_cap_flag) {
    return;
  }
  add_key_val(&g_rt_bin_hdls, &g_rt_bin_hdl_cnt, (void*)fun, bin_hdl);
}
static void *get_bin_hdl(const char *fun)
{
  return get_key_val(g_rt_bin_hdls, g_rt_bin_hdl_cnt, (void*)fun);
}
static const char *get_origin_func(const char *fun)
{
  void **box = g_rt_bin_hdls;
  size_t count = g_rt_bin_hdl_cnt;
  for (int i = count - 1; i >= 0; i--) {
    if (!strcmp(box[i * 2], fun)) {
      return (const char *)box[i * 2];
    }
  }
  return NULL;
}
static void add_elf_hdl(void *elf, void *hdl)
{
  if (!g_rt_cap_flag) {
    return;
  }
  add_key_val(&g_rt_elf_hdls, &g_rt_elf_hdl_cnt, hdl, elf);
}
static void *get_elf_hdl(void *hdl)
{
  return get_key_val(g_rt_elf_hdls, g_rt_elf_hdl_cnt, hdl);
}
void get_meta_info(const char *elf, const char *fun, uint32_t *kernel_type, uint32_t ratio[2]);
static void add_kernel(const char *func_name, size_t arg_size, void *arg_data, uint32_t numBlocks, uint64_t stream_id, uint8_t legacy)
{
  if (!g_rt_cap_flag) {
    return;
  }
  if (!g_rt_kernel_info) {
    g_rt_kernel_info = (struct rt_info *)malloc(sizeof(struct rt_info) * MAX_HDL_CNT);
  }
  // if (g_rt_kernel_cnt >= MAX_HDL_CNT) {
  //   assert(0);
  // }
  struct rt_info *data = &g_rt_kernel_info[g_rt_kernel_cnt++];
  data->arg_size = arg_size;
  const size_t MAX_ARG_SIZE = 64 * 1024 * 1024;  // 64MB
  if (arg_size == 0 || arg_size > MAX_ARG_SIZE) {
    printf("[error] invalid arg_size: %zu\n", arg_size);
    return;
  }
  data->arg_data = malloc(arg_size);
  if (data->arg_data == NULL) {
    printf("[error] malloc arg_data failed\n");
    return;
  }
  memcpy_s(data->arg_data, arg_size, arg_data, arg_size);
  data->func_name = (void *)func_name;
  data->bin_hdl = get_bin_hdl(func_name);
  data->stream_id = stream_id;
  data->numBlocks = numBlocks;
  data->legacy = legacy;
  data->task_ratio[0] = 0;
  data->task_ratio[1] = 0;
  data->task_type = RT_TASK_KERNEL;
  data->event_type = 0;
  data->event_id = 0;
  data->event_addr = NULL;
  data->value = 0;
  data->value_size = 0;
  data->wait_flag = 0;
  data->value_addr = NULL;
  const char *elf = (const char *)get_elf_hdl(data->bin_hdl);
  if (elf) {
    get_meta_info(elf, func_name, &data->kernel_type, data->task_ratio);
  }
}
static void clear_data(void)
{
  if (g_rt_kernel_info) {
    for (int i = 0; i < g_rt_kernel_cnt; i++) {
      struct rt_info *data = &g_rt_kernel_info[i];
      free(data->arg_data);
    }
    free(g_rt_kernel_info);
    g_rt_kernel_info = NULL;
    g_rt_kernel_cnt = 0;
  }
  if (g_rt_bin_hdls) {
    free(g_rt_bin_hdls);
    g_rt_bin_hdls = NULL;
    g_rt_bin_hdl_cnt = 0;
  }
  if (g_rt_elf_hdls) {
    free(g_rt_elf_hdls);
    g_rt_elf_hdls = NULL;
    g_rt_elf_hdl_cnt = 0;
  }
}
void rt_start_capture(int dry_run)
{
  clear_data();
  g_rt_cap_flag = 1;
  g_rt_dry_run = dry_run;
}
void rt_stop_capture(void)
{
  g_rt_cap_flag = 0;
  g_rt_dry_run = 0;
}
void rt_get_kernel_info(struct rt_info **info, size_t *size)
{
  *info = g_rt_kernel_info;
  *size = g_rt_kernel_cnt;
}

//////////////////////////////////// asc launch kernel ////////////////////////////////////////
typedef struct tagRtDevBinary {
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    const void *data;  // binary data
    uint64_t length;   // binary length
} rtDevBinary_t;
typedef struct rtHostInputInfo {
    uint32_t addrOffset;
    uint32_t dataOffset;
} rtHostInputInfo_t;
typedef struct tagRtArgsEx {
    void *args;                     // args host mem addr
    rtHostInputInfo_t *hostInputInfoPtr;     // nullptr means no host mem input
    uint32_t argsSize;              // input + output + tiling addr size + tiling data size + host mem
    uint32_t tilingAddrOffset;      // tiling addr offset
    uint32_t tilingDataOffset;      // tiling data offset
    uint16_t hostInputInfoNum;      // hostInputInfo num
    uint8_t hasTiling;              // if has tiling: 0 means no tiling
    uint8_t isNoNeedH2DCopy;        // is no need host to device copy: 0 means need H2D copy,
                                    // others means doesn't need H2D copy.
    uint8_t reserved[4];
} rtArgsEx_t;
__attribute__((visibility("default")))
int rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl)
{
  typedef int (*binreg_fun)(const rtDevBinary_t *bin, void **hdl);
  static __thread binreg_fun fun;
  if (!fun) {
    fun = (binreg_fun)get_rt_func("rtDevBinaryRegister");
  }
  int ret = fun(bin, hdl);
  if (!ret) {
    add_elf_hdl((void *)bin->data, *hdl);
    printf("==============rtDevBinaryRegister=============\n");
  }
  return ret;
}
__attribute__((visibility("default")))
int rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t numBlocks, rtArgsEx_t *argsInfo,
  void *smDesc, void *stm, uint32_t flags, const void *cfgInfo)
{
  typedef int (*launch_fun)(const void *stubFunc, uint32_t numBlocks, rtArgsEx_t *argsInfo,
    void *smDesc, void *stm, uint32_t flags, const void *cfgInfo);
  static __thread launch_fun fun;
  if (!fun) {
    fun = (launch_fun)get_rt_func("rtKernelLaunchWithFlagV2");
  }
  add_kernel((const char *)stubFunc, argsInfo->argsSize, argsInfo->args, numBlocks, (uint64_t)stm, 1);
  printf("==============rtKernelLaunchWithFlagV2=============\n");
  if (g_rt_dry_run) {
    printf("===============dry run mode ===============\n");
    return 0;
  }
  return fun(stubFunc, numBlocks, argsInfo, smDesc, stm, flags, cfgInfo);
}
__attribute__((visibility("default")))
int rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName,
  const void *kernelInfoExt, uint32_t funcMode)
{
  typedef int (*funreg_fun)(void *binHandle, const void *stubFunc, const char *stubName,
    const void *kernelInfoExt, uint32_t funcMode);
  static __thread funreg_fun fun;
  if (!fun) {
    fun  = (funreg_fun)get_rt_func("rtFunctionRegister");
  }
  add_bin_hdl(binHandle, (const char *)stubFunc);
  printf("==============rtFunctionRegister=============\n");
  return fun(binHandle, stubFunc, stubName, kernelInfoExt, funcMode);
}
__attribute__((visibility("default")))
int rtKernelLaunch(const void *stubFunc, uint32_t numBlocks, void *args, uint32_t argsSize,
  void *smDesc, void *stm)
{
  typedef int (*launch_fun)(const void *stubFunc, uint32_t numBlocks, void *args, uint32_t argsSize,
    void *smDesc, void *stm);
  static __thread launch_fun fun;
  if (!fun) {
    fun = (launch_fun)get_rt_func("rtKernelLaunch");
  }
  add_kernel((const char *)stubFunc, argsSize, args, numBlocks, (uint64_t)stm, 1);
  printf("==============rtKernelLaunch=============\n");
  if (g_rt_dry_run) {
    printf("===============dry run mode ===============\n");
    return 0;
  }
  return fun(stubFunc, numBlocks, args, argsSize, smDesc, stm);
}
static int __rtsBinaryLoadFromData(const void *data, const uint64_t length, void *cfg, void **binHandle)
{
  static __thread int (*load_data)(const void *data, const uint64_t length, void *cfg, void **binHandle);
  if (!load_data) {
    load_data = (typeof(load_data))get_rt_func("rtsBinaryLoadFromData");
  }
  int ret = load_data(data, length, cfg, binHandle);
  add_elf_hdl((void *)data, *binHandle);
  printf("==============rtsBinaryLoadFromData=============\n");
  return ret;
}
__attribute__((visibility("default")))
int rtsBinaryLoadFromData(const void *data, const uint64_t length, void *cfg, void **binHandle)
{
  return __rtsBinaryLoadFromData(data, length, cfg, binHandle);
}
__attribute__((visibility("default")))
int rtsBinaryLoadFromFile(const void *path, void *cfg, void **binHandle)
{
  static __thread int (*load_file)(const void *data, void *cfg, void **binHandl);
  if (!load_file) {
    load_file = (typeof(load_file))get_rt_func("rtsBinaryLoadFromFile");
  }
  int fd = open(path, O_RDWR);
  size_t length = lseek(fd, 0, SEEK_END);
  if (length < 0) {
    close(fd);
    printf("################################################### rtsBinaryLoadFromFile called error1!\n");
    return load_file(path, cfg, binHandle);
  }
  lseek(fd, 0, SEEK_SET);
  void *data = malloc(length);
  size_t size = read(fd, data, length);
  if (size != length) {
    close(fd);
    printf("################################################### rtsBinaryLoadFromFile called error12\n");
    return load_file(path, cfg, binHandle);
  }
  close(fd);
  return __rtsBinaryLoadFromData(data, length, cfg, binHandle);
}
__attribute__((visibility("default")))
int rtsFuncGetByName(const void *bin_hdl, const char *func_name, void **func_hdl)
{
  typedef int (*funget_fun)(const void *bin_hdl, const char *func_name, void **func_hdl);
  static __thread funget_fun fun;
  if (!fun) {
    fun  = (funget_fun)get_rt_func("rtsFuncGetByName");
  }
  add_bin_hdl((void *)bin_hdl, (const char *)func_name);
  printf("==============rtsFuncGetByName============\n");
  return fun(bin_hdl, func_name, func_hdl);
}
int __rtsFuncGetName(const void *func_hdl, uint32_t len, char *name)
{
  typedef int (*getname_fun)(const void *func_hdl, uint32_t len, char *name);
  static __thread getname_fun fun;
  if (!fun) {
    fun  = (getname_fun)get_rt_func("rtsFuncGetName");
  }
  return fun(func_hdl, len, name);
}
__attribute__((visibility("default")))
int rtsLaunchKernelWithHostArgs(const void *func_hdl, uint32_t numBlocks, void *stm, void *cfg,
  void *host_args, uint32_t argsSize, void *place_holder, uint32_t cnt)
{
  typedef int (*launch_fun)(const void *func_hdl, uint32_t numBlocks, void *stm, void *cfg,
    void *host_args, uint32_t argsSize, void *place_holder, uint32_t cnt);
  static __thread launch_fun fun;
  if (!fun) {
    fun  = (launch_fun)get_rt_func("rtsLaunchKernelWithHostArgs");
  }
  printf("==============rtsLaunchKernelWithHostArgs============\n");
  char func_name[1024];
  if (__rtsFuncGetName(func_hdl, 1024, func_name)) {
    printf("################################################### rtsFuncGetName called error!\n");
    abort();
  }
  const char *stubFunc = get_origin_func(func_name);
  add_kernel(stubFunc, argsSize, host_args, numBlocks, (uint64_t)stm, 0);
  if (g_rt_dry_run) {
    printf("===============dry run mode ===============\n");
    return 0;
  }
  return fun(func_hdl, numBlocks, stm, cfg, host_args, argsSize, place_holder, cnt);
}
