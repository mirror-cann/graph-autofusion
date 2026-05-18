/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "autofuse_tiling_data.h"

#ifndef HOST_CODE_FILE
#define HOST_CODE_FILE ""
#endif
#ifndef DEVICE_CODE_FILE
#define DEVICE_CODE_FILE ""
#endif
#ifndef OUTPUT_DIR
#define OUTPUT_DIR ""
#endif

struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};

using GenerateTopnSolutionsFn = int64_t (*)(int64_t, int64_t, int64_t,
                                            const std::vector<std::map<std::string, std::string>> &,
                                            int64_t, std::vector<AutofuseTilingData> &,
                                            std::vector<int64_t> &, std::vector<int64_t> &, ResLimit *);
using GetTilingDataReprFn = std::string (*)(const AutofuseTilingData *);
using GetModeledPerfForTestingFn = double (*)(const AutofuseTilingData *);
using AutofuseTilingFn = int64_t (*)(uint32_t, uint32_t, uint32_t, AutofuseTilingData *, uint32_t *, uint32_t *,
                                     ResLimit *);

namespace {

std::string ReadFile(const std::string &path) {
  std::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

bool WriteFile(const std::string &path, const std::string &content) {
  std::ofstream out(path);
  if (!out.is_open()) return false;
  out << content;
  return true;
}

int RunCommand(const std::string &cmd) {
  int status = std::system(cmd.c_str());
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return -1;
}

bool FileExists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

#ifndef PYAUTOFUSE_DIR
#define PYAUTOFUSE_DIR ""
#endif
#ifndef AUTOFUSE_PYTHON_DIR
#define AUTOFUSE_PYTHON_DIR ""
#endif
#ifndef ASCEND_HOME_PATH
#define ASCEND_HOME_PATH ""
#endif

std::string PythonPreamble() {
  return
    "import sys, os, traceback\n"
    "pkg_dir = os.path.join('" + std::string(OUTPUT_DIR) + "', 'autofuse_pkg')\n"
    "os.makedirs(pkg_dir, exist_ok=True)\n"
    "autofuse_dir = os.path.join(pkg_dir, 'autofuse')\n"
    "try:\n"
    "    os.symlink('" + std::string(AUTOFUSE_PYTHON_DIR) + "', autofuse_dir)\n"
    "except FileExistsError:\n"
    "    pass\n"
    "pyautofuse_dst = os.path.join(autofuse_dir, 'pyautofuse.so')\n"
    "try:\n"
    "    os.symlink('" + std::string(PYAUTOFUSE_DIR) + "/pyautofuse.so', pyautofuse_dst)\n"
    "except FileExistsError:\n"
    "    pass\n"
    "sys.path.insert(0, pkg_dir)\n"
    "import autofuse.ascendc_compile as _ac\n"
    "_ac.ASCEND_PATH = '" + std::string(ASCEND_HOME_PATH) + "'\n";
}

int RunHostCompile(const std::string &tiling_def, const std::string &host_code, const std::string &output_file) {
  WriteFile(OUTPUT_DIR "/host_tiling_def.h", tiling_def);
  WriteFile(OUTPUT_DIR "/host_impl.cpp", host_code);

  std::string script_path = std::string(OUTPUT_DIR) + "/run_host_compile.py";
  WriteFile(script_path,
    PythonPreamble() +
    "try:\n"
    "    from autofuse.compile_adapter import host_compile\n"
    "    import os\n"
    "    os.makedirs('" + std::string(OUTPUT_DIR) + "/host_out', exist_ok=True)\n"
    "    td = open('" + std::string(OUTPUT_DIR) + "/host_tiling_def.h').read()\n"
    "    hc = open('" + std::string(OUTPUT_DIR) + "/host_impl.cpp').read()\n"
    "    host_compile(td, hc, [\n"
    "        '--graph_name=pgo_add_abs_inductor',\n"
    "        '--output_file=" + output_file + "',\n"
    "        '--output_path=" + std::string(OUTPUT_DIR) + "/host_out',\n"
    "        '--soc_version=Ascend910B'])\n"
    "except Exception:\n"
    "    traceback.print_exc()\n"
    "    sys.exit(1)\n");

  std::string cmd = "ASCEND_HOME_PATH=" + std::string(ASCEND_HOME_PATH) + " python3 " + script_path + " 2>&1";
  int ret = RunCommand(cmd);
  if (ret != 0) printf("host_compile failed, ret=%d\n", ret);
  return ret;
}

int RunKernelCompile(const std::string &tiling_def, const std::string &device_code,
                     const std::string &output_file, const std::string &work_dir,
                     const std::string &tiling_repr) {
  std::string mkdir_cmd = "mkdir -p " + work_dir;
  RunCommand(mkdir_cmd);
  WriteFile(work_dir + "/device_tiling_def.h", tiling_def);
  WriteFile(work_dir + "/device_impl.cpp", device_code);

  std::string repr_arg;
  if (!tiling_repr.empty()) {
    WriteFile(work_dir + "/tiling_repr.txt", tiling_repr);
    repr_arg = ", tiling_repr=open('" + work_dir + "/tiling_repr.txt').read()";
  }

  std::string script_path = work_dir + "/run_kernel_compile.py";
  WriteFile(script_path,
    PythonPreamble() +
    "try:\n"
    "    from autofuse.compile_adapter import kernel_compile\n"
    "    import os\n"
    "    os.makedirs('" + work_dir + "', exist_ok=True)\n"
    "    td = open('" + work_dir + "/device_tiling_def.h').read()\n"
    "    dc = open('" + work_dir + "/device_impl.cpp').read()\n"
    "    argv = ['--graph_name=pgo_add_abs_inductor',\n"
    "            '--output_file=" + output_file + "',\n"
    "            '--output_path=" + work_dir + "',\n"
    "            '--soc_version=Ascend910B',\n"
    "            '--compile_options=-D_GLIBCXX_USE_CXX11_ABI=0']\n"
    "    kernel_compile(td, dc, argv" + repr_arg + ")\n"
    "except Exception:\n"
    "    traceback.print_exc()\n"
    "    sys.exit(1)\n");

  std::string cmd = "ASCEND_HOME_PATH=" + std::string(ASCEND_HOME_PATH) + " python3 " + script_path + " 2>&1";
  int ret = RunCommand(cmd);
  if (ret != 0) printf("kernel_compile failed, ret=%d, work_dir=%s\n", ret, work_dir.c_str());
  return ret;
}

struct DlHandle {
  void *ptr = nullptr;
  explicit DlHandle(void *p) : ptr(p) {}
  ~DlHandle() { if (ptr) dlclose(ptr); }
  operator bool() const { return ptr != nullptr; }
};

}  // namespace
class TestBackendPgoAddAbsInductorSplitCompile : public testing::Test {
};

void PrepareInputs(std::string &tiling_def, std::string &host_code, std::string &device_code) {
  tiling_def = ReadFile(TILING_DEF_FILE);
  host_code = ReadFile(HOST_CODE_FILE);
  device_code = ReadFile(DEVICE_CODE_FILE);
  ASSERT_FALSE(tiling_def.empty()) << "tiling_def empty";
  ASSERT_FALSE(host_code.empty()) << "host_code empty";
  ASSERT_FALSE(device_code.empty()) << "device_code empty";
}

void CompileHostAndResolve(const std::string &tiling_def, const std::string &host_code,
                           std::string &host_bin, GenerateTopnSolutionsFn &gen_fn,
                           GetTilingDataReprFn &repr_fn, GetModeledPerfForTestingFn &perf_fn,
                           AutofuseTilingFn &autofuse_tiling_fn) {
  host_bin = OUTPUT_DIR "/pgo_add_abs_inductor_host.so";
  ASSERT_EQ(RunHostCompile(tiling_def, host_code, host_bin), 0);
  ASSERT_TRUE(FileExists(host_bin)) << "host so not found: " << host_bin;

  DlHandle host_handle(dlopen(host_bin.c_str(), RTLD_LAZY | RTLD_LOCAL));
  ASSERT_TRUE(host_handle) << "dlopen host failed: " << dlerror();
  gen_fn = reinterpret_cast<GenerateTopnSolutionsFn>(dlsym(host_handle.ptr, "GenerateTopnSolutions"));
  repr_fn = reinterpret_cast<GetTilingDataReprFn>(dlsym(host_handle.ptr, "GetTilingDataRepr"));
  perf_fn = reinterpret_cast<GetModeledPerfForTestingFn>(dlsym(host_handle.ptr, "GetModeledPerfForTesting"));
  autofuse_tiling_fn = reinterpret_cast<AutofuseTilingFn>(dlsym(host_handle.ptr, "AutofuseTiling"));
  ASSERT_NE(gen_fn, nullptr) << "GenerateTopnSolutions not found";
  ASSERT_NE(repr_fn, nullptr) << "GetTilingDataRepr not found";
  ASSERT_NE(perf_fn, nullptr) << "GetModeledPerfForTesting not found";
  ASSERT_NE(autofuse_tiling_fn, nullptr) << "AutofuseTiling not found";
}

void VerifyTopnUniqueness(GenerateTopnSolutionsFn gen_fn, GetTilingDataReprFn repr_fn,
                          GetModeledPerfForTestingFn perf_fn, const std::string &default_repr,
                          const ResLimit &res_limit,
                          const std::vector<std::map<std::string, std::string>> &input_configs);

std::string GenerateTopnAndRepr(GenerateTopnSolutionsFn gen_fn, GetTilingDataReprFn repr_fn,
                                GetModeledPerfForTestingFn perf_fn, AutofuseTilingFn autofuse_tiling_fn) {
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
  ResLimit res_limit = {1, 48, 0, 192 * 1024, {0}};
  const std::vector<std::map<std::string, std::string>> input_configs;

  // 1. Reject invalid topn
  std::vector<AutofuseTilingData> invalid_tiling_datas;
  std::vector<int64_t> invalid_workspaces;
  std::vector<int64_t> invalid_block_dims;
  EXPECT_EQ(gen_fn(32, 16, 16, input_configs, 0, invalid_tiling_datas, invalid_workspaces, invalid_block_dims, &res_limit), -1);
  EXPECT_TRUE(invalid_tiling_datas.empty());
  EXPECT_TRUE(invalid_workspaces.empty());
  EXPECT_TRUE(invalid_block_dims.empty());

  // 2. Default top1 must match AutofuseTiling baseline
  AutofuseTilingData default_tiling_data = {};
  uint32_t default_workspace = 0;
  uint32_t default_block_dim = 0;
  EXPECT_EQ(autofuse_tiling_fn(32, 16, 16, &default_tiling_data, &default_workspace, &default_block_dim, &res_limit), 0);

  EXPECT_EQ(gen_fn(32, 16, 16, input_configs, 1, tiling_datas, workspaces, block_dims, &res_limit), 0);
  EXPECT_EQ(tiling_datas.size(), 1U);
  EXPECT_EQ(workspaces.size(), 1U);
  EXPECT_EQ(block_dims.size(), 1U);
  EXPECT_EQ(workspaces[0], static_cast<int64_t>(default_workspace));
  EXPECT_EQ(block_dims[0], static_cast<int64_t>(default_block_dim));

  const std::string tiling_repr = repr_fn(&tiling_datas[0]);
  const std::string default_repr = repr_fn(&default_tiling_data);
  EXPECT_FALSE(tiling_repr.empty());
  EXPECT_EQ(tiling_repr, default_repr);
  EXPECT_NE(tiling_repr.find("AutofuseTilingData{"), std::string::npos);

  // 3. Multi-candidate uniqueness and modeled_perf ascending sort
  VerifyTopnUniqueness(gen_fn, repr_fn, perf_fn, default_repr, res_limit, input_configs);
  return tiling_repr;
}

void VerifyTopnUniqueness(GenerateTopnSolutionsFn gen_fn, GetTilingDataReprFn repr_fn,
                          GetModeledPerfForTestingFn perf_fn, const std::string &default_repr,
                          const ResLimit &res_limit,
                          const std::vector<std::map<std::string, std::string>> &input_configs) {
  std::vector<AutofuseTilingData> topn_tiling_datas;
  std::vector<int64_t> topn_workspaces;
  std::vector<int64_t> topn_block_dims;
  constexpr int64_t topn = 4;
  EXPECT_EQ(gen_fn(32, 16, 16, input_configs, topn, topn_tiling_datas, topn_workspaces, topn_block_dims, const_cast<ResLimit*>(&res_limit)), 0);
  if (topn_tiling_datas.empty()) {
    return;
  }
  EXPECT_GE(topn_tiling_datas.size(), 1U);
  EXPECT_LE(static_cast<int64_t>(topn_tiling_datas.size()), topn);
  EXPECT_EQ(topn_workspaces.size(), topn_tiling_datas.size());
  EXPECT_EQ(topn_block_dims.size(), topn_tiling_datas.size());
  EXPECT_EQ(repr_fn(&topn_tiling_datas[0]), default_repr);

  std::vector<std::string> reprs;
  reprs.reserve(topn_tiling_datas.size());
  for (const auto &tiling_data : topn_tiling_datas) {
    const std::string repr = repr_fn(&tiling_data);
    EXPECT_FALSE(repr.empty());
    reprs.push_back(repr);
  }
  auto sorted_reprs = reprs;
  std::sort(sorted_reprs.begin(), sorted_reprs.end());
  const auto unique_end = std::unique(sorted_reprs.begin(), sorted_reprs.end());
  EXPECT_EQ(unique_end, sorted_reprs.end());

  std::vector<AutofuseTilingData> sorted_by_perf = topn_tiling_datas;
  std::sort(sorted_by_perf.begin(), sorted_by_perf.end(), [&](const AutofuseTilingData &lhs,
                                                             const AutofuseTilingData &rhs) {
    return perf_fn(&lhs) < perf_fn(&rhs);
  });
  for (size_t i = 0; i < sorted_by_perf.size(); ++i) {
    EXPECT_EQ(repr_fn(&sorted_by_perf[i]), repr_fn(&topn_tiling_datas[i]));
  }
}

void CompileAndVerifyKernels(const std::string &tiling_def, const std::string &device_code,
                             const std::string &tiling_repr) {
  const std::string static_dir = OUTPUT_DIR "/device_static";
  const std::string kernel_static = OUTPUT_DIR "/pgo_add_abs_inductor_static.so";
  ASSERT_EQ(RunKernelCompile(tiling_def, device_code, kernel_static, static_dir, tiling_repr), 0);
  ASSERT_TRUE(FileExists(kernel_static)) << "static device so not found: " << kernel_static;

  const std::string dynamic_dir = OUTPUT_DIR "/device_dynamic";
  const std::string kernel_dynamic = OUTPUT_DIR "/pgo_add_abs_inductor_dynamic.so";
  ASSERT_EQ(RunKernelCompile(tiling_def, device_code, kernel_dynamic, dynamic_dir, ""), 0);
  ASSERT_TRUE(FileExists(kernel_dynamic)) << "dynamic device so not found: " << kernel_dynamic;

  DlHandle static_handle(dlopen(kernel_static.c_str(), RTLD_NOW | RTLD_LOCAL));
  ASSERT_TRUE(static_handle) << "dlopen static device failed: " << dlerror();
  EXPECT_NE(dlsym(static_handle.ptr, "AutofuseLaunch"), nullptr) << "AutofuseLaunch missing in static so";

  DlHandle dynamic_handle(dlopen(kernel_dynamic.c_str(), RTLD_NOW | RTLD_LOCAL));
  ASSERT_TRUE(dynamic_handle) << "dlopen dynamic device failed: " << dlerror();
  EXPECT_NE(dlsym(dynamic_handle.ptr, "AutofuseLaunch"), nullptr) << "AutofuseLaunch missing in dynamic so";

  const std::string static_src = ReadFile(static_dir + "/device/pgo_add_abs_inductor_op_kernel.cpp");
  EXPECT_NE(static_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "static kernel should have constexpr tiling";
  EXPECT_EQ(static_src.find("const AutofuseTilingData t;"), std::string::npos)
      << "static kernel should not have non-const tiling";

  const std::string dynamic_src = ReadFile(dynamic_dir + "/device/pgo_add_abs_inductor_op_kernel.cpp");
  EXPECT_NE(dynamic_src.find("AutofuseTilingData t)"), std::string::npos)
      << "dynamic kernel should have tiling parameter";
  EXPECT_EQ(dynamic_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "dynamic kernel should not have constexpr tiling";
}

TEST_F(TestBackendPgoAddAbsInductorSplitCompile, SplitCompileChainWorks) {
  std::string tiling_def, host_code, device_code;
  PrepareInputs(tiling_def, host_code, device_code);

  std::string host_bin;
  GenerateTopnSolutionsFn gen_fn = nullptr;
  GetTilingDataReprFn repr_fn = nullptr;
  GetModeledPerfForTestingFn perf_fn = nullptr;
  AutofuseTilingFn autofuse_tiling_fn = nullptr;
  CompileHostAndResolve(tiling_def, host_code, host_bin, gen_fn, repr_fn, perf_fn, autofuse_tiling_fn);

  std::string tiling_repr = GenerateTopnAndRepr(gen_fn, repr_fn, perf_fn, autofuse_tiling_fn);
  ASSERT_FALSE(tiling_repr.empty());

  CompileAndVerifyKernels(tiling_def, device_code, tiling_repr);
}
