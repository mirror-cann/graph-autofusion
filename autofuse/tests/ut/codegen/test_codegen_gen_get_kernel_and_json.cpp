/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#define private public
#include "codegen.h"
#undef private

using namespace ge;
using namespace codegen;
using namespace testing;

namespace {
bool FailingCodegenFunc(const std::string &op_name, const ::ascir::FusedScheduledResult& fused_schedule_result,
                        std::map<std::string, std::string> &options,
                        std::map<std::string, std::string> &tiling_file_name_to_content, bool is_inductor_scene) {
  (void)op_name;
  (void)fused_schedule_result;
  (void)options;
  (void)tiling_file_name_to_content;
  (void)is_inductor_scene;
  return false;
}

bool SuccessCodegenFunc(const std::string &op_name, const ::ascir::FusedScheduledResult& fused_schedule_result,
                        std::map<std::string, std::string> &options,
                        std::map<std::string, std::string> &tiling_file_name_to_content, bool is_inductor_scene) {
  (void)op_name;
  (void)fused_schedule_result;
  (void)options;
  (void)is_inductor_scene;
  return true;
}
}  // namespace

class CodegenTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CodegenTest, GenerateTiling_ShouldReturnFailed_WhenInnerTilingCodegenFails)
{
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen.tiling_lib_.codegen_func_ = FailingCodegenFunc;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("test_graph");
  fused_schedule_result.node_idx_to_scheduled_results.push_back({});
  std::map<std::string, std::string> shape_info;
  std::map<std::string, std::string> tiling_file_name_to_content;

  EXPECT_EQ(codegen.GenerateTiling(fused_schedule_result, shape_info, "", "0", tiling_file_name_to_content),
            ge::FAILED);
}

TEST_F(CodegenTest, GenGetKernelAndJson_ShouldReturnInvalidString_WhenKernelPathIsInvalid)
{
  codegen::CodegenOptions opt;
  codegen::Codegen codegen(opt);
  std::string kernel_path = "invalid_kernel_path";
  std::string json_path = "invalid_json_path";
  std::string result = codegen.GenGetKernelAndJson(kernel_path, json_path);
  EXPECT_EQ(result, "");
}

TEST_F(CodegenTest, GenerateTilingForInductor_ShouldReturnFailed_WhenInnerTilingCodegenFails)
{
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen.tiling_lib_.codegen_func_ = FailingCodegenFunc;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("test_graph");
  fused_schedule_result.node_idx_to_scheduled_results.push_back({});
  std::map<std::string, std::string> tiling_file_name_to_content;

  EXPECT_EQ(codegen.GenerateTilingForInductor(fused_schedule_result, tiling_file_name_to_content), ge::FAILED);
}

TEST_F(CodegenTest, GenerateTilingForInductor_LegacyApi_ShouldReturnContent_WhenInnerTilingSucceeds)
{
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen.tiling_lib_.codegen_func_ = SuccessCodegenFunc;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("test_graph");
  fused_schedule_result.node_idx_to_scheduled_results.push_back({});

  auto result = codegen.GenerateTilingForInductor(fused_schedule_result);
  EXPECT_FALSE(result.empty());
}

TEST_F(CodegenTest, GenerateTiling_LegacyApi_ShouldReturnContent_WhenInnerTilingSucceeds)
{
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen.tiling_lib_.codegen_func_ = SuccessCodegenFunc;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("test_graph");
  fused_schedule_result.node_idx_to_scheduled_results.push_back({});
  std::map<std::string, std::string> shape_info;

  auto result = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "0");
  EXPECT_FALSE(result.empty());
}