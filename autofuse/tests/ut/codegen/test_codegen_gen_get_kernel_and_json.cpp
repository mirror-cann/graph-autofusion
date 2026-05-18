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
#include <fstream>
#include <vector>
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
  fused_schedule_result.fused_graph_name = af::AscendString("test_graph");
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
  fused_schedule_result.fused_graph_name = af::AscendString("test_graph");
  fused_schedule_result.node_idx_to_scheduled_results.push_back({});
  std::map<std::string, std::string> tiling_file_name_to_content;

  EXPECT_EQ(codegen.GenerateTilingForInductor(fused_schedule_result, tiling_file_name_to_content), ge::FAILED);
}

TEST_F(CodegenTest, GenerateTilingForInductor_LegacyApi_ShouldReturnContent_WhenInnerTilingSucceeds)
{
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen.tiling_lib_.codegen_func_ = SuccessCodegenFunc;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString("test_graph");
  fused_schedule_result.node_idx_to_scheduled_results.push_back({});

  auto result = codegen.GenerateTilingForInductor(fused_schedule_result);
  EXPECT_FALSE(result.empty());
}

TEST_F(CodegenTest, GenerateTiling_LegacyApi_ShouldReturnContent_WhenInnerTilingSucceeds)
{
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen.tiling_lib_.codegen_func_ = SuccessCodegenFunc;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString("test_graph");
  fused_schedule_result.node_idx_to_scheduled_results.push_back({});
  std::map<std::string, std::string> shape_info;

  auto result = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "0");
  EXPECT_FALSE(result.empty());
}

// ==================== GenGetKernelAndJson 新增功能测试 ====================

TEST_F(CodegenTest, GenGetKernelAndJson_ShouldUseStaticConstArray_WhenKernelFileExists)
{
  codegen::CodegenOptions opt;
  codegen::Codegen codegen(opt);
  
  std::string kernel_path = "/tmp/test_kernel_conv2d.bin";
  std::string json_path = "/tmp/test_kernel_conv2d.json";
  
  std::ofstream kernel_file(kernel_path, std::ios::binary);
  std::vector<uint8_t> kernel_data = {0x01, 0x02, 0x03, 0x04, 0x05};
  kernel_file.write(reinterpret_cast<char*>(kernel_data.data()), kernel_data.size());
  kernel_file.flush();
  kernel_file.close();
  
  std::ofstream json_file(json_path);
  json_file << "{}";
  json_file.close();
  
  std::string result = codegen.GenGetKernelAndJson(kernel_path, json_path);
  
  EXPECT_NE(result.find("static const uint8_t temp_kernel[]"), std::string::npos);
  EXPECT_NE(result.find("sizeof(temp_kernel)"), std::string::npos);
  
  std::remove(kernel_path.c_str());
  std::remove(json_path.c_str());
}

TEST_F(CodegenTest, GenGetKernelAndJson_ShouldUseTempDataNotDataMethod_WhenKernelFileExists)
{
  codegen::CodegenOptions opt;
  codegen::Codegen codegen(opt);
  
  std::string kernel_path = "/tmp/test_kernel_conv2d2.bin";
  std::string json_path = "/tmp/test_kernel_conv2d2.json";
  
  std::ofstream kernel_file(kernel_path, std::ios::binary);
  std::vector<uint8_t> kernel_data = {0xAA, 0xBB, 0xCC};
  kernel_file.write(reinterpret_cast<char*>(kernel_data.data()), kernel_data.size());
  kernel_file.flush();
  kernel_file.close();
  
  std::ofstream json_file(json_path);
  json_file << "{}";
  json_file.close();
  
  std::string result = codegen.GenGetKernelAndJson(kernel_path, json_path);
  
  EXPECT_NE(result.find("kernel_bin.data(), temp_kernel"), std::string::npos);
  EXPECT_EQ(result.find("temp_kernel.data()"), std::string::npos);
  
  std::remove(kernel_path.c_str());
  std::remove(json_path.c_str());
}

TEST_F(CodegenTest, GenGetKernelAndJson_ShouldResizeWithSizeof_WhenKernelFileExists)
{
  codegen::CodegenOptions opt;
  codegen::Codegen codegen(opt);
  
  std::string kernel_path = "/tmp/test_kernel_conv2d3.bin";
  std::string json_path = "/tmp/test_kernel_conv2d3.json";
  
  std::ofstream kernel_file(kernel_path, std::ios::binary);
  std::vector<uint8_t> kernel_data(128, 0x42);
  kernel_file.write(reinterpret_cast<char*>(kernel_data.data()), kernel_data.size());
  kernel_file.flush();
  kernel_file.close();
  
  std::ofstream json_file(json_path);
  json_file << "{}";
  json_file.close();
  
  std::string result = codegen.GenGetKernelAndJson(kernel_path, json_path);
  
  EXPECT_NE(result.find("kernel_bin.resize(sizeof(temp_kernel))"), std::string::npos);
  
  std::remove(kernel_path.c_str());
  std::remove(json_path.c_str());
}

TEST_F(CodegenTest, GenGetKernelAndJson_ShouldReturnEmptyVector_WhenKernelFileEmpty)
{
  codegen::CodegenOptions opt;
  codegen::Codegen codegen(opt);
  
  std::string kernel_path = "/tmp/empty_kernel_conv2d.bin";
  std::string json_path = "/tmp/empty_kernel_conv2d.json";
  
  std::ofstream kernel_file(kernel_path, std::ios::binary);
  kernel_file.flush();
  kernel_file.close();
  
  std::ofstream json_file(json_path);
  json_file << "{}";
  json_file.close();
  
  std::string result = codegen.GenGetKernelAndJson(kernel_path, json_path);
  
  EXPECT_NE(result.find("std::vector<uint8_t> temp_kernel = {}"), std::string::npos);
  
  std::remove(kernel_path.c_str());
  std::remove(json_path.c_str());
}