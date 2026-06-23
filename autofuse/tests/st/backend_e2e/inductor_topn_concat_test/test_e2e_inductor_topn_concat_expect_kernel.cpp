/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "tikicpulib.h"
#include "autofuse_tiling_data.h"

struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};

extern "C" int64_t GenerateTopnSolutions(const std::vector<std::map<std::string, std::string>> &input_configs,
                                         int64_t topn, std::vector<AutofuseTilingData> &tiling_datas,
                                         std::vector<int64_t> &workspaces, std::vector<int64_t> &block_dims,
                                         ResLimit *res_limit = nullptr);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint32_t *workspaceSize, uint32_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);
std::string GetTilingDataRepr(const AutofuseTilingData *tiling_data);

namespace {
constexpr uint32_t kUbReservedSize = 256;

struct DefaultTilingResult {
  AutofuseTilingData tiling_data = {};
  uint32_t workspace = 0;
  uint32_t block_dim = 0;
};

testing::AssertionResult GenerateDefaultTiling(const ResLimit &res_limit, DefaultTilingResult &default_tiling) {
  const int64_t ret = AutofuseTiling(&default_tiling.tiling_data, &default_tiling.workspace, &default_tiling.block_dim,
                                     res_limit.aiv_num, res_limit.ub_size - kUbReservedSize);
  if (ret != 0) {
    return testing::AssertionFailure() << "AutofuseTiling failed, ret: " << ret;
  }
  return testing::AssertionSuccess();
}
}  // namespace

class E2EBackendInductorTopnConcatCode : public testing::Test {};

TEST_F(E2EBackendInductorTopnConcatCode, GenerateTopnSolutionsTop1MatchesAutofuseTiling) {
  ResLimit res_limit = {1, 48, 0, 192 * 1024, {0}};
  const std::vector<std::map<std::string, std::string>> input_configs;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;

  DefaultTilingResult default_tiling;
  ASSERT_TRUE(GenerateDefaultTiling(res_limit, default_tiling));

  ASSERT_EQ(GenerateTopnSolutions(input_configs, 1, tiling_datas, workspaces, block_dims, &res_limit), 0);
  ASSERT_EQ(tiling_datas.size(), 1U);
  ASSERT_EQ(workspaces.size(), 1U);
  ASSERT_EQ(block_dims.size(), 1U);
  EXPECT_EQ(GetTilingDataRepr(&tiling_datas[0]), GetTilingDataRepr(&default_tiling.tiling_data));
  EXPECT_EQ(workspaces[0], static_cast<int64_t>(default_tiling.workspace));
  EXPECT_EQ(block_dims[0], static_cast<int64_t>(default_tiling.block_dim));
}

TEST_F(E2EBackendInductorTopnConcatCode, GenerateTopnSolutionsRejectsInvalidTopn) {
  ResLimit res_limit = {1, 48, 0, 192 * 1024, {0}};
  const std::vector<std::map<std::string, std::string>> input_configs;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;

  EXPECT_EQ(GenerateTopnSolutions(input_configs, 0, tiling_datas, workspaces, block_dims, &res_limit), -1);
  EXPECT_TRUE(tiling_datas.empty());
  EXPECT_TRUE(workspaces.empty());
  EXPECT_TRUE(block_dims.empty());
}

// Verify that dedup yields at least 1 valid candidate for topn>1 request on single-tiling-key fixture
TEST_F(E2EBackendInductorTopnConcatCode, GenerateTopnSolutionsReturnsDeduplicatedCandidates) {
  ResLimit res_limit = {1, 48, 0, 192 * 1024, {0}};
  const std::vector<std::map<std::string, std::string>> input_configs;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;

  DefaultTilingResult default_tiling;
  ASSERT_TRUE(GenerateDefaultTiling(res_limit, default_tiling));

  constexpr int64_t topn = 5;
  EXPECT_EQ(GenerateTopnSolutions(input_configs, topn, tiling_datas, workspaces, block_dims, &res_limit), 0);

  // This fixture has only 1 tiling key per group; dedup yields single result.
  ASSERT_GE(tiling_datas.size(), 1U);
  ASSERT_EQ(workspaces.size(), 1U);
  ASSERT_EQ(block_dims.size(), 1U);

  const std::string repr = GetTilingDataRepr(&tiling_datas[0]);
  EXPECT_EQ(repr, GetTilingDataRepr(&default_tiling.tiling_data));
  EXPECT_EQ(workspaces[0], static_cast<int64_t>(default_tiling.workspace));
  EXPECT_EQ(block_dims[0], static_cast<int64_t>(default_tiling.block_dim));
}
