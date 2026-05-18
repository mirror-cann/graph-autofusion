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

extern "C" int64_t GenerateTopnSolutions(
                                          const std::vector<std::map<std::string, std::string>> &input_configs,
                                          int64_t topn,
                                          std::vector<AutofuseTilingData> &tiling_datas,
                                          std::vector<int64_t> &workspaces,
                                          std::vector<int64_t> &block_dims, ResLimit *res_limit = nullptr);
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim,
                                   uint32_t aiv_num, uint32_t ub_size);
std::string GetTilingDataRepr(const AutofuseTilingData *tiling_data);

class E2EBackendInductorTopnConcatCode : public testing::Test {
};

TEST_F(E2EBackendInductorTopnConcatCode, GenerateTopnSolutionsTop1MatchesAutofuseTiling) {
  ResLimit res_limit = {1, 48, 0, 192 * 1024, {0}};
  const std::vector<std::map<std::string, std::string>> input_configs;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;

  AutofuseTilingData default_tiling_data = {};
  uint32_t default_workspace = 0;
  uint32_t default_block_dim = 0;
  ASSERT_EQ(AutofuseTiling(&default_tiling_data, &default_workspace, &default_block_dim,
                           res_limit.aiv_num, res_limit.ub_size - 256), 0);

  ASSERT_EQ(GenerateTopnSolutions(input_configs, 1, tiling_datas, workspaces, block_dims, &res_limit), 0);
  ASSERT_EQ(tiling_datas.size(), 1U);
  ASSERT_EQ(workspaces.size(), 1U);
  ASSERT_EQ(block_dims.size(), 1U);
  EXPECT_EQ(GetTilingDataRepr(&tiling_datas[0]), GetTilingDataRepr(&default_tiling_data));
  EXPECT_EQ(workspaces[0], static_cast<int64_t>(default_workspace));
  EXPECT_EQ(block_dims[0], static_cast<int64_t>(default_block_dim));
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

  constexpr int64_t topn = 5;
  EXPECT_EQ(GenerateTopnSolutions(input_configs, topn, tiling_datas, workspaces, block_dims, &res_limit), 0);

  // This fixture has only 1 tiling key per group; dedup yields single result.
  ASSERT_GE(tiling_datas.size(), 1U);
  ASSERT_EQ(workspaces.size(), 1U);
  ASSERT_EQ(block_dims.size(), 1U);

  const std::string repr = GetTilingDataRepr(&tiling_datas[0]);
  EXPECT_FALSE(repr.empty());
  EXPECT_NE(repr.find("AutofuseTilingData{"), std::string::npos);
  EXPECT_GT(block_dims[0], 0);
  EXPECT_GE(workspaces[0], 0);
}
