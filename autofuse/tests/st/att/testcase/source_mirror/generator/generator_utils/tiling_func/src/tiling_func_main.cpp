/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "kernel_context_holder_builder.h"
#include "stub_info.h"
#include "struct_info.h"
using namespace optiling;
using namespace att;
int case_index = 0;
uint64_t RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024U;
bool RunCase(std::vector<std::vector<int64_t>> shape, uint64_t expect_tiling_key) {
  case_index++;
  KernelContextHolderBuilder builder;
  builder.AddInput(InOutput(ge::GeShape(shape[0]), ge::FORMAT_ND, ge::DT_FLOAT16)) // x1
         .AddInput(InOutput(ge::GeShape(shape[1]), ge::FORMAT_ND, ge::DT_FLOAT16)) // x2
         .AddInput(InOutput(ge::GeShape(shape[3]), ge::FORMAT_ND, ge::DT_FLOAT16)) // gamma
         .AddInput(InOutput(ge::GeShape(shape[4]), ge::FORMAT_ND, ge::DT_FLOAT16)) // beta
         .AddInput(InOutput(ge::GeShape(shape[2]), ge::FORMAT_ND, ge::DT_FLOAT16)); // bias
  std::vector<int64_t> output_shape = shape[0];
  output_shape[shape[0].size() - 1] = 1;
  auto tiling_context_holder = builder
                                   .AddOutput(InOutput(ge::GeShape(shape[0]), ge::FORMAT_ND, ge::DT_FLOAT16))    // y
                                   .AddOutput(InOutput(ge::GeShape(output_shape), ge::FORMAT_ND, ge::DT_FLOAT))  // mean
                                   .AddOutput(InOutput(ge::GeShape(output_shape), ge::FORMAT_ND, ge::DT_FLOAT))  // rtsd
                                   .AddOutput(InOutput(ge::GeShape(shape[0]), ge::FORMAT_ND, ge::DT_FLOAT16))    // x
                                   .SetTilingData(10240)
                                   .SetWorkSpace(1600)
                                   .SetCompileInfo(2)
                                   .SetPlatformInfo()
                                   .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
                                   .AddPrivateAtt({"additional_output", ge::AnyValue::CreateFrom<bool>(true)})
                                   .Build();
  gert::TilingContext *tiling_context = reinterpret_cast<gert::TilingContext *>(tiling_context_holder.context_);
  const auto status = GetTiling(tiling_context);
  if ((status) == ge::GRAPH_SUCCESS) {
    GET_TILING_DATA(tmpTiling, tiling_context->GetRawTilingData()->GetData());
    PrintTilingData(tmpTiling);
    if (tmpTiling.A == 0 && tmpTiling.R == 0) {
      std::cout << "Original axis is 0." << std::endl;
      return false;
    }
    if (tiling_context->GetTilingKey() == expect_tiling_key) {
      if (tiling_context->GetTilingKey() != 1151 ||
          tiling_context->GetWorkspaceSizes(1)[0] >= RESERVED_WORKSPACE_SIZE_910B + 4 * shape[0][0] * shape[0][1]) {
        std::cout << "Case " << case_index << " tiling func execute success." << std::endl;
        return true;
      }
    }
  }
  std::cout << "Case " << case_index << " tiling func execute failed." << std::endl;
  return false;
}

struct TestCase {
  std::vector<std::vector<int64_t>> shapes;
  uint64_t expect_key;
};

int main() {
  const TestCase cases[] = {
    {{{1536, 128}, {1536, 128}, {1536, 128}, {128}, {128}}, 1101},
    {{{1000, 1024}, {1000, 1024}, {1000, 1024}, {1024}, {1024}}, 1101},
    {{{1000, 2048}, {1000, 2048}, {1000, 2048}, {2048}, {2048}}, 1101},
    {{{1000, 4096}, {1000, 4096}, {1000, 4096}, {4096}, {4096}}, 1101},
    {{{1000, 8192}, {1000, 8192}, {1000, 8192}, {8192}, {8192}}, 1101},
    {{{1000, 16384}, {1000, 16384}, {1000, 16384}, {16384}, {16384}}, 1111},
    {{{1000, 32768}, {1000, 32768}, {1000, 32768}, {32768}, {32768}}, 1151},
    {{{200, 65536}, {200, 65536}, {200, 65536}, {65536}, {65536}}, 1151},
    {{{200, 131072}, {200, 131072}, {200, 131072}, {131072}, {131072}}, 1151},
    {{{80, 262144}, {80, 262144}, {80, 262144}, {262144}, {262144}}, 1151},
    {{{40, 524288}, {40, 524288}, {40, 524288}, {524288}, {524288}}, 1151},
    {{{40, 1048576}, {40, 1048576}, {40, 1048576}, {1048576}, {1048576}}, 1151},
    {{{960, 1024}, {960, 1024}, {960, 1024}, {1024}, {1024}}, 1101},
    {{{1000, 23456}, {1000, 23456}, {1000, 23456}, {23456}, {23456}}, 1111},
    {{{10000, 61}, {10000, 61}, {10000, 61}, {61}, {61}}, 1101},
    {{{500, 11000}, {500, 11000}, {500, 11000}, {11000}, {11000}}, 1101},
    {{{8, 1234567}, {8, 1234567}, {8, 1234567}, {1234567}, {1234567}}, 1151},
    {{{4567, 1567}, {4567, 1567}, {4567, 1567}, {1567}, {1567}}, 1101},
    {{{4567, 2345}, {4567, 2345}, {4567, 2345}, {2345}, {2345}}, 1101},
    {{{4321, 4567}, {4321, 4567}, {4321, 4567}, {4567}, {4567}}, 1101},
    {{{64, 24, 128}, {64, 24, 128}, {64, 24, 128}, {128}, {128}}, 1101},
    {{{1000, 1024}, {1000, 1024}, {1024}, {1024}, {1024}}, 1102},
    {{{1000, 16384}, {1000, 16384}, {16384}, {16384}, {16384}}, 1112},
    {{{200, 65536}, {200, 65536}, {65536}, {65536}, {65536}}, 1152},
  };
  bool ret = true;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    ret &= RunCase(cases[i].shapes, cases[i].expect_key);
    std::cout << "case " << (i + 1) << " -- " << ret << std::endl;
  }
  return ret ? 0 : -1;
}