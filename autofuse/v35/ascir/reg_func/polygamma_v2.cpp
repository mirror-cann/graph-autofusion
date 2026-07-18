/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "default_reg_func_v2.h"

namespace af {
namespace ascir {
std::vector<std::unique_ptr<af::TmpBufDesc>> CalcPolygammaTmpSizeV2(const af::AscNode &node) {
  constexpr uint32_t POLYGAMMA_FLOAT_CALC_PROC = 8;
  constexpr uint32_t POLYGAMMA_ONE_REPEAT_BYTE_SIZE = 256;
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  uint32_t calc_tmp_buf = POLYGAMMA_ONE_REPEAT_BYTE_SIZE * POLYGAMMA_FLOAT_CALC_PROC;
  GELOGD("Node %s[%s] temp buffer size: %u", node.GetTypePtr(), node.GetNamePtr(), calc_tmp_buf);
  Expression tmp_size = af::Symbol(calc_tmp_buf);
  af::TmpBufDesc desc = {tmp_size, -1};
  std::vector<std::unique_ptr<af::TmpBufDesc>> tmp_buf_descs;
  tmp_buf_descs.emplace_back(std::make_unique<af::TmpBufDesc>(desc));
  return tmp_buf_descs;
}
}  // namespace ascir
}  // namespace af
