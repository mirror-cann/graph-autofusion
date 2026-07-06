/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_specific_params_builder.h"

#include "ascir_node_param/ascir_node_param.h"
#include "common/checker.h"

namespace att {
af::Status FillReduceSpecificParams(const af::AscNodePtr &ge_node, NodeInfo &node_info) {
  const auto params = ascir_param::GetAscirNodeParams(ge_node);
  if (params == nullptr) {
    return af::SUCCESS;
  }
  if (params->status == ascir_param::ParamBuildStatus::kSkipped) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(params->status == ascir_param::ParamBuildStatus::kBuilt, "Build ascir node params failed, node[%s].",
                 node_info.name.c_str());
  const auto *reduce = ascir_param::GetSpecificParams<ascir_param::ReduceNodeParams>(*params);
  GE_ASSERT_NOTNULL(reduce, "Reduce specific params is null, node[%s].", node_info.name.c_str());
  node_info.reduce_specific_params = *reduce;
  return af::SUCCESS;
}
}  // namespace att
