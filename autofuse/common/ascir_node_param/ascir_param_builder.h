/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_NODE_PARAM_ASCIR_PARAM_BUILDER_H_
#define ASCIR_NODE_PARAM_ASCIR_PARAM_BUILDER_H_

#include "ascir_node_param/ascir_node_param.h"

namespace ascir_param {
bool IsReduceParamSupported(const std::string &api_name);
af::Status EnrichAscirGraphNodeParams(const af::AscGraph &graph);
}  // namespace ascir_param

#endif  // ASCIR_NODE_PARAM_ASCIR_PARAM_BUILDER_H_
