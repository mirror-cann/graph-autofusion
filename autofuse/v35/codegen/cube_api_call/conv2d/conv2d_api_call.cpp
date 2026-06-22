/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "conv2d_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status Conv2DApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_ASSERT_SUCCESS(ascgen_utils::ParseConv2DAttr(node, conv_attr_data_));
  return ge::SUCCESS;
}

Status Conv2DApiCall::PreProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                 const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                 std::string &result) const {
  (void)tpipe;
  (void)current_axis;
  (void)outputs;
  (void)result;
  return ge::SUCCESS;
}

Status Conv2DApiCall::GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const {
  (void)tpipe;
  (void)tiler;
  (void)ss;
  return ge::SUCCESS;
}

Status Conv2DApiCall::PostProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  (void)tpipe;
  (void)current_axis;
  (void)outputs;
  (void)result;
  return ge::SUCCESS;
}

Status Conv2DApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                               std::string &result) const {
  (void)tpipe;
  (void)current_axis;
  (void)result;
  return ge::SUCCESS;
}

Status Conv2DApiCall::GenerateMacro(std::string &result) const {
  stringstream ss;

  // 后面抽取专用基类, 此处先覆盖下
  Tiler tiler;
  TPipe t_pipe("tpipe", tiler);
  GE_ASSERT_SUCCESS(PreProcess(t_pipe, {}, {}, result));
  GE_ASSERT_SUCCESS(GenerateFuncDefinition(t_pipe, tiler, ss));
  GE_ASSERT_SUCCESS(Generate(t_pipe, {}, result));
  GE_ASSERT_SUCCESS(PostProcess(t_pipe, {}, {}, result));
  return ge::SUCCESS;
}

static ApiCallRegister<Conv2DApiCall> register_conv2d_api_call("Conv2DApiCall");
}  // namespace codegen
