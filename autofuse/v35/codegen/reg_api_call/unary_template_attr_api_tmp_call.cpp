/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unary_template_attr_api_tmp_call.h"

#include <sstream>
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"
#include "common/checker.h"

namespace codegen {
using namespace std;

Status UnaryTemplateAttrApiTmpCall::ParseAttr(const ascir::NodeView &node) {
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("n", this->n_), "Failed to get attr n, node = %s",
                          node->GetNamePtr());
  return af::SUCCESS;
}

Status UnaryTemplateAttrApiTmpCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                             const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                             const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                             std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  int64_t life_time_axis_id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "UnaryTemplateAttrApiTmpCall cannot find tmp buffer id to use.");
  int64_t id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x.actual_size.Str()),
                               tpipe.tmp_buf.name + "_" + std::to_string(id));

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x.dtype, dtype_name), "UnaryTemplateAttrApiTmpCall get data type:%d failed",
                    static_cast<int32_t>(x.dtype));

  stringstream ss;
  ss << this->api_name_ << "<" << dtype_name << ", " << this->n_ << ">(" << y << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << tpipe.tmp_buf << "_" << std::to_string(id)
     << ", " << x.actual_size << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<UnaryTemplateAttrApiTmpCall> register_unary_template_attr_api_tmp_call(
    "UnaryTemplateAttrApiTmpCall");
}  // namespace codegen
