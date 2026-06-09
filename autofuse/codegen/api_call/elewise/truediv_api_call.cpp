/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "truediv_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils//asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status TrueDivApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  size_t x1_idx = 0;
  size_t x2_idx = 1;
  bool switch_scalar = false;
  // 对于input[0]为Data节点，input[1]为Scalar节点，scheduler无法调换顺序，需要codegen调换顺序
  // input[0] input[1] 都为Scalar输入的情况不进行调换
  if (inputs[0].get().IsAnyScalar() && !(inputs[1].get().IsAnyScalar())) {
    x1_idx = 1;
    x2_idx = 0;
    switch_scalar = true;
  }
  auto x1 = inputs[x1_idx].get();
  auto x2 = inputs[x2_idx].get();

  GELOGD("x2, is_constant:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d",
         static_cast<int32_t>(x2.is_constant), static_cast<int32_t>(x2.is_ub_scalar),
         static_cast<int32_t>(x2.need_gen_get_value_of_ub_scalar));

  auto y = outputs[0].get();
  stringstream ss;
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "TrueDivApiCall cannot find tmp buffer id to use.");
  id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x1.actual_size.Str()), tpipe.tmp_buf.name + "_" + std::to_string(id));

  std::string x1_dtype_name;
  std::string x2_dtype_name;
  std::string y_dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x1.dtype, x1_dtype_name), "Codegen get x1 data type:%d failed",
                    static_cast<int32_t>(x1.dtype));
  GE_CHK_STATUS_RET(Tensor::DtypeName(x2.dtype, x2_dtype_name), "Codegen get x2 data type:%d failed",
                    static_cast<int32_t>(x2.dtype));
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, y_dtype_name), "Codegen get y data type:%d failed",
                    static_cast<int32_t>(y.dtype));
  GE_ASSERT_TRUE(x1_dtype_name == x2_dtype_name, "x1_dtype_name:%s, x2_dtype_name:%s", x1_dtype_name.c_str(),
                 x2_dtype_name.c_str());
  const std::string is_scalar_latter = switch_scalar ? "false" : "true";

  if (x1.IsAnyScalar() && x2.IsAnyScalar()) { // 两个输入都是Scalar
    ss << this->api_name_ << "s<" << x1_dtype_name << ", " << y_dtype_name << ">("
       << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << "(" << x1_dtype_name << ")" << x1.GetScalarValue() << ", "
       << "(" << x1_dtype_name << ")" << x2.GetScalarValue()
       << ");" << std::endl;
  } else if (x1.IsAnyScalar() || x2.IsAnyScalar()) { // 只有1个输入是Scalar
    GE_ASSERT_TRUE(id != -1L, "TrueDivApiCall cannot find tmp buffer id to use.");
    ss << this->api_name_ << "s<" << x1_dtype_name << ", " << y_dtype_name << ", " << is_scalar_latter << ">("
       << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], "
       << "(" << x1_dtype_name << ")" << x2.GetScalarValue() << ", "
       << tpipe.tmp_buf << "_" << std::to_string(id) << ", "
       << x1.actual_size << ");" << std::endl;
  } else {
    GE_ASSERT_TRUE(id != -1L, "TrueDivApiCall cannot find tmp buffer id to use.");
    ss << this->api_name_ << "<" << x1_dtype_name << ", " << y_dtype_name << ">("
       << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], "
       << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], "
       << tpipe.tmp_buf << "_" << std::to_string(id) << ", " << x1.actual_size << ");" << std::endl;
  }

  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<TrueDivApiCall> register_truediv_api_call("TrueDivApiCall");

}  // namespace codegen