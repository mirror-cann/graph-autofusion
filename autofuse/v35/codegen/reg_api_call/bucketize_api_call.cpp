/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "bucketize_api_call.h"

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

Status BucketizeApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("right", this->right_),
                          "Failed to get Bucketize right attr, node = %s", node->GetNamePtr());
  GELOGI("Bucketize name:%s, right:%d", node->GetNamePtr(), static_cast<int32_t>(this->right_));
  return af::SUCCESS;
}

Status BucketizeApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  size_t x1_idx = 0;
  size_t x2_idx = 1;

  auto x1 = inputs[x1_idx].get();
  auto x2 = inputs[x2_idx].get();

  auto y = outputs[0].get();
  stringstream ss;
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "BucketizeApiCall cannot find tmp buffer id to use.");
  id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x1.actual_size.Str()),
                               tpipe.tmp_buf.name + "_" + std::to_string(id));

  // BucketizeExtend(dst, src, boundaries, sharedTmpBuffer, calCount, boundCount, right)
  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1 << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2 << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], " << tpipe.tmp_buf << "_" << std::to_string(id)
     << ", " << x1.actual_size << ", " << x2.actual_size << ", " << (this->right_ ? "true" : "false") << ");"
     << std::endl;

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<BucketizeApiCall> register_bucketize_api_call("BucketizeApiCall");

}  // namespace codegen
