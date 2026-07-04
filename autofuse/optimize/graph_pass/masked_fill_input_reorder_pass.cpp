/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "masked_fill_input_reorder_pass.h"

#include "ascir_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "schedule_utils.h"
#include "graph/utils/op_desc_utils_ex.h"

namespace optimize {
namespace {
constexpr int32_t kMaskedFillInputXIndex = 0;
constexpr int32_t kMaskedFillInputMaskIndex = 1;
constexpr int32_t kMaskedFillInputValueIndex = 2;
constexpr size_t kMaskedFillMinInputCount = 3;

constexpr int32_t kSelectInputMaskIndex = 0;
constexpr int32_t kSelectInputValueIndex = 1;
constexpr int32_t kSelectInputXIndex = 2;

ge::DataType GetDtypeFromInput(const af::InDataAnchor *in_anchor) {
  if (in_anchor == nullptr) {
    return ge::DT_UNDEFINED;
  }
  auto peer_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_anchor == nullptr) {
    return ge::DT_UNDEFINED;
  }
  auto owner_node = peer_anchor->GetOwnerNode();
  if (owner_node == nullptr) {
    return ge::DT_UNDEFINED;
  }
  auto asc_node = std::dynamic_pointer_cast<af::AscNode>(owner_node);
  if (asc_node == nullptr) {
    return ge::DT_UNDEFINED;
  }
  auto outputs = asc_node->outputs();
  auto output_index = static_cast<size_t>(peer_anchor->GetIdx());
  if (output_index >= outputs.size()) {
    return ge::DT_UNDEFINED;
  }
  return outputs[output_index]->attr.dtype;
}

Status UpdateInputDtypes(const af::OpDescPtr &op_desc, const std::shared_ptr<af::AscNode> &node,
                         ge::DataType dtype_mask, ge::DataType dtype_value, ge::DataType dtype_x) {
  node->inputs[kSelectInputMaskIndex].attr.dtype = dtype_mask;
  node->inputs[kSelectInputValueIndex].attr.dtype = dtype_value;
  node->inputs[kSelectInputXIndex].attr.dtype = dtype_x;

  auto input_desc_mask = op_desc->MutableInputDesc(kSelectInputMaskIndex);
  auto input_desc_value = op_desc->MutableInputDesc(kSelectInputValueIndex);
  auto input_desc_x = op_desc->MutableInputDesc(kSelectInputXIndex);
  GE_ASSERT_NOTNULL(input_desc_mask);
  GE_ASSERT_NOTNULL(input_desc_value);
  GE_ASSERT_NOTNULL(input_desc_x);
  input_desc_mask->SetDataType(dtype_mask);
  input_desc_value->SetDataType(dtype_value);
  input_desc_x->SetDataType(dtype_x);
  return af::SUCCESS;
}

Status ReorderMaskedFillInput(const std::shared_ptr<af::AscNode> &node) {
  GELOGD("MaskedFillInputReorderPass: reorder inputs for node %s[%s]", node->GetTypePtr(), node->GetNamePtr());

  auto in_anchors = node->GetAllInDataAnchorsPtr();
  if (in_anchors.size() < kMaskedFillMinInputCount) {
    GELOGD("MaskedFillInputReorderPass: node %s has %zu inputs (< %zu), skip reorder.", node->GetNamePtr(),
           in_anchors.size(), kMaskedFillMinInputCount);
    return af::SUCCESS;
  }

  ge::DataType dtype_x = GetDtypeFromInput(in_anchors[kMaskedFillInputXIndex]);
  ge::DataType dtype_mask = GetDtypeFromInput(in_anchors[kMaskedFillInputMaskIndex]);
  ge::DataType dtype_value = GetDtypeFromInput(in_anchors[kMaskedFillInputValueIndex]);

  GE_ASSERT_SUCCESS(ScheduleUtils::SwapInputIndex(node, kMaskedFillInputXIndex, kMaskedFillInputMaskIndex));
  GE_ASSERT_SUCCESS(ScheduleUtils::SwapInputIndex(node, kMaskedFillInputMaskIndex, kMaskedFillInputValueIndex));

  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  af::ascir_op::Select select_op("tmp_select");
  op_desc->SetType(af::ascir_op::Select::Type);
  op_desc->SetIrRelated(af::OpDescUtils::GetOpDescFromOperator(select_op));
  af::OpDescUtilsEx::ResetFuncHandle(op_desc);
  select_op.BreakConnect();
  node->attr.type = af::ascir_op::Select::Type;

  return UpdateInputDtypes(op_desc, node, dtype_mask, dtype_value, dtype_x);
}
}  // namespace

Status MaskedFillInputReorderPass::RunPass(af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() != "MaskedFill") {
      continue;
    }
    GE_ASSERT_SUCCESS(ReorderMaskedFillInput(node));
  }
  return af::SUCCESS;
}

}  // namespace optimize
