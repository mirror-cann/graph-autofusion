/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __V35_UT_COMMON_H__
#define __V35_UT_COMMON_H__

#include <cstring>

#include "runtime_stub.h"
#include "platform_context.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"

using namespace ge;
using namespace af;

namespace {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<af::ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<af::ComputeGraph>(name);
    node_type_ = node_type;
  }

  af::NodePtr AddNode(const std::string &name, const std::string &type, const int in_cnt, const int out_cnt,
                      const std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<af::GeTensorDesc>();
    tensor_desc->SetShape(af::GeShape(std::move(shape)));
    tensor_desc->SetFormat(ge::FORMAT_NCHW);
    tensor_desc->SetDataType(ge::DT_FLOAT);

    auto op_desc = std::make_shared<af::OpDesc>(name, (node_type_ == "") ? type : "AscGraph");
    for (std::int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (std::int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](af::Operator &op) { return ge::GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  void AddDataEdge(const af::NodePtr &src_node, const std::int32_t src_idx, const af::NodePtr &dst_node,
                   const std::int32_t dst_idx) {
    af::GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  af::ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  af::ComputeGraphPtr graph_;
  std::string node_type_;
};
}  // namespace

namespace ge {
class RuntimeStubV2 : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    (void)strcpy_s(version, maxLen, "Ascend910_9591");
    return RT_ERROR_NONE;
  }

  rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) override {
    return CopyRuntimeSocSpecValue(label, key, val, maxLen, RuntimeSocSpecDefaults{"3510", "0"});
  }
};

// 公共函数：设置运行时存根
inline void SetupRuntimeStub() {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2>();
  ge::RuntimeStub::SetInstance(stub_v2);
}
}  // namespace ge

#endif
