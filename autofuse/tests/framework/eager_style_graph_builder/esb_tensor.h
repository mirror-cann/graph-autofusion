/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_TENSOR_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_TENSOR_H_
#include "graph/node.h"
#include "graph/symbolizer/symbolic.h"
#include "attribute_group/attr_group_symbolic_desc.h"
class EsbGraph;
class EsbTensor {
 public:
  // 注意，这是个内部类，调用着需要保证传入的`producer`不为空,`index`合法
  EsbTensor(EsbGraph &owner, af::NodePtr producer, int32_t index)
      : owner_graph_(owner), producer_(std::move(producer)), producer_out_index_(index) {}

  af::Status SetOriginShape(const af::GeShape &shape) {
    auto td = GetTd();
    GE_ASSERT_NOTNULL(td);
    td->SetOriginShape(shape);
    return af::SUCCESS;
  }

  af::Status SetStorageShape(const af::GeShape &shape) {
    auto td = GetTd();
    GE_ASSERT_NOTNULL(td);
    td->SetShape(shape);
    return af::SUCCESS;
  }

  af::Status SetShape(const af::GeShape &shape) {
    GE_ASSERT_SUCCESS(SetOriginShape(shape));
    GE_ASSERT_SUCCESS(SetStorageShape(shape));
    return af::SUCCESS;
  }

  af::Status SetOriginSymbolShape(const std::vector<af::Expression> &shape) {
    const auto td = GetTd();
    GE_ASSERT_NOTNULL(td);
    const auto attr = td->GetOrCreateAttrsGroup<af::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(attr);
    attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims() = shape;
    return af::SUCCESS;
  }
  af::Status SetInputOriginSymbolShape(const std::vector<af::Expression> &shape) {
    const auto input_td = GetInputTd();
    GE_ASSERT_NOTNULL(input_td);
    const auto input_attr = input_td->GetOrCreateAttrsGroup<af::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(input_attr);
    input_attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims() = shape;
    return af::SUCCESS;
  }
  af::Status SetSymbolShape(const std::vector<af::Expression> &shape) {
    GE_ASSERT_SUCCESS(SetOriginSymbolShape(shape));
    return af::SUCCESS;
  }
  af::Status SetInputSymbolShape(const std::vector<af::Expression> &shape) {
    GE_ASSERT_SUCCESS(SetInputOriginSymbolShape(shape));
    return af::SUCCESS;
  }
  af::Status SetSymbolShape(const char *const *shape_str, const int64_t shape_str_num) {
    std::vector<af::Expression> shape;
    shape.reserve(shape_str_num);
    for (int64_t i = 0; i < shape_str_num; ++i) {
      shape.emplace_back(af::Expression::Parse(shape_str[i]));
    }
    return SetSymbolShape(shape);
  }
  af::Status SetInputSymbolShape(const char *const *shape_str, const int64_t shape_str_num) {
    std::vector<af::Expression> shape;
    shape.reserve(shape_str_num);
    for (int64_t i = 0; i < shape_str_num; ++i) {
      shape.emplace_back(af::Expression::Parse(shape_str[i]));
    }
    return SetInputSymbolShape(shape);
  }
  af::NodePtr GetProducer() const {
    return producer_;
  }

  EsbGraph &GetOwner() {
    return owner_graph_;
  }
  af::OutDataAnchorPtr GetAnchor() {
    return producer_->GetOutDataAnchor(producer_out_index_);
  }

  EsbGraph &GetOwnerGraph() const {
    return owner_graph_;
  }

 private:
  af::GeTensorDescPtr GetTd() {
    return producer_->GetOpDesc()->MutableOutputDesc(producer_out_index_);
  }
  af::GeTensorDescPtr GetInputTd() {
    return producer_->GetOpDesc()->MutableInputDesc(producer_out_index_);
  }

 private:
  EsbGraph &owner_graph_;
  af::NodePtr producer_;
  int32_t producer_out_index_;
};

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_TENSOR_H_
