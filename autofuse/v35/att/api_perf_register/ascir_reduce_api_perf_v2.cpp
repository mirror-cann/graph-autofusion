/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "v35/att/api_perf_register/ascir_reduce_api_perf_v2.h"

#include <memory>
#include "codegen_api_param/codegen_api_param.h"
#include "v35/att/api_perf_register/ascendc_api_perf/reduce_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"

namespace att {
namespace ascir_reduce_v2 {
namespace {
void SetReduceDims(const Expr &first, const Expr &last, ascendcapi_v2::ReducePattern pattern,
                   NodeDetail &node_detail) {
  node_detail.input_dims = {first, last};
  node_detail.output_dims = {pattern == ascendcapi_v2::ReducePattern::kRA ? last : first};
}

void MergeReduceDimsByLastAxis(ascendcapi_v2::ReducePattern pattern, NodeDetail &node_detail) {
  auto &in_dims = node_detail.input_dims;
  if (in_dims.size() <= 2U) {
    return;
  }
  Expr first = CreateExpr(1);
  for (size_t i = 0U; i < in_dims.size() - 1U; ++i) {
    first = first * in_dims[i];
  }
  GELOGD("[ATT Reduce] fallback merged dims by last axis: node[%s], pattern[%s], first[%s], last[%s].",
         node_detail.name.c_str(), pattern == ascendcapi_v2::ReducePattern::kAR ? "AR" : "RA", first.Str().get(),
         in_dims.back().Str().get());
  SetReduceDims(first, in_dims.back(), pattern, node_detail);
}

ge::Status TryGetReuseSource(const NodeInfo &node, bool &is_reuse_source) {
  if (node.reduce_specific_params.reuse.valid) {
    is_reuse_source = node.reduce_specific_params.reuse.is_reuse_source;
    return ge::SUCCESS;
  }
  GELOGW("[ATT Reduce] reuse-source fallback by input fanout, node[%s].", node.name.c_str());
  GE_ASSERT_NOTNULL(node.node_ptr, "Reduce reuse-source: node_ptr is null, node[%s].", node.name.c_str());
  auto node_in_anchor = node.node_ptr->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(node_in_anchor, "Reduce reuse-source: in_anchor is null, node[%s].", node.name.c_str());
  auto peer_out_anchor = node_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor, "Reduce reuse-source: peer_out_anchor is null, node[%s].", node.name.c_str());
  const auto &in_node = std::dynamic_pointer_cast<ge::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(in_node, "Reduce reuse-source: in_node is null, node[%s].", node.name.c_str());
  is_reuse_source = in_node->GetOutAllNodes().size() == 1UL;
  GELOGD("[ATT Reduce] reuse-source: is_reuse[%d], node[%s].", static_cast<int32_t>(is_reuse_source),
         node.name.c_str());
  return ge::SUCCESS;
}

ge::Status ConvertReducePattern(codegen::ReducePattern pattern, ascendcapi_v2::ReducePattern &perf_pattern) {
  switch (pattern) {
    case codegen::ReducePattern::kAR:
      perf_pattern = ascendcapi_v2::ReducePattern::kAR;
      return ge::SUCCESS;
    case codegen::ReducePattern::kRA:
      perf_pattern = ascendcapi_v2::ReducePattern::kRA;
      return ge::SUCCESS;
    default:
      GE_ASSERT_TRUE(false, "Reduce pattern is unknown.");
  }
}

ge::Status ConvertReduceMergeMode(codegen::ReduceMergeMode mode, ascendcapi_v2::ReduceMergeMode &perf_mode) {
  switch (mode) {
    case codegen::ReduceMergeMode::kNone:
      perf_mode = ascendcapi_v2::ReduceMergeMode::kNone;
      return ge::SUCCESS;
    case codegen::ReduceMergeMode::kCopy:
      perf_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
      return ge::SUCCESS;
    case codegen::ReduceMergeMode::kMergeByElementwise:
      perf_mode = ascendcapi_v2::ReduceMergeMode::kMergeByElementwise;
      return ge::SUCCESS;
    default:
      GE_ASSERT_TRUE(false, "Reduce merge mode is unknown.");
  }
}

bool IsMultiReduceMode(ascendcapi_v2::ReduceMergeMode mode) {
  return mode != ascendcapi_v2::ReduceMergeMode::kNone;
}

ge::Status BuildCurrentShapeParams(const std::vector<TensorShapeInfo> &input_shapes,
                                   const std::vector<TensorShapeInfo> &output_shapes,
                                   const codegen::ReduceSpecificParams &base_params,
                                   ascendcapi_v2::ReduceMergeMode merge_mode,
                                   codegen::ReduceSpecificParams &shape_params) {
  codegen::ReduceSpecificParamBuildInput input;
  input.reduce_type = base_params.reduce_type;
  input.input_repeats = input_shapes[0].repeats;
  input.input_strides = input_shapes[0].strides;
  input.output_dims = output_shapes[0].dims;
  input.output_strides = output_shapes[0].strides;
  input.dtype_size = input_shapes[0].data_type_size;
  input.pattern = base_params.pattern;
  input.need_multi_reduce = IsMultiReduceMode(merge_mode);
  input.merge_times = base_params.merge_times.IsValid() ? base_params.merge_times : CreateExpr(1);
  input.reuse = base_params.reuse;
  return codegen::BuildReduceSpecificParams(input, shape_params);
}

ge::Status BuildReduceContext(const std::vector<TensorShapeInfo> &input_shapes,
                              const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                              ascendcapi_v2::ReduceApiPerfContext &context) {
  const auto &params = node.reduce_specific_params;
  GE_ASSERT_TRUE(params.valid, "Reduce specific params is invalid, node[%s].", node.name.c_str());
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, context.node_detail));
  GE_ASSERT_TRUE(!context.node_detail.input_dims.empty(), "Reduce input dims is empty, node[%s].", node.name.c_str());
  GE_ASSERT_SUCCESS(ConvertReducePattern(params.pattern, context.pattern));
  GE_ASSERT_SUCCESS(ConvertReduceMergeMode(params.merge_mode, context.merge_mode));
  codegen::ReduceSpecificParams current_shape_params;
  GE_ASSERT_SUCCESS(BuildCurrentShapeParams(input_shapes, output_shapes, params, context.merge_mode,
                                            current_shape_params));
  if (current_shape_params.merged_dims.valid) {
    SetReduceDims(current_shape_params.merged_dims.first, current_shape_params.merged_dims.last, context.pattern,
                  context.node_detail);
  } else {
    MergeReduceDimsByLastAxis(context.pattern, context.node_detail);
  }
  GE_ASSERT_TRUE(context.node_detail.input_dims.size() == 2U, "Reduce dims must be {first,last}, node[%s].",
                 node.name.c_str());
  GE_ASSERT_SUCCESS(TryGetReuseSource(node, context.is_reuse_source),
                    "Reduce reuse-source branch is unknown, node[%s].", node.name.c_str());
  context.merge_size = current_shape_params.merge_size;
  context.merge_times = params.merge_times.IsValid() ? params.merge_times : CreateExpr(1);
  GELOGD(
      "[ATT Reduce] BuildReduceContext: node[%s], pattern[%s], merge_mode[%d], reuse[%d], "
      "merge_size[%s], merge_times[%s].",
      node.name.c_str(), context.pattern == ascendcapi_v2::ReducePattern::kAR ? "AR" : "RA",
      static_cast<int32_t>(context.merge_mode), static_cast<int32_t>(context.is_reuse_source),
      context.merge_size.Str().get(), context.merge_times.Str().get());
  return ge::SUCCESS;
}

bool HasReduceSpecificParams(const NodeInfo &node) {
  return node.reduce_specific_params.valid;
}
}  // namespace

ge::Status ElementwiseMaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MaxPerf(node_info, perf_res));
  return ge::SUCCESS;
}

ge::Status ElementwiseMinApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MinPerf(node_info, perf_res));
  return ge::SUCCESS;
}

ge::Status ElementwiseSumApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SumPerf(node_info, perf_res));
  return ge::SUCCESS;
}

ge::Status ElementwiseMeanApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MeanPerf(node_info, perf_res));
  return ge::SUCCESS;
}

ge::Status ElementwiseProdApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MulPerf(node_info, perf_res));
  return ge::SUCCESS;
}

ge::Status ReduceMaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceMaxPerf(context, perf_res));
  return ge::SUCCESS;
}

ge::Status ReduceMinApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceMinPerf(context, perf_res));
  return ge::SUCCESS;
}

ge::Status ReduceAnyApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceAnyPerf(context, perf_res));
  return ge::SUCCESS;
}

ge::Status ReduceAllApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceAllPerf(context, perf_res));
  return ge::SUCCESS;
}

ge::Status ReduceSumApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceSumPerf(context, perf_res));
  return ge::SUCCESS;
}

ge::Status ReduceMeanApi(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                         PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceMeanPerf(context, perf_res));
  return ge::SUCCESS;
}

ge::Status ReduceProdApi(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                         PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceProdPerf(context, perf_res));
  return ge::SUCCESS;
}

ge::Status MaxApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceMaxApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMaxApi(input_shapes, output_shapes, node, perf_res);
}

ge::Status MinApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceMinApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMinApi(input_shapes, output_shapes, node, perf_res);
}

ge::Status AnyApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceAnyApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMaxApi(input_shapes, output_shapes, node, perf_res);
}

ge::Status AllApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceAllApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMinApi(input_shapes, output_shapes, node, perf_res);
}

ge::Status SumApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceSumApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseSumApi(input_shapes, output_shapes, node, perf_res);
}

ge::Status MeanApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                   const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceMeanApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMeanApi(input_shapes, output_shapes, node, perf_res);
}

ge::Status ProdApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                   const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceProdApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseProdApi(input_shapes, output_shapes, node, perf_res);
}
}  // namespace ascir_reduce_v2
}  // namespace att
