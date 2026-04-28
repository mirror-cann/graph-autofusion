/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AF_INC_GRAPH_UTILS_OP_DESC_UTILS_H_
#define AF_INC_GRAPH_UTILS_OP_DESC_UTILS_H_

#include <string>
#include <vector>
#include "graph/def_types.h"
#include "graph/node.h"
#include "graph/operator.h"
#include "graph/runtime_inference_context.h"

/*lint -e148*/
namespace af {
using ConstGeTensorBarePtr = const GeTensor *;
class OpDescUtils {
 public:
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<OpDesc>>;
  using GetConstInputOnRuntimeFun =
      std::function<graphStatus(const ConstNodePtr &node, const size_t index, GeTensorPtr &tensor)>;

  OpDescUtils() = default;
  ~OpDescUtils() = default;
  static bool HasQuantizeFactorParams(const OpDescPtr& op_desc);
  static bool HasQuantizeFactorParams(const OpDesc& op_desc);
  static std::vector<NodePtr> GetConstInputNode(const Node& node);
  static std::vector<NodeToOutAnchor> GetConstInputNodeAndAnchor(const Node &node);
  static std::vector<ConstGeTensorPtr> GetInputData(const std::vector<NodePtr>& input_nodes);
  static std::vector<ConstGeTensorPtr> GetWeightsFromNodes(
      const std::vector<NodeToOutAnchor>& input_nodes_2_out_anchors);

  static std::vector<ConstGeTensorPtr> GetWeights(const Node& node);
  static std::vector<ConstGeTensorPtr> GetWeights(const ConstNodePtr& node);
  static std::vector<GeTensorPtr> MutableWeights(const Node& node);
  static std::vector<GeTensorPtr> MutableWeights(const NodePtr node);
  static graphStatus SetWeights(Node &node, const std::vector<af::GeTensorPtr>& weights);
  static graphStatus SetWeights(NodePtr node, const std::vector<af::GeTensorPtr> &weights);
  static graphStatus SetWeights(Node &node, const std::map<int, af::GeTensorPtr> &weights_map);
  static graphStatus ClearWeights(const NodePtr node);
  static graphStatus SetNoneConstNodeWeights(Node &node, const std::map<int, af::GeTensorPtr> &weights_map);
  static graphStatus SetNoneConstNodeWeights(Node &node, const std::vector<af::GeTensorPtr> &weights);
  static bool ClearInputDesc(const OpDescPtr op_desc, const uint32_t index);
  static bool ClearInputDesc(const NodePtr& node);
  static bool ClearOutputDesc(const OpDescPtr& op_desc, const uint32_t index);
  static bool ClearOutputDesc(const NodePtr& node);
  static std::vector<NodePtr> GetConstInputs(const Node& node, const uint32_t depth = 64U);
  static std::vector<NodePtr> GetConstInputs(const ConstNodePtr& node);
  static size_t GetNonConstInputsSize(const Node& node);
  static size_t GetNonConstInputsSize(const ConstNodePtr node);
  // Index: Indicates the index of all non const inputs
  static GeTensorDesc GetNonConstInputTensorDesc(const Node& node, const size_t index_non_const = 0UL);
  static GeTensorDesc GetNonConstInputTensorDesc(const ConstNodePtr& node, const size_t index_non_const = 0UL);
  static bool GetNonConstInputIndex(const Node& node, const size_t index_non_const, size_t& index);
  static bool GetNonConstInputIndex(const ConstNodePtr& node, const size_t index_non_const, size_t& index);
  // Index: Indicates the index of all inputs
  static bool IsNonConstInput(const Node& node, const size_t index = 0UL);
  static bool IsNonConstInput(const ConstNodePtr& node, const size_t index = 0UL);

  static std::vector<af::GeTensorDesc> GetNonConstTensorDesc(const ConstNodePtr& node);
  static graphStatus AddConstOpToAnchor(const InDataAnchorPtr in_anchor, const GeTensorPtr& tensor_ptr);

  static Operator CreateOperatorFromOpDesc(OpDescPtr op_desc);
  static Operator CreateOperatorFromNode(ConstNodePtr node_ptr);
  static OpDescPtr GetOpDescFromOperator(const Operator& oprt);
  static graphStatus CopyOperatorLinks(const std::map<std::string, Operator> &src_op_list,
                                       std::map<std::string, Operator> &dst_op_list);
  static graphStatus CopyOperators(const ComputeGraphPtr &dst_compute_graph,
                                   const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                   const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                   const std::map<std::string, Operator> &src_op_list,
                                   std::map<std::string, Operator> &dst_op_list);
  static OpDescPtr CloneOpDesc(const ConstOpDescPtr &org_op_desc);
  __attribute__((weak)) static OpDescPtr CopyOpDesc(const ConstOpDescPtr &org_op_desc);
  __attribute__((weak)) static OpDescPtr CreateConstOp(const GeTensorPtr& tensor_ptr);
  static OpDescPtr CreateConstOp(const GeTensorPtr& tensor_ptr, const bool copy);
  static OpDescPtr CreateConstOpZeroCopy(const GeTensorPtr& tensor_ptr);

  static graphStatus SetSubgraphInstanceName(const std::string &subgraph_name,
      const std::string &subgraph_instance_name, OpDescPtr &op_desc);
  static ConstGeTensorBarePtr GetInputConstData(const Operator &op, const uint32_t idx);
  // deprecated
  static void SetRuntimeContextToOperator(const Operator &op, RuntimeInferenceContext *const context);
  static void SetCallbackGetConstInputFuncToOperator(const Operator &op,
                                                     GetConstInputOnRuntimeFun get_const_input_func);
  static bool HasCallbackGetConstInputFunc(const Operator &op);
  static std::map<size_t, std::pair<size_t, size_t>> GetInputIrIndexes2InstanceIndexesPairMap(const OpDescPtr &op_desc);
  static std::map<size_t, std::pair<size_t, size_t>> GetOutputIrIndexes2InstanceIndexesPairMap(
      const OpDescPtr &op_desc);

  static graphStatus GetIrInputInstanceDescRange(const OpDescPtr &op,
                                                 std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range);

  __attribute__((weak)) static graphStatus GetIrInputRawDescRange(const OpDescPtr &op,
      std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range);

  static graphStatus GetIrOutputDescRange(const OpDescPtr &op,
                                          std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range);

  static af::graphStatus GetInputIrIndexByInstanceIndex(const OpDescPtr &op_desc,
                                                        size_t instance_index, size_t &ir_index);
  static af::graphStatus GetOutputIrIndexByInstanceIndex(const OpDescPtr &op_desc,
                                                         size_t instance_index, size_t &ir_index);
  static af::graphStatus GetInstanceNum(const OpDescPtr &op_desc, size_t ir_index, size_t start_index,
                                        const std::map<uint32_t, std::string> &valid_index_2_names,
                                        size_t &instance_num);
  static af::graphStatus GetPromoteIrInputList(const OpDescPtr &op_desc,
                                               std::vector<std::vector<size_t>> &promote_index_list);
  static af::graphStatus GetPromoteInstanceInputList(const OpDescPtr &op_desc,
                                                     std::vector<std::vector<size_t>> &promote_index_list);
  static af::graphStatus GetIrInputDtypeSymIds(const OpDescPtr &op_desc, std::vector<std::string> &dtype_sym_ids);
  static af::graphStatus GetIrOutputDtypeSymIds(const OpDescPtr &op_desc, std::vector<std::string> &dtype_sym_ids);
 private:
  static GeTensorPtr MutableWeights(OpDesc& op_desc);
  static GeTensorPtr MutableWeights(const OpDescPtr op_desc);
  static graphStatus SetWeights(OpDesc& op_desc, const GeTensorPtr weight);
  static graphStatus SetWeights(OpDescPtr op_desc, const GeTensorPtr weight);
};
}  // namespace af
/*lint +e148*/
#endif  // INC_GRAPH_UTILS_OP_DESC_UTILS_H_
