/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_ASC_GRAPH_UTILS_H
#define METADEF_CXX_ASC_GRAPH_UTILS_H

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/serialization/attr_serializer.h"
#include "proto/af_ascendc_ir.pb.h"

namespace af {
using ge::graphStatus;
class AscGraphUtils {
 public:
  static ComputeGraphPtr GetComputeGraph(const AscGraph &asc_graph);
  static Status FromComputeGraph(const ComputeGraphPtr &compute_graph, AscGraph &graph);
  /**
 * @param compute_graph的node对象是Node类型时候，接口内部转换为AscNode
 * @return
 */
  static graphStatus ConvertComputeGraphToAscGraph(const ComputeGraphPtr &compute_graph, AscGraph &asc_graph);
  static graphStatus SerializeToBinary(const AscGraph &asc_graph, std::string &output);
  static graphStatus SerializeToReadable(const AscGraph &asc_graph, std::string &output);
  static graphStatus SerializeToProto(const AscGraph &asc_graph, af_ascendc_ir::proto::AscGraphDef &asc_graph_def);
  static graphStatus DeserializeFromBinary(const std::string &to_be_deserialized, AscGraph &out_asc_graph);
  static graphStatus DeserializeFromReadable(const std::string &to_be_deserialized, AscGraph &out_asc_graph);
  static graphStatus DeserializeFromProto(const af_ascendc_ir::proto::AscGraphDef &asc_graph_def, AscGraph &asc_graph);
};
class AscNodeSerializeUtils {
 public:
  static graphStatus SerializeIrDef(const AscNode &node, af_ascendc_ir::proto::IrDef &ir_def);
  static graphStatus SerializeAttrGroupsDef(const AscNode &node,
                                            af::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def);
};

class AscNodeDeserializeUtils {
 public:
  static graphStatus DeserializeIrDef(const af_ascendc_ir::proto::IrDef &ir_def, AscNode &node);
  static graphStatus DeserializeAttrGroupsDef(const af::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def,
                                              AscNode &node);
};
class ExpressionSerializer : public GeIrAttrSerializer {
 public:
  ExpressionSerializer() = default;
  graphStatus Serialize(const AnyValue &av, GeIrAttrDef &def) override;
  graphStatus Deserialize(const GeIrAttrDef &def, AnyValue &av) override;
};
}  // namespace af

#endif  // METADEF_CXX_ASC_GRAPH_UTILS_H
