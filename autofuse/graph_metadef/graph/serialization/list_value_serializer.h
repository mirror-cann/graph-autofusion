/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_GRAPH_SERIALIZATION_LIST_VALUE_SERIALIZER_H_
#define METADEF_GRAPH_SERIALIZATION_LIST_VALUE_SERIALIZER_H_

#include "attr_serializer.h"
#include "attr_serializer_registry.h"
#include "proto/af_ir.pb.h"
#include "graph/ge_attr_value.h"

namespace af {
class ListValueSerializer : public GeIrAttrSerializer {
 public:
  ListValueSerializer() = default;
  graphStatus Serialize(const AnyValue &av, GeIrAttrDef &def) override;
  graphStatus Deserialize(const GeIrAttrDef &def, AnyValue &av) override;

 private:
  static graphStatus SerializeListInt(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListString(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListFloat(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListBool(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListGeTensorDesc(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListGeTensor(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListBuffer(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListGraphDef(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListNamedAttrs(const AnyValue &av, GeIrAttrDef &def);
  static graphStatus SerializeListDataType(const AnyValue &av, GeIrAttrDef &def);

  static graphStatus DeserializeListInt(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListString(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListFloat(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListBool(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListGeTensorDesc(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListGeTensor(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListBuffer(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListGraphDef(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListNamedAttrs(const GeIrAttrDef &def, AnyValue &av);
  static graphStatus DeserializeListDataType(const GeIrAttrDef &def, AnyValue &av);
};
}  // namespace ge

#endif // METADEF_GRAPH_SERIALIZATION_LIST_VALUE_SERIALIZER_H_
