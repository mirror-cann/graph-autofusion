/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_GRAPH_SERIALIZATION_NAMED_ATTRS_SERIALIZER_H_
#define METADEF_GRAPH_SERIALIZATION_NAMED_ATTRS_SERIALIZER_H_

#include "attr_serializer.h"
#include "attr_serializer_registry.h"
#include "proto/af_ir.pb.h"
#include "graph/ge_attr_value.h"

namespace af {
class NamedAttrsSerializer : public GeIrAttrSerializer {
 public:
  NamedAttrsSerializer() = default;
  graphStatus Serialize(const AnyValue &av, GeIrAttrDef &def) override;
  graphStatus Deserialize(const GeIrAttrDef &def, AnyValue &av) override;

  graphStatus Serialize(const af::NamedAttrs &named_attr, proto::NamedAttrs *proto_attr) const;
  graphStatus Deserialize(const proto::NamedAttrs &proto_attr, af::NamedAttrs &named_attrs) const;
};
}  // namespace ge

#endif // METADEF_GRAPH_SERIALIZATION_NAMED_ATTRS_SERIALIZER_H_
