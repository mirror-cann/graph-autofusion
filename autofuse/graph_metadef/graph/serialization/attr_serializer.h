/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AF_METADEF_CXX_ATTR_SERIALIZER_H
#define AF_METADEF_CXX_ATTR_SERIALIZER_H

#ifdef AUTOFUSE_USE_GE_METADEF
#include "proto/ge_ir.pb.h"
#else
#include "proto/af_ir.pb.h"
#endif

#include "graph/any_value.h"

namespace af {
using ge::AnyValue;
using ge::graphStatus;
using ge::GetTypeId;
#ifdef AUTOFUSE_USE_GE_METADEF
using GeIrAttrDef = ::ge::proto::AttrDef;
using GeIrAttrSerializer = ge::GeIrAttrSerializer;
#else
using GeIrAttrDef = proto::AttrDef;
class GeIrAttrSerializer {
 public:
  virtual graphStatus Serialize(const AnyValue &av, GeIrAttrDef &def) = 0;
  virtual graphStatus Deserialize(const GeIrAttrDef &def, AnyValue &av) = 0;
  virtual ~GeIrAttrSerializer() = default;
  GeIrAttrSerializer() = default;
  GeIrAttrSerializer(const GeIrAttrSerializer &) = delete;
  GeIrAttrSerializer &operator=(const GeIrAttrSerializer &) = delete;
  GeIrAttrSerializer(GeIrAttrSerializer &&) = delete;
  GeIrAttrSerializer &operator=(GeIrAttrSerializer &&) = delete;
};
#endif
}  // namespace af

#endif  // AF_METADEF_CXX_ATTR_SERIALIZER_H
