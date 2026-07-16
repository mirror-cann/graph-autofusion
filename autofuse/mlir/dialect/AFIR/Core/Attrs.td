// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Core/Attrs.td - AFIR attribute definitions ----------*- tablegen -*-===//
//
// Attributes for the AFIR dialect. These carry the schedule/tensor semantics
// that the schedule-result bridge needs to preserve (axis, position/mem,
// vectorization, graph-level tiling info).
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ATTRS
#define AFIR_ATTRS

include "Core/Base.td"
include "mlir/IR/AttrTypeBase.td"
include "Core/Enums.td"

//===----------------------------------------------------------------------===//
// Position Attributes
//===----------------------------------------------------------------------===//
def PositionConfigAttr : AFIR_Attr<"PositionConfig", "pos_config"> {
  let summary = "Position configuration including location, depth and buffer mode";

  let parameters = (ins
    "::mlir::afir::Position":$position,
    DefaultValuedParameter<"uint32_t", "0">:$depth,
    DefaultValuedParameter<"bool", "false">:$is_double_buffer
  );

  let assemblyFormat = "`<`"
    "$position"
    "(`,` `depth` `=` $depth^)?"
    "(`,` `is_double_buffer` `=` $is_double_buffer^)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// Tensor attribute groups (dtype/axis/mem layout/vectorization/mem management)
//===----------------------------------------------------------------------===//
def AscTensorGroupsAttr : AFIR_Attr<"AscTensorGroups", "asc_tensor"> {
  let summary = "Ascend tensor attribute groups";
  let description = [{
    Comprehensive tensor attributes including axis information, vectorization
    info, and memory management (tensor_id, reuse_id, position).
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"int64_t">:$vectorized_axis,
    OptionalArrayRefParameter<"Attribute", "array of vectorized stride expressions">:$vectorized_strides,
    "int64_t":$tensor_id,
    DefaultValuedParameter<"int64_t", "-1">:$reuse_id,
    "PositionConfigAttr":$position_config,
    DefaultValuedParameter<"int64_t", "-1">:$position_id
  );

  let assemblyFormat = "`<`"
    "(`vectorized_axis` `=` `[` $vectorized_axis^ `]` `,`"
    " `vectorized_strides` `=` `[` $vectorized_strides `]` `,`)? "
    "`tensor_id` `=` $tensor_id `,`"
    "(`reuse_id` `=` $reuse_id^ `,` )?"
    "`position` `=` $position_config"
    "(`,` `position_id` `=` $position_id^)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// Axis and Loop Attributes (maps from af::Axis / ge.proto.AxisDef)
//===----------------------------------------------------------------------===//

def AxisAttr : AFIR_Attr<"Axis", "axis"> {
  let summary = "Axis definition for computational loops";
  let description = [{
    Defines an axis for loop-based computation including axis type, size
    (symbolic expression string), alignment, binding info, and split source.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$id,
    "StringAttr":$name,
    "::mlir::afir::AxisType":$axis_type,
    DefaultValuedParameter<"bool", "false">:$bind_block,
    "StringAttr":$size,
    OptionalParameter<"StringAttr">:$align,
    OptionalArrayRefParameter<"int64_t">:$from
  );

  let assemblyFormat = "`<`"
    "`id` `=` $id `,`"
    "`name` `=` $name `,`"
    "`axis_type` `=` $axis_type `,`"
    "(`bind_block` `=` $bind_block^ `,`)?"
    "`size` `=` $size"
    "(`,` `align` `=` $align^)?"
    "(`,` `from` `=` `[` $from^ `]`)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// Temporary Buffer Attributes
//===----------------------------------------------------------------------===//

def TmpBufDescAttr : AFIR_Attr<"TmpBufDesc", "tmp_buf_desc"> {
  let summary = "Temporary buffer description";

  let parameters = (ins
    "StringAttr":$size,
    DefaultValuedParameter<"int64_t", "-1">:$life_time_axis_id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Graph Attributes (tiling key + axes + graph type + size vars)
//===----------------------------------------------------------------------===//

def AscGraphAttrGroupsAttr : AFIR_Attr<"AscGraphAttrGroups", "asc_graph"> {
  let summary = "Ascend graph attribute groups";

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$tiling_key,
    ArrayRefParameter<"AxisAttr", "array of Axis">:$axes,
    "::mlir::afir::AscGraphType":$type,
    OptionalArrayRefParameter<"StringAttr", "array of size variable strings">:$size_var
  );

  let assemblyFormat = "`<`"
    "(`tiling_key` `=` $tiling_key^ `,`)?"
    "`axes` `=` `[` $axes `]` `,`"
    "`type` `=` $type"
    "(`,` `size_var` `=` `[` $size_var^ `]`)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// Scalar value attribute (for Data/Output nodes)
//===----------------------------------------------------------------------===//

def ScalarValueAttr : AFIR_Attr<"ScalarValue", "value"> {
  let summary = "representing the value of scalar";

  let parameters = (ins
    "StringAttr":$value
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Array Attributes
//===----------------------------------------------------------------------===//

def TmpBufDescArrayAttr : TypedArrayAttrBase<TmpBufDescAttr, "Array of TmpBufDesc attributes">;

def AscTensorGroupsArrayAttr : TypedArrayAttrBase<AscTensorGroupsAttr, "Array of AscTensorGroups attributes"> {
    let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), {::mlir::afir::AscTensorGroupsAttr::get($_builder.getContext(), ::llvm::ArrayRef<int64_t>{}, ::llvm::ArrayRef<::mlir::Attribute>{}, -1, -1, ::mlir::afir::PositionConfigAttr::get($_builder.getContext(), $0, 0, false), -1)})";
}

#endif // AFIR_ATTRS
