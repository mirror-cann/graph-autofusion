// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Ops/Basic/data.td - AFIR data operations ------------*- tablegen -*-===//

#ifndef AFIR_DATA_OPS
#define AFIR_DATA_OPS

include "Dialect.td"
include "Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_ScalarOp : AFIR_NullaryOpBase<"scalar", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32], "Invalid", "Scalar", 1> {
  let arguments = !con(
    (ins
      AnyAttrOf<[I8Attr, UI8Attr, I16Attr, UI16Attr, I32Attr, UI32Attr, I64Attr, UI64Attr, F16Attr, F32Attr]>:$value
    ),
    commonArgs
  );
}
def AFIR_IndexExprOp : AFIR_NullaryOpBase<"index_expr", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32], "Invalid", "Vector", 1>;
def AFIR_LoadOp : AFIR_UnaryOp<"load", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Load", "MTE2", 1>;
def AFIR_StoreOp : AFIR_UnaryOp<"store", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Store", "MTE3", 1>;

#endif // AFIR_DATA_OPS
