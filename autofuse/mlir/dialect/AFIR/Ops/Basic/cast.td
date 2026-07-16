// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Ops/Basic/cast.td - AFIR cast operations ------------*- tablegen -*-===//

#ifndef AFIR_CAST_OPS
#define AFIR_CAST_OPS

include "Dialect.td"
include "Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_CastOp : AFIR_UnaryOpBase<"cast",
    [Pure], [I4, I8, UI8, I16, I32, UI32, I64, UI64, F16, F32, BF16],
    [I4, I8, UI8, I16, UI16, I32, UI32, I64, F16, F32, BF16],
    "Elementwise", "Vector", 1>;

#endif // AFIR_CAST_OPS
