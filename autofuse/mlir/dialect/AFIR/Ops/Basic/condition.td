// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Ops/Basic/condition.td - AFIR condition operations --*- tablegen -*-===//

#ifndef AFIR_CONDITION_OPS
#define AFIR_CONDITION_OPS

include "Dialect.td"
include "Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

class AFIR_TernaryConditionOp<string mnemonic, list<Trait> traits = [], list<Type> conditionTypes, list<Type> types> :
    AFIR_TernaryOpBase<mnemonic, traits, conditionTypes, types, types, types, "Elementwise", "Vector", 1>;

def AFIR_SelectOp : AFIR_TernaryConditionOp<"select", [Pure], [UI8], [I16, I32, I64, F16, F32]>;
def AFIR_WhereOp : AFIR_TernaryConditionOp<"where", [Pure], [UI8], [I16, I32, I64, F16, F32]>;

#endif // AFIR_CONDITION_OPS
