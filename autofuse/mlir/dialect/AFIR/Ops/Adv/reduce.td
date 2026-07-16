// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Ops/Adv/reduce.td - AFIR reduce operations ----------*- tablegen -*-===//

#ifndef AFIR_REDUCE_OPS
#define AFIR_REDUCE_OPS

include "Dialect.td"
include "Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"

class AFIR_UnaryReduceOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_UnaryOpBase<mnemonic, traits # [InferTensorTypeAdaptor], types, types, "Reduce", "Vector", 1> {
  let arguments = !con(
    (ins
      TensorOf<types>:$input,
      I32Attr:$axis
    ),
    commonArgs
  );
  let extraClassDeclaration = baseMethodDecl # [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
  let extraClassDefinition = baseMethods # [{
    bool $cppClass::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      if (l.size() != r.size() || l.size() != 1) return false;
      return succeeded(verifyCompatibleShape(l[0], r[0]));
    }
  }];
}

def AFIR_MaxOp : AFIR_UnaryReduceOp<"max", [Pure], [F16, F32]>;
def AFIR_MinOp : AFIR_UnaryReduceOp<"min", [Pure], [F16, F32]>;
def AFIR_SumOp : AFIR_UnaryReduceOp<"sum", [Pure], [I32, F32]>;
def AFIR_MeanOp : AFIR_UnaryReduceOp<"mean", [Pure], [F32]>;
def AFIR_ProdOp : AFIR_UnaryReduceOp<"prod", [Pure], [F32]>;
def AFIR_AnyOp : AFIR_UnaryReduceOp<"any", [Pure], [F32]>;
def AFIR_AllOp : AFIR_UnaryReduceOp<"all", [Pure], [F32]>;

#endif // AFIR_REDUCE_OPS
