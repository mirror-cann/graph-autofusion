// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Base.td - AFIR operation base classes ---------------*- tablegen -*-===//
//
// Op base classes for the AFIR dialect (Nary/Nullary/Unary/Binary/Ternary/
// Variadic). Ported faithfully from Ascend-MLIR-gsr. Type-inference traits are
// added per-op-family in the op .td files (e.g. math.td), not here.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_BASE
#define AFIR_BASE

include "Dialect.td"
include "Core/Attrs.td"

//===----------------------------------------------------------------------===//
// Common per-op attributes
//===----------------------------------------------------------------------===//
class AFIR_CommonAttrs {
  dag commonArgs = (ins
    OptionalAttr<AffineMapArrayAttr>:$indexing_maps,
    DefaultValuedOptionalAttr<I32Attr, "-1">:$loop_axis,
    DefaultValuedOptionalAttr<DictionaryAttr, "{}">:$ir_attr_def,
    OptionalAttr<TmpBufDescArrayAttr>:$tmp_buffers,
    DefaultValuedAttr<AscTensorGroupsArrayAttr, "::mlir::afir::Position::GM">:$outputs
  );
}

//===----------------------------------------------------------------------===//
// Base AFIR Operation
//===----------------------------------------------------------------------===//
class AFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AFIR_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Base op carrying compute info (ComputeType / ComputeUnit)
//===----------------------------------------------------------------------===//
class AFIR_BaseOp<string mnemonic,
                          list<Trait> traits,
                          string computeType = "Invalid",
                          string computeUnit = "Invalid"> :
    AFIR_Op<mnemonic, traits> {
  code getComputeInfoDecl = [{
  static afir::ComputeType getComputeType();
  static afir::ComputeUnit getComputeUnit();
  }];
  code getComputeInfo = [{
    afir::ComputeType $cppClass::getComputeType() {
      return afir::ComputeType::}] # computeType # [{;
    }
    afir::ComputeUnit $cppClass::getComputeUnit() {
      return afir::ComputeUnit::}] # computeUnit # [{;
    }
  }];
  let extraClassDeclaration = getComputeInfoDecl;
  let extraClassDefinition = getComputeInfo;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// N-ary op base
//===----------------------------------------------------------------------===//
class AFIR_NaryOp<string mnemonic,
                          list<Trait> traits,
                          int numOperands,
                          int numResults = 1,
                          string computeType = "Invalid",
                          string computeUnit = "Invalid",
                          bit finalize = 0> :
    AFIR_BaseOp<mnemonic, traits, computeType, computeUnit> {
  code baseMethodDecl = [{
  int getNumberOfOperands();
  int getNumberOfResults();
  }] # getComputeInfoDecl;
  code baseMethods = [{
      int $cppClass::getNumberOfOperands() { return }] # numOperands # [{; }
      int $cppClass::getNumberOfResults() { return }] # numResults # [{; }
    }] # getComputeInfo;
  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethodDecl, "");
  let extraClassDefinition = !if(finalize, baseMethods, "");
}

//===----------------------------------------------------------------------===//
// Nullary op base (0 inputs, 1 result)
//===----------------------------------------------------------------------===//
class AFIR_NullaryOpBase<string mnemonic,
                       list<Trait> traits,
                       list<Type> types,
                       string computeType = "Invalid",
                       string computeUnit = "Invalid",
                       bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 0, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = commonArgs;
  let results = (outs TensorOf<types>:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Unary op base (1 input, 1 result)
//===----------------------------------------------------------------------===//
class AFIR_UnaryOpBase<string mnemonic,
                       list<Trait> traits,
                       list<Type> inputTypes,
                       list<Type> resultTypes,
                       string computeType = "Invalid",
                       string computeUnit = "Invalid",
                       bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 1, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins TensorOf<inputTypes>:$input),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

class AFIR_UnaryOp<string mnemonic,
                    list<Trait> traits = [],
                    list<Type> types,
                    string computeType,
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_UnaryOpBase<mnemonic, traits, types, types, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// Binary op base (2 inputs, 1 result)
//===----------------------------------------------------------------------===//
class AFIR_BinaryOpBase<string mnemonic,
                        list<Trait> traits,
                        list<Type> lhsTypes,
                        list<Type> rhsTypes,
                        list<Type> resultTypes,
                        string computeType = "Invalid",
                        string computeUnit = "Invalid",
                        bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 2, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins
      TensorOf<lhsTypes>:$lhs,
      TensorOf<rhsTypes>:$rhs
    ),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let hasVerifier = 1;
}

class AFIR_BinaryOp<string mnemonic,
                    list<Trait> traits = [],
                    list<Type> types,
                    string computeType,
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_BinaryOpBase<mnemonic, traits, types, types, types, computeType, computeUnit, finalize>;

class AFIR_BinaryHomoOp<string mnemonic,
                        list<Trait> traits = [],
                        list<Type> inputTypes,
                        list<Type> resultTypes,
                        string computeType,
                        string computeUnit,
                        bit finalize = 0> :
    AFIR_BinaryOpBase<mnemonic, traits, inputTypes, inputTypes, resultTypes, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// Ternary op base (3 inputs, 1 result)
//===----------------------------------------------------------------------===//
class AFIR_TernaryOpBase<string mnemonic,
                         list<Trait> traits,
                         list<Type> input1Types,
                         list<Type> input2Types,
                         list<Type> input3Types,
                         list<Type> resultTypes,
                         string computeType = "Invalid",
                         string computeUnit = "Invalid",
                         bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 3, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins
      TensorOf<input1Types>:$input1,
      TensorOf<input2Types>:$input2,
      TensorOf<input3Types>:$input3
    ),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $input1 `,` $input2 `,` $input3 attr-dict `:`
    `(` type($input1) `,` type($input2) `,` type($input3) `)` `->` type($result)
  }];
  let hasVerifier = 1;
}

class AFIR_TernaryOp<string mnemonic,
                    list<Trait> traits = [],
                    list<Type> types,
                    string computeType,
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_TernaryOpBase<mnemonic, traits, types, types, types, types, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// Variadic op base (N inputs, 1 result)
//===----------------------------------------------------------------------===//
class AFIR_VariadicOpBase<string mnemonic,
                         list<Trait> traits,
                         list<Type> inputTypes,
                         list<Type> resultTypes,
                         string computeType = "Invalid",
                         string computeUnit = "Invalid",
                         bit finalize = 0> :
    AFIR_BaseOp<mnemonic, traits, computeType, computeUnit>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins Variadic<TensorOf<inputTypes>>:$inputs),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs) `->` type($result)
  }];
  code baseMethodDecl = [{
  int getNumberOfOperands();
  int getNumberOfResults();
  }] # getComputeInfoDecl;
  code baseMethods = [{
      int $cppClass::getNumberOfOperands() { return getInputs().size(); }
      int $cppClass::getNumberOfResults() { return 1; }
    }] # getComputeInfo;

  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethodDecl, "");
  let extraClassDefinition = !if(finalize, baseMethods, "");
}

class AFIR_VariadicOp<string mnemonic,
                      list<Trait> traits = [],
                      list<Type> types,
                      string computeType,
                      string computeUnit,
                      bit finalize = 0> :
    AFIR_VariadicOpBase<mnemonic, traits, types, types, computeType, computeUnit, finalize>;

#endif // AFIR_BASE
