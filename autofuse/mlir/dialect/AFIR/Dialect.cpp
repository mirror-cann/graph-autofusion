/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Dialect.cpp - AFIR dialect implementation -----------------*- C++ -*-===//
//
// Autofuse MLIR migration demo — AFIR dialect registration + custom attribute
// parsing/printing.
//
//===----------------------------------------------------------------------===//

#include "AFIR/AFIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------===//
// AFIR dialect initialization
//===----------------------------------------------------------------------===//

// Include generated dialect implementation
#include "AFIR/Dialect.cpp.inc"

// Include generated enum definitions
#include "AFIR/Enums.cpp.inc"

// Include generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "AFIR/Attrs.cpp.inc"

// Include generated operation definitions
#define GET_OP_CLASSES
#include "AFIR/Ops.cpp.inc"

void mlir::afir::AFIRDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "AFIR/Attrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "AFIR/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing (AFIR uses standard MLIR types)
//===----------------------------------------------------------------------===//

mlir::Type mlir::afir::AFIRDialect::parseType(mlir::DialectAsmParser & /*parser*/) const {
  return mlir::Type();
}

void mlir::afir::AFIRDialect::printType(mlir::Type /*type*/, mlir::DialectAsmPrinter & /*printer*/) const {}

//===----------------------------------------------------------------------===//
// Attribute Parsing and Printing
//===----------------------------------------------------------------------===//

mlir::Attribute mlir::afir::AFIRDialect::parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const {
  llvm::StringRef attrType;
  mlir::Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrType, type, attr);
  if (parseResult.has_value() && mlir::succeeded(parseResult.value())) {
    return attr;
  }
  return mlir::Attribute();
}

void mlir::afir::AFIRDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {
  (void)generatedAttributePrinter(attr, printer);
}
