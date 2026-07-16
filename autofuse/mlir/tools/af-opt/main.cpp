/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/Registration.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "AFIR/AFIR.h"
#include "AFIRToAscendC/AFIRToAscendC.h"
#include "AFIRToAscendCQueue/AFIRToAscendCQueue.h"

namespace {
constexpr const char *kAutofuseMlirVersion = "af-mlir-phase1";

void PrintAutofuseMlirVersion(llvm::raw_ostream &os) {
  os << "  Autofuse MLIR version " << kAutofuseMlirVersion << "\n";
}
}  // namespace

int main(int argc, char **argv) {
  llvm::cl::AddExtraVersionPrinter(PrintAutofuseMlirVersion);

  mlir::DialectRegistry registry;
  mlir::ascir::registerDialects(registry);
  mlir::ascendc::registerInlinerInterfaces(registry);
  mlir::ascir::registerExtensions(registry);
  mlir::ascir::registerPasses();

  // Autofuse AFIR dialect + demo lowering passes.
  registry.insert<mlir::afir::AFIRDialect>();
  mlir::afir::registerAFIRToAscendCPass();
  mlir::afir::registerAFIRToAscendCQueuePass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Autofuse MLIR migration optimizer\n", registry));
}
