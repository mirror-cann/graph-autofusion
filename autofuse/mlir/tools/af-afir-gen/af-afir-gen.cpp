/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- af-afir-gen.cpp - Emit AFIR from a built-in schedule example ----*- C++ -*-===//
//
// Standalone driver that exercises the AfirBuilder core with a hand-built
// elewise+broadcast KernelInfo and prints the resulting AFIR module. Pipe into
// `af-opt --convert-afir-to-ascendc` to see the ascendc lowering.
//
//===----------------------------------------------------------------------===//

#include "AfirBuilder/AfirBuilder.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace af_demo;

static KernelInfo makeElewiseBrcExample() {
  KernelInfo k;
  k.name = "elewise_brc";

  TensorInfo xTy{{4, 8}, "float16", /*tensor_id=*/0, /*position=*/0, {0, 1}};
  TensorInfo bTy{{1, 8}, "float16", /*tensor_id=*/1, /*position=*/0, {1}};
  TensorInfo yTy{{4, 8}, "float16", /*tensor_id=*/6, /*position=*/0, {0, 1}};
  k.inputs = {{"x", xTy}, {"b", bTy}};
  k.outputs = {{"y", yTy}};

  GraphInfo g;
  g.name = "elewise_brc";
  g.tiling_key = 0;
  g.size_vars = {"M", "N"};
  g.axes = {
      {/*id=*/0, "m", "original", "M"},
      {/*id=*/1, "n", "tile_inner", "N"},
  };

  auto ubIn = [](std::vector<int64_t> shape, int64_t tid) {
    return TensorInfo{std::move(shape), "float16", tid, /*position=vector_in*/ 1, {0, 1}};
  };
  auto ubCalc = [](int64_t tid) { return TensorInfo{{4, 8}, "float16", tid, /*position=vector_calc*/ 3, {0, 1}}; };

  g.nodes = {
      {"x", "Data", {}, TensorInfo{{4, 8}, "float16", 0, 0, {0, 1}}, {}},
      {"b", "Data", {}, TensorInfo{{1, 8}, "float16", 1, 0, {1}}, {}},
      {"load_x", "Load", {"x"}, ubIn({4, 8}, 2), {0, 1}},
      {"load_b", "Load", {"b"}, ubIn({1, 8}, 3), {0, 1}},
      {"bc_b", "Broadcast", {"load_b"}, ubCalc(4), {0, 1}},
      {"add", "Add", {"load_x", "bc_b"}, ubCalc(5), {0, 1}},
      {"store_y", "Store", {"add"}, TensorInfo{{4, 8}, "float16", 6, /*vector_out*/ 2, {0, 1}}, {0, 1}},
      {"y", "Output", {"store_y"}, yTy, {}},
  };

  k.graphs = {g};
  return k;
}

int main() {
  mlir::MLIRContext context;
  KernelInfo kernel = makeElewiseBrcExample();
  auto module = BuildAfirModule(context, kernel);
  if (!module) {
    llvm::errs() << "AfirBuilder failed\n";
    return 1;
  }
  module->print(llvm::outs());
  return 0;
}
