// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt -split-input-file %s | FileCheck %s

// Roundtrip coverage for the full AFIR op set: unary math, binary/compare,
// condition (where), concat, cast, and reduce (with their verifiers and
// return-type inference interfaces active).

func.func @allops(%x: tensor<4x8xf32>, %b: tensor<1x8xf32>, %c: tensor<4x8xui8>) -> tensor<8xf32> {
  %e   = afir.exp %x : tensor<4x8xf32> -> tensor<4x8xf32>
  %add = afir.add %e, %b : (tensor<4x8xf32>, tensor<1x8xf32>) -> tensor<4x8xf32>
  %ge  = afir.ge %add, %x : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xui8>
  %sel = afir.where %ge, %add, %x : (tensor<4x8xui8>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %cat = afir.concat %sel, %x {concat_axis = 0 : i32} : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<8x8xf32>
  %cst = afir.cast %c : tensor<4x8xui8> -> tensor<4x8xf32>
  %sum = afir.sum %cst {axis = 0 : i32} : tensor<4x8xf32> -> tensor<8xf32>
  return %sum : tensor<8xf32>
}

// CHECK-LABEL: func.func @allops
// CHECK: afir.exp
// CHECK: afir.add
// CHECK: afir.ge
// CHECK: afir.where
// CHECK: afir.concat
// CHECK: afir.cast
// CHECK: afir.sum
// CHECK: return

// -----

// Reduce return-type inference: dropping axis 1 of 4x8 yields 4.
// RUN: af-opt -split-input-file %s | FileCheck %s --check-prefix=REDUCE

func.func @reduce_infer(%a: tensor<4x8xf32>) -> tensor<4xf32> {
  %0 = "afir.sum"(%a) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// REDUCE-LABEL: func.func @reduce_infer
// REDUCE: afir.sum
// REDUCE-SAME: -> tensor<4xf32>
