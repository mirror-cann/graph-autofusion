// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: af-opt --convert-afir-to-ascendc %s | FileCheck %s --check-prefix=LOWER

// Demo: AFIR elementwise ops roundtrip, then lower to the ascendc dialect.

func.func @elewise(%a: tensor<4x8xf16>, %b: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %0 = afir.add %a, %b : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf16>
  %1 = afir.mul %0, %b : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf16>
  return %1 : tensor<4x8xf16>
}

// ROUNDTRIP-LABEL: func.func @elewise
// ROUNDTRIP-SAME: tensor<4x8xf16>
// ROUNDTRIP: afir.add
// ROUNDTRIP: afir.mul
// ROUNDTRIP: return

// LOWER-LABEL: func.func @elewise
// LOWER-SAME: !ascendc.local_tensor<4x8xf16>
// LOWER: ascendc.tbuf : <veccalc>
// LOWER: ascendc.tbuf.get_tensor
// LOWER: ascendc.add_l3
// LOWER: ascendc.mul_l3
// LOWER: return
// LOWER-NOT: afir.
