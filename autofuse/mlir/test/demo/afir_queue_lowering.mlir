// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --convert-afir-to-ascendc-queue %s | FileCheck %s
// RUN: af-opt --convert-afir-to-ascendc-queue %s | af-opt | FileCheck %s --check-prefix=ROUNDTRIP

// Queue-faithful lowering: a buffer-annotated AFIR vector kernel
// (load -> add -> store) becomes the full queue-driven ascendc form that a
// real Ascend C kernel compiles from — pipe + queue + alloc/enque/deque +
// global_tensor + data_copy_l2 + add_l2. Positions come from the schedule's
// `outputs` attribute (VECIN for loads, VECOUT for the result).

func.func @add_kernel(%x: tensor<32xf16>, %y: tensor<32xf16>) -> tensor<32xf16> {
  %lx = afir.load %x {outputs = [#afir.asc_tensor<tensor_id = 0, position = <vector_in>>]} : tensor<32xf16> -> tensor<32xf16>
  %ly = afir.load %y {outputs = [#afir.asc_tensor<tensor_id = 1, position = <vector_in>>]} : tensor<32xf16> -> tensor<32xf16>
  %s = afir.add %lx, %ly {outputs = [#afir.asc_tensor<tensor_id = 2, position = <vector_out>>]} : (tensor<32xf16>, tensor<32xf16>) -> tensor<32xf16>
  %st = afir.store %s {outputs = [#afir.asc_tensor<tensor_id = 3, position = <vector_out>>]} : tensor<32xf16> -> tensor<32xf16>
  return %st : tensor<32xf16>
}

// CHECK-LABEL: func.func @add_kernel
// CHECK-SAME: memref<?xf16, 22 : i32>
// CHECK-SAME: !emitasc.py_struct<"TilingData"
// CHECK-SAME: ascendc.aicore
// CHECK-SAME: cann.num_inputs = 2
// CHECK: ascendc.pipe
// CHECK: ascendc.queue : <vecin, 1>
// CHECK: ascendc.pipe.init_queue
// CHECK: ascendc.que_bind.alloc_tensor
// CHECK: ascendc.global_tensor.set_global_buffer
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.local_tensor<32xf16>, !ascendc.global_tensor<32xf16>, index
// CHECK: ascendc.que_bind.enque_tensor
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: ascendc.queue : <vecout, 1>
// CHECK: ascendc.add_l2 {{.*}} {ascendc.unit = "AiCore.Vector"}
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.global_tensor<32xf16>, !ascendc.local_tensor<32xf16>, index
// CHECK: return
// CHECK-NOT: afir.

// ROUNDTRIP: func.func @add_kernel
// ROUNDTRIP: ascendc.add_l2

// Unary elementwise (load -> abs -> store) lowers the same way, emitting the
// single-source ascendc unary L2 op abs_l2 {dst, src, calCount}.
func.func @abs_kernel(%x: tensor<32xf16>) -> tensor<32xf16> {
  %lx = afir.load %x {outputs = [#afir.asc_tensor<tensor_id = 0, position = <vector_in>>]} : tensor<32xf16> -> tensor<32xf16>
  %a = afir.abs %lx {outputs = [#afir.asc_tensor<tensor_id = 1, position = <vector_out>>]} : tensor<32xf16> -> tensor<32xf16>
  %st = afir.store %a {outputs = [#afir.asc_tensor<tensor_id = 2, position = <vector_out>>]} : tensor<32xf16> -> tensor<32xf16>
  return %st : tensor<32xf16>
}

// CHECK-LABEL: func.func @abs_kernel
// CHECK-SAME: cann.num_inputs = 1
// CHECK: ascendc.queue : <vecin, 1>
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.queue : <vecout, 1>
// CHECK: ascendc.abs_l2 {{.*}} {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, index
// CHECK: return
// CHECK-NOT: afir.

// ROUNDTRIP: func.func @abs_kernel
// ROUNDTRIP: ascendc.abs_l2

// Broadcast (load -> broadcast <1x16> to <4x16> -> store). Shapes ride as i32
// constant operands; the op carries constRank and the emitter recovers the
// stretched axis from the tensor types. Buffer sizes follow each tensor's own
// element count, so the VECIN queue is smaller than the VECOUT one.
func.func @bcast_kernel(%x: tensor<1x16xf16>) -> tensor<4x16xf16> {
  %lx = afir.load %x {outputs = [#afir.asc_tensor<tensor_id = 0, position = <vector_in>>]} : tensor<1x16xf16> -> tensor<1x16xf16>
  %b = afir.broadcast %lx {outputs = [#afir.asc_tensor<tensor_id = 1, position = <vector_out>>]} : tensor<1x16xf16> -> tensor<4x16xf16>
  %st = afir.store %b {outputs = [#afir.asc_tensor<tensor_id = 2, position = <vector_out>>]} : tensor<4x16xf16> -> tensor<4x16xf16>
  return %st : tensor<4x16xf16>
}

// CHECK-LABEL: func.func @bcast_kernel
// CHECK-SAME: cann.num_inputs = 1
// CHECK: ascendc.queue : <vecin, 1>
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.local_tensor<1x16xf16>, !ascendc.global_tensor<1x16xf16>, index
// CHECK: ascendc.queue : <vecout, 1>
// CHECK: ascendc.broadcast_l2 {{.*}} {ascendc.unit = "AiCore.Vector", constRank = 2 : i32{{.*}}} : !ascendc.local_tensor<4x16xf16>, !ascendc.local_tensor<1x16xf16>, i32, i32, i32, i32
// CHECK: return
// CHECK-NOT: afir.

// ROUNDTRIP: func.func @bcast_kernel
// ROUNDTRIP: ascendc.broadcast_l2
