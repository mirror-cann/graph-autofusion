// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Ops.td - AFIR operation definitions -----------------*- tablegen -*-===//

#ifndef AFIR_OPS
#define AFIR_OPS

include "Dialect.td"
include "Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//

include "Ops/Basic/data.td"
include "Ops/Basic/math.td"
include "Ops/Basic/broadcast.td"
include "Ops/Basic/cast.td"
include "Ops/Basic/condition.td"

//===----------------------------------------------------------------------===//
// Advanced Operations
//===----------------------------------------------------------------------===//

include "Ops/Adv/concat.td"
include "Ops/Adv/reduce.td"

#endif // AFIR_OPS
