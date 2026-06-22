/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_BRC_BUF_GRAPH_H_
#define TEST_BRC_BUF_GRAPH_H_
#include "ascir_ops.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace att {
// case1 : b轴为r轴
void BrcBufBeforeAutoFuse1(af::AscGraph &graph);
void BrcBufAfterScheduler1(af::AscGraph &graph);
void BrcBufAfterQueBufAlloc1(af::AscGraph &graph);

// case2 : b轴非r轴
void BrcBufBeforeAutoFuse2(af::AscGraph &graph);

// case3 : 无reduce节点
void BrcBufBeforeAutoFuse3(af::AscGraph &graph);
void BrcBufAfterScheduler3(af::AscGraph &graph);
void BrcBufAfterQueBufAlloc3(af::AscGraph &graph);

// case4 : 无brc节点
void BrcBufBeforeAutoFuse4(af::AscGraph &graph);
void BrcBufAfterScheduler4(af::AscGraph &graph);
void BrcBufAfterQueBufAlloc4(af::AscGraph &graph);
}  // namespace att
#endif
