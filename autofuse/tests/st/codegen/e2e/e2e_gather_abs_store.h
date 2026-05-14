/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_E2E_GATHER_ABS_STORE_H__
#define __TEST_E2E_GATHER_ABS_STORE_H__

#include "ascendc_ir.h"

void GatherAbsStore_AfterQueBufAlloc(af::AscGraph &graph);
void GatherAbsStore_AfterScheduler_z1z2_splitTo_z1z2TBz1z2Tbz1z2t(af::AscGraph &graph);
void GatherAbsStore_AfterGetApiInfo(af::AscGraph &graph);
void GatherAbsStore_AfterInferOutput(af::AscGraph &graph);
void GatherAbsStore_BeforeAutofuse(af::AscGraph &graph);

#endif