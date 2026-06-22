/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_E2E_LOAD_RECIPROCAL_STORE_H__
#define __TEST_E2E_LOAD_RECIPROCAL_STORE_H__

#include "ascendc_ir.h"

void LoadReciprocalStore_BeforeAutofuse(af::AscGraph &graph);
void LoadReciprocalStore_AfterInferOutput(af::AscGraph &graph);
void LoadReciprocalStore_AfterGetApiInfo(af::AscGraph &graph);
void LoadReciprocalStore_AfterQueBufAlloc(af::AscGraph &graph);

void LoadReciprocalStore_AfterScheduler_z0z1z2_splitTo_z0z1TBz0z1Tbz1tz2(af::AscGraph &graph);
#endif
