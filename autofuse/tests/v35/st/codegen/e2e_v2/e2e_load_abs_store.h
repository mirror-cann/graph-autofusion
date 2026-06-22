/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_E2E_LOAD_ABS_STORE_H__
#define __TEST_E2E_LOAD_ABS_STORE_H__

#include "ascendc_ir.h"

void LoadAbsStore_BeforeAutofuse(af::AscGraph &graph);
void LoadAbsStore_BeforeAutofuse_DiscreteStore(af::AscGraph &graph);
void LoadAbsStore_BeforeAutofuse_DiscreteStoreMergeAxis(af::AscGraph &graph);
void LoadAbsStore_BeforeAutofuse_StoreScalar(af::AscGraph &graph);
void LoadAbsStore_BeforeAutofuse_StoreEmptyTensor(af::AscGraph &graph);
void LoadAbsStore_AfterInferOutput(af::AscGraph &graph);
void LoadAbsStore_AfterGetApiInfo(af::AscGraph &graph);
void LoadAbsStore_AfterScheduler(af::AscGraph &graph);
void LoadAbsStore_AfterQueBufAlloc(af::AscGraph &graph);

void LoadAbsStore_AfterScheduler_z0_SplitTo_z0TBz0Tbz0t(af::AscGraph &graph);
void LoadAbsStore_BeforeAutofuse_z0_SplitTo_z0TBz0Tbz0t(af::AscGraph &graph);
void LoadAbsStore_AfterScheduler_z0_SplitTo_z0Bz0b(af::AscGraph &graph);
void LoadAbsStore_AfterScheduler_DiscreteStoreMergeAxis(af::AscGraph &graph);
void LoadAbsStore_AfterScheduler_store_scalar(af::AscGraph &graph);

void LoadAbsStore_AfterScheduler_z0z1z2_splitTo_z0z1TBz0z1Tbz1tz2(af::AscGraph &graph);
#endif
