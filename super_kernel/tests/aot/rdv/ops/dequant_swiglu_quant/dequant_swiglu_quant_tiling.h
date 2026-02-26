/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"

struct SwiGluTilingData
{
public:
    uint32_t is32BAligned = 1;
    uint32_t isDoubleBuffer = 0;
    uint64_t rowLen = 32;
    uint64_t colLen = 16;
    uint32_t baseRowLen = 1;
    uint32_t baseColLen = 16;
    uint32_t activateLeft = 0;
    uint32_t biasIsEmpty = 0;
    uint32_t quantScaleIsEmpty = 1;
    uint32_t activateScaleIsEmpty = 0;
    uint64_t swiColLen = 0;
    uint64_t perRowLen = 0;
    uint64_t modRowLen = 0;
    uint32_t usedCoreNum = 32;
    uint8_t SwiGluTilingDataPH[4] = {0, 0, 0, 0};
};

class DequantSwigluQuantBaseTilingData
{
public:
    int64_t inDimx = 9216;
    int64_t inDimy = 4096;
    int64_t outDimy = 2048;
    int64_t UbFactorDimx = 4;
    int64_t UbFactorDimy = 2048;
    int64_t usedCoreNum = 36;
    int64_t maxCoreNum = 36;
    int64_t inGroupNum = 1;
    int64_t hasBias = 0;
    int64_t quantMode = 1;
    int64_t actRight = 1;
    int64_t quantScaleDtype = 0;
    int64_t groupIndexDtype = 1;
    int64_t needSmoothScale = 1;
    int64_t biasDtype = 0;
    int64_t speGroupType = 0;
    int64_t activationScaleIsEmpty = 0;
    int64_t quantIsOne = 0;
    int64_t swigluMode = 0;
    float clampLimit = 7.0;
    float gluAlpha = 1.7020000219345093;
    float gluBias = 1.0;
    uint8_t DequantSwigluQuantBaseTilingDataPH[4] = {0, 0, 0, 0};
};

#if defined(CONST_TILING)
const DequantSwigluQuantBaseTilingData STATIC_TILING_VAR = {9216, 4096, 2048, 4, 2048, 36, 36, 1, 0, 1, 1, 0, 1, 1, 0,0,0,0,0,7.0,1.7020000219345093, 1.0,{0,0,0,0}};
#endif