/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef __WEIGHTQUANTBATCHMATMULV2TILINGDATA_HEADER__
#define __WEIGHTQUANTBATCHMATMULV2TILINGDATA_HEADER__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct cWeightQuantBatchMatmulV2MsdTilingData
{
public:
    uint8_t cubeBlockDimN = 1;
    uint8_t cubeBlockDimM = 1;
    uint8_t cubeBlockDimK = 0;
    uint8_t hasBias = 0;
    uint16_t v1BaseM = 1;
    uint16_t preloadTimes = 0;
    uint16_t taskNSize = 0;
    uint16_t taskSingleCoreNSize = 0;
    uint16_t postProcessBaseM = 0;
    uint16_t postProcessSingleCoreM = 0;
    uint32_t preProcessUsedVecNum = 32; 
    uint32_t v1BaseK = 320;
    uint64_t mSize = 32; 
    uint64_t kSize = 320;
    uint64_t nSize = 256;
    uint64_t groupPack = 0;
    uint64_t groupSize = 0;
    AscendC::tiling::TCubeTiling matmulTiling = {1, 64, 256, 320, 320, 64, 256, 320, 64, 256, 128, 3, 3, 1, 1, 0, 0, 0, 0, 122880, 65536, 0, 1, 1, 1, 1, 3, 3, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
};


struct WeightQuantBatchMatmulV2MsdGroupTilingData
{
    uint8_t vecBlockDimN;
    uint8_t cubeBlockDimK;
    uint8_t cubeBlockDimN;
    uint8_t vec1SingleCoreM;
    uint8_t hasBias;
    uint8_t reserve1;
    uint16_t reserve2;
    uint32_t reserve3;
    uint32_t singleCoreK;
    uint32_t vecSingleCoreN;
    uint32_t singleCoreGroup;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t groupSize;
    AscendC::tiling::TCubeTiling matmulTiling;
}__attribute__((__may_alias__));


struct WeightQuantBatchMatmulV2TilingData
{
    uint8_t vecBlockDimN = 7;
    uint8_t vecBlockDimK = 1;
    uint8_t cubeBlockDimN = 1;
    uint8_t cubeBlockDimM = 16;
    uint8_t cubeBlockDimK = 0;
    uint8_t kPadSize = 0;
    uint8_t nPadSize = 0;
    uint8_t haveBatchA = 0;
    uint8_t haveBatchB = 0;
    uint8_t reserve1 = 0;
    uint8_t reserve2 = 0;
    uint8_t reserve3 = 0;
    uint16_t vecSingleKGroupNum = 0;
    uint16_t vecSingleKTailGroupNum = 0;
    uint16_t AL1Pingpong = 0;
    uint16_t BL1Pingpong = 0;
    uint32_t vecSingleK = 48;
    uint32_t vecSingleN = 256;
    uint32_t vec2SingleM = 0;
    uint32_t vecSingleKTail = 0;
    uint32_t vecSingleNTail = 0;
    uint32_t wInQueueSize = 0;
    uint32_t offsetInQueueSize = 0;
    uint32_t scaleInQueueSize = 0;
    uint32_t wOutQueueSize = 0;
    uint32_t antiQuantTmpBufferSize = 0;
    uint32_t vecCubeNRatio = 0;
    uint32_t vecCubeTailNRatio = 0;
    uint32_t vecCubeKRatio = 0;
    uint32_t vecCubeTailKRatio = 0;
    uint32_t cubeTailM = 512;
    uint32_t cubeTailN = 256;
    uint32_t cubeSingleNLoop = 1;
    uint32_t cubeSingleNTailLoop = 1;
    uint32_t repeatAxisMax = 0;
    uint64_t vecSingleKLoop = 7;
    uint64_t vecSingleNLoop = 1;
    uint64_t vecSingleKTailLoop = 0;
    uint64_t vecSingleNTailLoop = 1;
    uint64_t vec2SingleMLoop = 0;
    uint64_t kAlign = 320;
    uint64_t nAlign = 256;
    uint64_t kSize = 320;
    uint64_t nSize = 256;
    uint64_t groupSize = 0;
    uint64_t mSize = 8192;
    uint64_t blockBatch = 1;
    uint64_t shapeBatch = 1;
    uint64_t mAubSize = 0;
    uint64_t kAubSize = 0;
    uint64_t nBubSize = 0;
    uint64_t kBubSize = 0;
    uint64_t mCubSize = 0;
    uint64_t nCubSize = 0;
    uint64_t mAL1Size = 0;
    uint64_t kAL1Size = 0;
    uint64_t nBL1Size = 0;
    uint64_t kBL1Size = 0;
    AscendC::tiling::TCubeTiling matmulTiling = {1, 8192, 256, 320, 320, 512, 256, 320, 128, 256, 64, 20, 2, 4, 1, 0, 131072, 1, 0, 393216, 131072, 0, 1, 1, 1, 1, 5, 1, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
};

struct WeightQuantBatchMatmulV2MsdTilingData
{
    uint8_t cubeBlockDimN = 1;
    uint8_t cubeBlockDimM = 1;
    uint8_t cubeBlockDimK = 0;
    uint8_t hasBias = 0;
    uint16_t v1BaseM = 1;
    uint16_t preloadTimes = 0;
    uint16_t taskNSize = 0;
    uint16_t taskSingleCoreNSize = 0;
    uint16_t postProcessBaseM = 0;
    uint16_t postProcessSingleCoreM = 0;
    uint32_t preProcessUsedVecNum = 32; 
    uint32_t v1BaseK = 320;
    uint64_t mSize = 32; 
    uint64_t kSize = 320;
    uint64_t nSize = 256;
    uint64_t groupPack = 0;
    uint64_t groupSize = 0;
    AscendC::tiling::TCubeTiling matmulTiling = {1, 64, 256, 320, 320, 64, 256, 320, 64, 256, 128, 3, 3, 1, 1, 0, 0, 0, 0, 122880, 65536, 0, 1, 1, 1, 1, 3, 3, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
};

#endif // __WEIGHTQUANTBATCHMATMULV2TILINGDATA_HEADER__

