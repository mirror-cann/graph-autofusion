#ifndef __DYNAMICQUANTTILINGDATA_HEADER__
#define __DYNAMICQUANTTILINGDATA_HEADER__
#include "kernel_tiling/kernel_tiling.h"

class DynamicQuantTilingData
{
public:
    uint32_t coreNum = 40;
    uint32_t rowLen = 7168;
    uint32_t headCoreNum = 16;
    uint32_t rowPerHeadCore = 231;
    uint32_t rowPerTailCore = 230;
    uint32_t multiRowNumHeadCore = 2;
    uint32_t multiRowNumTailCore = 2;
    uint32_t innerLoopEle = 0;
    uint32_t innerLoopTimes = 0;
    uint32_t innerLoopTail = 0;
    uint32_t groupNum = 0;
    uint32_t alignGroupNum = 0;
    uint32_t hasSmooth = 0;
    uint32_t unused = 0;
    uint32_t sizeH = 0;
    uint32_t sizeX = 0;
    uint32_t sizeZOut = 0;
    uint32_t sizeCopyRow = 0;
    uint32_t numCopyRow = 0;
    uint32_t numHeadCore = 0;
    uint32_t numTailCore = 0;
    uint32_t numHeadTimes = 0;
    uint32_t numTailTimes = 0;
    uint32_t numLastTailRow = 0;
    uint32_t alignType = 0;
    uint8_t DynamicQuantTilingDataPH[4] = {0, 0, 0, 0};
};

#if defined(CONST_TILING)
const DynamicQuantTilingData STATIC_TILING_VAR = {48, 7168, 0, 192, 192, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,{0, 0, 0, 0}};
#endif

#endif // __DYNAMICQUANTTILINGDATA_HEADER__