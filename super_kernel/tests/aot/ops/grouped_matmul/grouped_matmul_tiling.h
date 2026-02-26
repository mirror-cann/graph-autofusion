#pragma once

#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"


struct GMMBaseParams
{
    uint32_t groupNum;
    uint32_t coreNum;
    uint32_t activeType;
    uint32_t ubBaseK;
    uint32_t ubBaseN;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t singleWeight;
    uint32_t singleX;
    uint32_t singleY;
    int32_t groupType;
    uint32_t singleN;
    uint32_t quantParam;
    uint32_t groupListType;
    uint32_t m;
    uint32_t hasBias;
    uint64_t workspaceSize;
    uint64_t totalInGroup;
    uint64_t k;
    uint64_t n;
    uint64_t vBaseM;
    uint64_t parallNum;
    uint64_t quantGroupNum;
    uint64_t isPreTiling;
    uint32_t withOffset = 0;
    uint8_t GMMBaseParamsPH[4] = {};
};

struct GMMArray
{
    int32_t mList[128];
    int32_t kList[128];
    int32_t nList[128];
};

struct GMMTilingData
{
    GMMBaseParams gmmBaseParams;
    GMMArray gmmArray;
    TCubeTiling mmTilingData;
};

/*
{
    GMMBaseParams gmmBaseParams = {1, 20, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
    GMMArray gmmArray = {{32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, };
    TCubeTiling mmTilingData = {20, 32, 32, 32, 32, 32, 256, 32, 32, 256, 64, 56, 8, 1, 1, 0, 0, 0, 0, 18432, 32768, 0, 1, 1, 1, 1, 28, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
}
*/

