/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_node.cpp
 * \brief
 */


#include <map>
#include <unordered_map>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include "runtime/kernel.h"
#include "sk_node.h"
#include "sk_scope_launch.h"
#include "sk_common.h"

namespace {
using SkBindMap = std::unordered_map<uint64_t, std::array<uint64_t, 4>>;
using SkAllBinMap = std::unordered_map<aclrtBinHandle, SkBindMap>;

struct CoreFuncInitContext {
    ResolvedFunctionInfo* info;
    size_t coreIdx;
    size_t splitIdx;
    SkBindMap::iterator bindIt;
};

SkBindMap InitSuperKernelBindMap(aclrtBinHandle binHdl)
{
    struct __attribute__((packed)) SknlValuePayload {
        uint32_t res;
        SknlMapInfo info;
    };
    constexpr size_t payloadSize = sizeof(SknlValuePayload);

    size_t metaNum = 0;
    CHECK_ACL(rtBinaryGetMetaNum(binHdl, RT_BINARY_TYPE_SK_INFO, &metaNum));

    SK_LOGI("InitSuperKernelBindMap: binHdl=0x%lx, metaNum=%lu, payloadSize=%zu",
        (uint64_t)binHdl, metaNum, payloadSize);

    std::vector<uint8_t> dataPool(metaNum * payloadSize);
    std::vector<size_t> infoSize(metaNum, payloadSize);
    std::vector<void *> metaDataList(metaNum);

    for (size_t i = 0; i < metaNum; ++i) {
        metaDataList[i] = &dataPool[i * payloadSize];
    }

    CHECK_ACL(rtBinaryGetMetaInfo(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, metaDataList.data(),
        infoSize.data()));

    SkBindMap bindMap;
    for (size_t i = 0; i < metaNum; ++i) {
        SknlValuePayload *payload = (SknlValuePayload *)metaDataList[i];
        SknlMapInfo localInfo;
        memcpy_s(&localInfo, sizeof(SknlMapInfo), &(payload->info), sizeof(SknlMapInfo));

        SK_LOGI("InitSuperKernelBindMap: [%zu] globalFunc=0x%lx, skFunc[0]=0x%lx, skFunc[1]=0x%lx, skFunc[2]=0x%lx, skFunc[3]=0x%lx",
            i, (uint64_t)localInfo.globalFunc,
            (uint64_t)localInfo.sknlFunc[0],
            (uint64_t)localInfo.sknlFunc[1],
            (uint64_t)localInfo.sknlFunc[2],
            (uint64_t)localInfo.sknlFunc[3]);

        bindMap[(uint64_t)(localInfo.globalFunc)] = {
            (uint64_t)(localInfo.sknlFunc[0]),
            (uint64_t)(localInfo.sknlFunc[1]),
            (uint64_t)(localInfo.sknlFunc[2]),
            (uint64_t)(localInfo.sknlFunc[3])
        };
    }
    return bindMap;
}

const SkBindMap& GetSkBindMap(aclrtBinHandle binHdl)
{
    static SkAllBinMap allBinMap;
    auto it = allBinMap.find(binHdl);
    if (it != allBinMap.end()) {
        return it->second;
    }
    allBinMap[binHdl] = InitSuperKernelBindMap(binHdl);
    return allBinMap[binHdl];
}

size_t AlignUpAndClamp(size_t value, size_t coreIdx)
{
    constexpr size_t aicFuncMaxPrefetchCnt = 0x4000; // 16k
    constexpr size_t aivFuncMaxPrefetchCnt = 0x2000; // 8k
    constexpr size_t alignNum = 0x800; // 2k
    size_t prefetchCntValue = (value + alignNum - 1) & ~(alignNum - 1); // 以2k为单位向上取整
    if (coreIdx == 0 && prefetchCntValue > aicFuncMaxPrefetchCnt) {
        prefetchCntValue = aicFuncMaxPrefetchCnt;
    } else if (coreIdx == 1 && prefetchCntValue > aivFuncMaxPrefetchCnt) {
        prefetchCntValue = aivFuncMaxPrefetchCnt;
    }
    return prefetchCntValue / alignNum;
}

void InitSingleCoreFunc(const CoreFuncInitContext& ctx, aclrtBinHandle binHdl, void *binDevAddr)
{
    std::string coreName = "";
    if (ctx.coreIdx == 0) {
        coreName = "AIC";
    } else if (ctx.coreIdx == 1) {
        coreName = "AIV";
    }
    uint64_t skFuncOffset = ctx.bindIt->second[ctx.splitIdx];
    ctx.info->funcAddr[ctx.coreIdx] = skFuncOffset + (uint64_t)binDevAddr;
    void *binHostAddr = nullptr;
    uint32_t binHostSize = 0;
    CHECK_ACL(rtGetBinBuffer(binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize));
    std::string symbolName;
    uint64_t funcSize = 0;
    if (GetFuncSymbolInfo(static_cast<const char*>(binHostAddr), binHostSize, skFuncOffset,
                          symbolName, funcSize)) {
        ctx.info->prefetchCnt[ctx.coreIdx] = AlignUpAndClamp(funcSize, ctx.coreIdx);
        SK_LOGI("InitKernelResolvedFuncs: split[%zu] %s symbol=%s, size=0x%lx",
                ctx.splitIdx, coreName.c_str(), symbolName.c_str(), funcSize);
    } else {
        SK_LOGW("InitKernelResolvedFuncs: split[%zu] Failed to get %s symbol info, prefetchCnt[%zu]=0",
                ctx.splitIdx, coreName.c_str(), ctx.coreIdx);
    }
}

bool InitKernelResolvedFuncs(KernelInfos &kernelInfos)
{
    aclrtBinHandle binHdl = kernelInfos.binHdl;
    aclrtFuncHandle oriFuncHdl = kernelInfos.funcHdl;
    if (binHdl == nullptr || oriFuncHdl == nullptr) {
        SK_LOGE("InitKernelResolvedFuncs: invalid bin handle or function handle for kernel %s", kernelInfos.funcName.c_str());
        return false;
    }
    SkBindMap bindMap = GetSkBindMap(binHdl);
    size_t binDevSize = 0;
    void *binDevAddr = nullptr;
    CHECK_ACL(aclrtBinaryGetDevAddress(binHdl, &binDevAddr, &binDevSize));
    void *addr[2] = {nullptr, nullptr}; // {aic addr, aiv addr}
    CHECK_ACL(aclrtGetFunctionAddr(oriFuncHdl, addr, addr + 1));

    uint64_t aicOffset = (uint64_t)addr[0] - (uint64_t)binDevAddr;
    uint64_t aivOffset = (uint64_t)addr[1] - (uint64_t)binDevAddr;
    SK_LOGI("InitKernelResolvedFuncs: funcName=%s, binDevAddr=0x%lx, binDevSize=%lu, aicAddr=0x%lx, aivAddr=0x%lx",
        kernelInfos.funcName.c_str(), (uint64_t)binDevAddr, binDevSize, (uint64_t)addr[0], (uint64_t)addr[1]);
    SK_LOGI("InitKernelResolvedFuncs: aicOffset=0x%lx, aivOffset=0x%lx", aicOffset, aivOffset);

    auto aicItor = bindMap.find(aicOffset);
    auto aivItor = bindMap.find(aivOffset);
    SK_LOGI("InitKernelResolvedFuncs: bindMap size=%lu, aicFound=%d, aivFound=%d",
        bindMap.size(), aicItor != bindMap.end(), aivItor != bindMap.end());

    for (size_t i = 0; i < kMaxSplitBinCount; ++i) {
        ResolvedFunctionInfo info{};
        if (aicItor != bindMap.end()) {
            CoreFuncInitContext aicCtx = {&info, 0, i, aicItor};
            InitSingleCoreFunc(aicCtx, binHdl, binDevAddr);
        }
        if (aivItor != bindMap.end()) {
            CoreFuncInitContext aivCtx = {&info, 1, i, aivItor};
            InitSingleCoreFunc(aivCtx, binHdl, binDevAddr);
        }
        SK_LOGI("InitKernelResolvedFuncs: split[%zu] funcAddr[0]=0x%lx, funcAddr[1]=0x%lx, "
                "prefetchCnt[0]=0x%lx, prefetchCnt[1]=0x%lx",
                i, info.funcAddr[0], info.funcAddr[1], info.prefetchCnt[0], info.prefetchCnt[1]);
        kernelInfos.resolvedFuncs[i] = info;
    }
    return true;
}

SkKernelType NormalizeKernelType(uint32_t kernelType, const uint32_t taskRatio[2]) {
    switch (kernelType) {
        case K_TYPE_AIC:
        case K_TYPE_AIC_ROLLBACK:
            return SkKernelType::AIC_ONLY;
        case K_TYPE_AIV:
        case K_TYPE_AIV_ROLLBACK:
        case K_TYPE_MIX_AIV_MAIN:
            return SkKernelType::AIV_ONLY;
        case K_TYPE_MIX_AIC_MAIN:
            if (taskRatio[1] == 0) {
                return SkKernelType::AIC_ONLY;
            }
            if (taskRatio[1] == 1) {
                return SkKernelType::MIX_AIC_1_1;
            }
            if (taskRatio[1] == 2) {
                return SkKernelType::MIX_AIC_1_2;
            }
            break;
        default:
            break;
    }
    return SkKernelType::DEFAULT;
}
} // namespace

bool SuperKernelBaseNode::InitNode() {
    if (originTask == nullptr) {
        SK_LOGE("Origin task is null for nodeIdxInStream %lu in streamIdxInGraph %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelKernelNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    if (aclrtTaskGetKernelParams(*originTask, &kernelParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to get kernel params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    JudgeTaskKernelInfo scopeKernelInfo;
    if (IsScopeKernel(kernelParams, &scopeKernelInfo)){
        SK_LOGI("Kernel node %s for task %u in stream %u is scope kernel.", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
        isFusible = scopeKernelInfo.isFuseEnable;
        isScopeNode = true;
        if (isFusible && scopeKernelInfo.scopeName != nullptr){
            isScopeBegin = scopeKernelInfo.isBegin;
            isScopeEnd = !isScopeBegin;
            char* rawPtr = scopeKernelInfo.scopeName.get();
            scopeName = std::string(rawPtr);
        }
    }

    if (kernelParams.taskGrp != nullptr) {
        SK_LOGI("Kernel task group is not null for task %u in stream %u, which cannot be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
        return true;
    }
    SK_LOGI("Kernel node %s for task %u in stream %u can be fused in super kernel.", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
    isFusible = true;

    nodeInfos.kernelInfos.taskRatio[0] = kernelParams.sk_task_ratio[0];
    nodeInfos.kernelInfos.taskRatio[1] = kernelParams.sk_task_ratio[1];
    nodeInfos.kernelInfos.kernelType = NormalizeKernelType(kernelParams.sk_kernel_type, kernelParams.sk_task_ratio); // todo : 通过aclrtGetFunctionAttribute进行获取 参数选项：,ratio也是通过参数选项进行获取
    nodeInfos.kernelInfos.numBlocks = originTask->kernel.numBlocks;
    nodeInfos.kernelInfos.devArgs = originTask->kernel.devArgs;
    nodeInfos.kernelInfos.binHdl = reinterpret_cast<aclrtBinHandle>(kernelParams.binHandle);
    nodeInfos.kernelInfos.funcHdl = kernelParams.funcHandle;
    if (kernelParams.func_name != nullptr) {
        nodeInfos.kernelInfos.funcName = kernelParams.func_name;  // todo ： 需要调用aclrtGetFunctionName进行获取
    }
    if (!nodeInfos.kernelInfos.funcName.empty() && nodeInfos.kernelInfos.binHdl != nullptr) {
        InitKernelResolvedFuncs(nodeInfos.kernelInfos);
    }
    return true;
}

bool SuperKernelKernelNode::InValidateNode() {
    if (!isFusible) {
        SK_LOGE("Kernel node %s for task %u in stream %u can not be fused in super kernel, which should not been invalidated.", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    SK_LOGI("Invalidating kernel node %s for task %u in stream %u, which will be fused in super kernel.", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
    kernelParams.flag = aclrtTaskFlag::ACL_RT_TASK_INVALID;
    if (aclrtTaskSetKernelParams(*originTask, &kernelParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate kernel node %s for task %u in stream %u", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelKernelNode::Update(const UpdateContext &ctx) {
    // 无参数 → invalid
    if (ctx.launchInfo == nullptr || ctx.skEntryFunc == nullptr) {
        return InValidateNode();
    }

    // 有参数 → 持久化到 RTS buffer
    aclrtArgsHandle ahdl = nullptr;
    aclrtParamHandle phdl = nullptr;
    size_t memSize = 0;
    size_t devArgsSize = 0;
    void *buf = nullptr;
    void *argsPtr = nullptr;

    const size_t MAX_HANDLE_MEM_SIZE = 1024 * 1024;  // 1MB
    const size_t MAX_ARGS_MEM_SIZE = 256 * 1024 * 1024;  // 64MB
    CHECK_ACL(aclrtKernelArgsGetHandleMemSize(ctx.skEntryFunc, &memSize));
    if (memSize == 0 || memSize > MAX_HANDLE_MEM_SIZE) {
        SK_LOGE("invalid memSize: %zu", memSize);
        return false;
    }
    ahdl = (aclrtArgsHandle)malloc(memSize);
    if (ahdl == nullptr) {
        SK_LOGE("malloc memSize failed");
        return false;
    }
    CHECK_ACL(aclrtKernelArgsGetMemSize(ctx.skEntryFunc, ctx.launchInfo->devArgs.get()->skHeader.totalSize, &devArgsSize));
    if (devArgsSize == 0 || devArgsSize > MAX_ARGS_MEM_SIZE) {
        SK_LOGE("invalid devArgsSize: %zu", devArgsSize);
        return false;
    }
    void *devArgs = nullptr;
    devArgs = malloc(devArgsSize);
    if (devArgs == nullptr) {
        SK_LOGE("malloc devArgsSize failed");
        return false;
    }
    CHECK_ACL(aclrtKernelArgsInitByUserMem(ctx.skEntryFunc, ahdl, devArgs, devArgsSize));

    CHECK_ACL(aclrtKernelArgsAppendPlaceHolder(ahdl, &phdl));
    CHECK_ACL(aclrtKernelArgsGetPlaceHolderBuffer(ahdl, phdl, ctx.launchInfo->devArgs.get()->skHeader.totalSize, (void**)&argsPtr));
    errno_t err = memcpy_s(argsPtr, ctx.launchInfo->devArgs.get()->skHeader.totalSize, ctx.launchInfo->devArgs.get(), ctx.launchInfo->devArgs.get()->skHeader.totalSize);
    if (err != 0) {
        SK_LOGE("memcpy_s failed");
        return false;
    }
    CHECK_ACL(aclrtKernelArgsFinalize(ahdl));

    kernelParams.funcHandle = ctx.skEntryFunc;
    kernelParams.argsHandle = ahdl;
    kernelParams.numBlocks = ctx.launchInfo->entryInfo.numBlocks;
    kernelParams.flag = aclrtTaskFlag::ACL_RT_TASK_VALID;

    bool ret = (aclrtTaskSetKernelParams(*originTask, &kernelParams) == ACL_SUCCESS);
    if (!ret) {
        SK_LOGE("Failed to update kernel node %s for task %u in stream %u", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
    } else {
        SK_LOGI("Updated kernel node %s for task %u in stream %u with argsHandle", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
    }

    return ret;
}

bool SuperKernelEventNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    if (aclrtTaskGetEventParams(*originTask, &eventParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to get event params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    nodeInfos.syncInfos.eventId = eventParams.eventId;
    nodeInfos.syncInfos.addrValue = eventParams.eventAddr;
    if (eventParams.eventType != aclrtEventType::ACL_RT_EVENT_MEMORY) {
         SK_LOGI("Event type is not memory based for task %u in stream %u, which cannot be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
         return true;
    }
    SK_LOGI("Event type of task %u in stream %u is memory based, which can be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
    isFusible = true;
    return true;
}

bool SuperKernelEventNode::InValidateNode() {
    if (!isFusible) {
        SK_LOGE("Event node with eventId %lu for task %u in stream %u can not be fused in super kernel, which should not been invalidated.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    SK_LOGI("Invalidating event node with eventId %lu for task %u in stream %u, which will be fused in super kernel.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
    eventParams.flag = aclrtTaskFlag::ACL_RT_TASK_INVALID;
    if (aclrtTaskSetEventParams(*originTask, &eventParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate event node with eventId %lu for task %u in stream %u", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelMemoryNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    if (aclrtTaskGetMemValueParams(*originTask, &memValueParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to get memory value params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    nodeInfos.syncInfos.eventId = reinterpret_cast<uint64_t>(memValueParams.valueAddr);
    nodeInfos.syncInfos.addrValue = memValueParams.valueAddr;
    SK_LOGI("Memory value type is write for task %u in stream %u, which can be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
    isFusible = true;
    return true;
}

bool SuperKernelMemoryNode::InValidateNode() {
    if (!isFusible) {
        SK_LOGE("Memory node with eventId %lu for task %u in stream %u can not be fused in super kernel, which should not been invalidated.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    SK_LOGI("Invalidating memory node with eventId %lu for task %u in stream %u, which will be fused in super kernel.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
    memValueParams.flag = aclrtTaskFlag::ACL_RT_TASK_INVALID;
    if (aclrtTaskSetMemValueParams(*originTask, &memValueParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate memory node with eventId %lu for task %u in stream %u", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelDefaultNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    SK_LOGI("Default task type for task %lu in stream %u, which cannot be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
    return true;
}

bool SuperKernelDefaultNode::InValidateNode() {
    SK_LOGE("Default task type for task %lu in stream %u should not be invalidated.", nodeIdxInStream, streamIdxInGraph);
    return false;
}
