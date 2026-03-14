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
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include "runtime/kernel.h"
#include "sk_log.h"
#include "sk_node.h"
#include "sk_scope_launch.h"
#include "sk_common.h"
#include "runtime/kernel.h"

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
    if (metaNum == 0) {
        SK_LOGW("metaNum is zero!");
        return SkBindMap();
    }

    SK_LOGI("binHdl=0x%lx, metaNum=%lu, payloadSize=%zu",
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

        SK_LOGI("[%zu] globalFunc=0x%lx, skFunc[0]=0x%lx, skFunc[1]=0x%lx, skFunc[2]=0x%lx, skFunc[3]=0x%lx",
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
        SK_LOGI("split[%zu] %s symbol=%s, size=0x%lx",
                ctx.splitIdx, coreName.c_str(), symbolName.c_str(), funcSize);
    } else {
        SK_LOGW("split[%zu] Failed to get %s symbol info, prefetchCnt[%zu]=0",
                ctx.splitIdx, coreName.c_str(), ctx.coreIdx);
    }
}

bool InitKernelResolvedFuncs(KernelInfos &kernelInfos)
{
    aclrtBinHandle binHdl = kernelInfos.binHdl;
    aclrtFuncHandle oriFuncHdl = kernelInfos.funcHdl;
    if (binHdl == nullptr || oriFuncHdl == nullptr) {
        SK_LOGE("invalid bin handle or function handle for kernel %s", kernelInfos.funcName.c_str());
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
    SK_LOGI("funcName=%s, binDevAddr=0x%lx, binDevSize=%lu, aicAddr=0x%lx, aivAddr=0x%lx",
        kernelInfos.funcName.c_str(), (uint64_t)binDevAddr, binDevSize, (uint64_t)addr[0], (uint64_t)addr[1]);

    SK_LOGI("aicOffset=0x%lx, aivOffset=0x%lx", aicOffset, aivOffset);

    auto aicItor = bindMap.find(aicOffset);
    auto aivItor = bindMap.find(aivOffset);
    SK_LOGI("bindMap size=%lu, aicFound=%d, aivFound=%d",
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

        SK_LOGI("split[%zu] funcAddr[0]=0x%lx, funcAddr[1]=0x%lx, "
                "prefetchCnt[0]=0x%lx, prefetchCnt[1]=0x%lx",
                i, info.funcAddr[0], info.funcAddr[1], info.prefetchCnt[0], info.prefetchCnt[1]);
        kernelInfos.resolvedFuncs[i] = info;
    }
    return true;
}

SkKernelType NormalizeKernelType(uint32_t kernelType, const uint32_t taskRatio[2]) {
    switch (kernelType) {
        case ACL_KERNEL_TYPE_CUBE:
            return SkKernelType::AIC_ONLY;
        case ACL_KERNEL_TYPE_VECTOR:
            return SkKernelType::AIV_ONLY;
        case ACL_KERNEL_TYPE_MIX:
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

struct JudgeTaskKernelInfo {
    bool isBegin = false;
    bool isEnd = false;
    bool isPlaceholder = false;
    bool isFuseEnable = true;
    std::unique_ptr<char[]> scopeName;
};

bool IsScopeKernel(aclmdlRIKernelTaskParams params, JudgeTaskKernelInfo* info) {
    const char* defaultScopeName = "default_sk_scope_name";
    const char* placeholderName = "sk_placeholder_kernel";
    const char* targetBeginName = "sk_scope_kernel_begin";
    const char* targetEndName = "sk_scope_kernel_end";
    char kernelName[MAX_SCOPE_NAME_LENN] = {0};
    int ret = aclrtGetFunctionName(params.funcHandle, sizeof(kernelName), kernelName);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("get kernel name failed, ret: %d", ret);
        return false;
    }
    bool isBegin = (strcmp(kernelName, targetBeginName) == 0);
    bool isEnd = (strcmp(kernelName, targetEndName) == 0);
    bool isPlaceholder = (strcmp(kernelName, placeholderName) == 0);
    if (!isBegin && !isEnd && !isPlaceholder) {
        SK_LOGD("current kernel is not scope kernel, current kernel name is: %s", kernelName);
        return false;
    }
    auto parseArgsAddr = std::make_unique<ScopeKernelArgs>();
    ret = aclrtMemcpy((void*)parseArgsAddr.get(), sizeof(ScopeKernelArgs), params.args, sizeof(ScopeKernelArgs), 
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("aclrtMemcpy failed, ret: %d", ret);
        return false;
    }
    parseArgsAddr->name[MAX_SCOPE_NAME_LENN - 1] = '\0';
    size_t nameLen = strlen(parseArgsAddr->name);
    info->scopeName = std::make_unique<char[]>(nameLen + 1);
    errno_t res = memcpy_s(info->scopeName.get(), nameLen + 1, parseArgsAddr->name, nameLen + 1);
    if (res != 0) {
        SK_LOGE("memcpy_s failed, ret: %d", res);
        return false;
    }
    info->isBegin = isBegin;
    info->isEnd = isEnd;
    info->isPlaceholder = isPlaceholder;
    if (strcmp(info->scopeName.get(), defaultScopeName) == 0) {
        info->isFuseEnable = false;
    }
    SK_LOGD("Success parse scope kernel task, kernelName: %s, scopeName: %s, isBegin: %d, isFuseEnable: %d", kernelName, 
        info->scopeName.get(), info->isBegin, info->isFuseEnable);
    return true;
}

bool SuperKernelKernelNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    nodeType = SkNodeType::NODE_KERNEL;
    aclError aclRet = aclmdlRITaskGetParams(*originTask, &taskParams);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to get kernel params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    JudgeTaskKernelInfo scopeKernelInfo;
    auto &kernelParams = taskParams.kernelTaskParams;
    if (IsScopeKernel(kernelParams, &scopeKernelInfo)){
        SK_LOGI("Kernel node for task %u in stream %u is scope kernel.", nodeIdxInStream, streamIdxInGraph);
        isScopeNode = true;
        isFusible = scopeKernelInfo.isFuseEnable;
        isScopeBegin = scopeKernelInfo.isBegin;
        isScopeEnd = scopeKernelInfo.isEnd;
        isPlaceholder = scopeKernelInfo.isPlaceholder;
        if (isFusible && scopeKernelInfo.scopeName != nullptr){
            char* rawPtr = scopeKernelInfo.scopeName.get();
            scopeName = std::string(rawPtr);
        }
    } else {
        isFusible = true;
    }

    if (taskParams.taskGrp != nullptr) {
        SK_LOGI("Kernel task group is not null for task %u in stream %u, which cannot be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
        return true;
    }
    SK_LOGI("Kernel node for task %u in stream %u can be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);

    int64_t kernelType = 0;
    int64_t taskRatio = 0;
    CHECK_ACL(aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_TYPE, &kernelType));
    CHECK_ACL(aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_RATIO, &taskRatio));

    int16_t* taskRatioInt16 = (int16_t*)(&(taskRatio));
    uint32_t skTaskTatio[2] = {(uint32_t)(taskRatioInt16[0]), (uint32_t)(taskRatioInt16[1])};

    nodeInfos.kernelInfos.taskRatio[0] = skTaskTatio[0];
    nodeInfos.kernelInfos.taskRatio[1] = skTaskTatio[1];
    nodeInfos.kernelInfos.kernelType = NormalizeKernelType((uint32_t)(kernelType), skTaskTatio);
    nodeInfos.kernelInfos.numBlocks = kernelParams.numBlocks;
    nodeInfos.kernelInfos.devArgs = kernelParams.args;
    aclRet = aclrtFunctionGetBinary(kernelParams.funcHandle, &nodeInfos.kernelInfos.binHdl);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to get kernel bin handle");
        return false;
    }
    nodeInfos.kernelInfos.funcHdl = kernelParams.funcHandle;

    // Calculate vecNum and cubeNum based on kernel type and numBlocks
    uint32_t numBlocks = kernelParams.numBlocks;
    SkKernelType kt = nodeInfos.kernelInfos.kernelType;
    if (kt == SkKernelType::AIC_ONLY || kt == SkKernelType::MIX_AIC_1_0) {
        nodeInfos.kernelInfos.cubeNum = numBlocks;
        nodeInfos.kernelInfos.vecNum = 0;
    } else if (kt == SkKernelType::AIV_ONLY || kt == SkKernelType::MIX_AIV_1_0) {
        nodeInfos.kernelInfos.cubeNum = 0;
        nodeInfos.kernelInfos.vecNum = numBlocks;
    } else if (kt == SkKernelType::MIX_AIC_1_1) {
        nodeInfos.kernelInfos.cubeNum = numBlocks;
        nodeInfos.kernelInfos.vecNum = numBlocks;
    } else if (kt == SkKernelType::MIX_AIC_1_2) {
        nodeInfos.kernelInfos.cubeNum = numBlocks;
        nodeInfos.kernelInfos.vecNum = numBlocks << 1;
    }

    char tmpFuncName[256] = {0};
    CHECK_ACL(aclrtGetFunctionName(kernelParams.funcHandle, sizeof(tmpFuncName), tmpFuncName));
    nodeInfos.kernelInfos.funcName = std::string(tmpFuncName);
    if (!nodeInfos.kernelInfos.funcName.empty() && nodeInfos.kernelInfos.binHdl != nullptr) {
        InitKernelResolvedFuncs(nodeInfos.kernelInfos);
    }
    return true;
}

bool SuperKernelKernelNode::InValidateNode() {
    if (!isFusible) {
        SK_LOGE("Kernel node for task %u in stream %u can not be fused in super kernel, which should not been invalidated.", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    SK_LOGI("Invalidating kernel node for task %u in stream %u, which will be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
    aclError aclRet = aclmdlRITaskDisable(*originTask);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate kernel node for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelKernelNode::Update(const UpdateContext &ctx) {
    if (!isFusible) {
        SK_LOGE("Kernel node for task %u in stream %u is not fusible, update rejected.",
            nodeIdxInStream, streamIdxInGraph);
        return false;
    }

    if (ctx.customParams != nullptr) {
        // update kernel with custom params for stream sync
        aclError aclRet = aclmdlRITaskSetParams(*originTask, ctx.customParams);
        if (aclRet != ACL_SUCCESS) {
            SK_LOGE("Failed to set kernel with custom params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
            return false;
        }
        SK_LOGI("Success to set kernel with custom params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return true;
    } else if (ctx.launchInfo != nullptr && ctx.skEntryFunc != nullptr) {
        taskParams.kernelTaskParams.args = static_cast<void*>(ctx.launchInfo->devArgs.get());
        taskParams.kernelTaskParams.argsSize = ctx.launchInfo->devArgs.get()->skHeader.totalSize;
        taskParams.kernelTaskParams.isHostArgs = true;

        taskParams.kernelTaskParams.funcHandle = ctx.skEntryFunc;
        taskParams.kernelTaskParams.numBlocks = ctx.launchInfo->entryInfo.numBlocks;
        taskParams.type = ACL_MODEL_RI_TASK_KERNEL;

        aclError aclRet = aclmdlRITaskSetParams(*originTask, &taskParams);
        if (aclRet != ACL_SUCCESS) {
            SK_LOGE("Failed to update kernel node for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
            return false;
        }
        SK_LOGI("Updated kernel node for task %u in stream %u with argsHandle", nodeIdxInStream, streamIdxInGraph);
        return true;
    }
    return InValidateNode();
}

bool SuperKernelMemoryNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    aclError aclRet = aclmdlRITaskGetParams(*originTask, &taskParams);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to get event params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }

    if (rtNodeType != ACL_MODEL_RI_TASK_VALUE_WRITE && rtNodeType != ACL_MODEL_RI_TASK_VALUE_WAIT) {
        SK_LOGI("Event type is not memory based for task %u in stream %u, which cannot be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
        return true;
    }

    if (rtNodeType == ACL_MODEL_RI_TASK_VALUE_WRITE) {
        auto& memoryParam = taskParams.valueWriteTaskParams;
        if (memoryParam.value == 0) {
            nodeType = SkNodeType::NODE_RESET;
        }
        if (memoryParam.value == 1) {
            nodeType = SkNodeType::NODE_NOTIFY;
        }
        nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memoryParam.devAddr));
        nodeInfos.syncInfos.addrValue = memoryParam.devAddr;
    } else {
        nodeType = SkNodeType::NODE_WAIT;
        auto& memoryParam = taskParams.valueWaitTaskParams;
        nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memoryParam.devAddr));
        nodeInfos.syncInfos.addrValue = memoryParam.devAddr;
    }

    SK_LOGI("Event type of task %u in stream %u is memory based, which can be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
    isFusible = true;
    return true;
}

bool SuperKernelMemoryNode::Update(const UpdateContext &ctx) {
    if (!isFusible) {
        SK_LOGE("Memory node with eventId %lu for task %u in stream %u is not fusible, update rejected.",
                nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }

    if (ctx.customParams != nullptr) {
        // update memory node with custom params for stream sync
        aclError aclRet = aclmdlRITaskSetParams(*originTask, ctx.customParams);
        if (aclRet != ACL_SUCCESS) {
            SK_LOGE("Failed to set custom params on memory node for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
            return false;
        }
        SK_LOGI("Updated memory node via custom params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return true;
    }

    return InValidateNode();
}

bool SuperKernelMemoryNode::InValidateNode() {
    if (!isFusible) {
        SK_LOGE("Memory node with eventId %lu for task %u in stream %u can not be fused in super kernel, which should not been invalidated.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    SK_LOGI("Invalidating memory node with eventId %lu for task %u in stream %u, which will be fused in super kernel.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
    aclError aclRet = aclmdlRITaskDisable(*originTask);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate kernel node for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
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
