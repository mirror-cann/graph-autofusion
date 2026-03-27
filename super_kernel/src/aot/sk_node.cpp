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
#include <sstream>
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
    SkBindMap::iterator bindIt;
    size_t splitIdx;
};

enum class SkNodeCoreType: uint32_t {
    AIC,
    AIV,
};

SkBindMap InitSuperKernelBindMap(aclrtBinHandle binHdl)
{
    struct __attribute__((packed)) SknlValuePayload {
        uint32_t res;
        SknlMapInfo info;
    };
    constexpr size_t payloadSize = sizeof(SknlValuePayload);

    size_t metaNum = 0;
    if (int ret = rtBinaryGetMetaNum(binHdl, RT_BINARY_TYPE_SK_INFO, &metaNum) != 0) {
        SK_LOGI("rtBinaryGetMetaNum failed, ret=%d", ret);
        return SkBindMap();
    }
    if (metaNum == 0) {
        SK_LOGI("metaNum is zero!");
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

    if (int ret = rtBinaryGetMetaInfo(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, metaDataList.data(),
        infoSize.data()) != 0) {
        SK_LOGE("rtBinaryGetMetaInfo failed, ret=%d", ret);
        return SkBindMap();
    }

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

        auto it = bindMap.find((uint64_t)(localInfo.globalFunc));
        if (it != bindMap.end()) {
            SK_LOGE("InitSuperKernelBindMap: globalFunc=0x%lx is duplicated", (uint64_t)localInfo.globalFunc);
            continue;
        }
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
    constexpr size_t aicFuncMaxPrefetchCnt = 0x800 * 16; // 32k
    constexpr size_t aivFuncMaxPrefetchCnt = 0x800 * 8; // 16k
    constexpr size_t alignNum = 0x800; // 2k
    size_t prefetchCntValue = (value + alignNum - 1) & ~(alignNum - 1); // 以2k为单位向上取整
    if (coreIdx == 0 && prefetchCntValue > aicFuncMaxPrefetchCnt) {
        prefetchCntValue = aicFuncMaxPrefetchCnt;
    } else if (coreIdx == 1 && prefetchCntValue > aivFuncMaxPrefetchCnt) {
        prefetchCntValue = aivFuncMaxPrefetchCnt;
    }
    return prefetchCntValue / alignNum;
}

template <SkNodeCoreType coreType>
bool InitSingleCoreFunc(const CoreFuncInitContext& ctx, aclrtBinHandle binHdl, void *binDevAddr, uint32_t& validFuncNum)
{
    std::string coreName = "";
    if (coreType == SkNodeCoreType::AIC) {
        coreName = "AIC";
    } else {
        coreName = "AIV";
    }
    constexpr uint32_t coreTypeId = static_cast<uint32_t>(coreType); // 0 : aic, 1 : aiv
    uint64_t skFuncOffset = ctx.bindIt->second[ctx.splitIdx];
    ctx.info->funcAddr[coreTypeId] = skFuncOffset + (uint64_t)binDevAddr;
    void *binHostAddr = nullptr;
    uint32_t binHostSize = 0;
    if (int ret = rtGetBinBuffer(binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize) != 0) {
        SK_LOGE("split[%zu] rtGetBinBuffer failed for %s, ret=%d", ctx.splitIdx,
            coreName.c_str(), ret);
        return false;
    }
    std::string symbolName = "";
    uint64_t funcSize = 0;
    if (GetFuncSymbolInfo(static_cast<const char*>(binHostAddr), binHostSize, skFuncOffset,
                          symbolName, funcSize)) {
        ctx.info->prefetchCnt[coreTypeId] = AlignUpAndClamp(funcSize, coreTypeId);
        SK_LOGI("split[%zu] %s symbol=%s, size=0x%lx",
                ctx.splitIdx, coreName.c_str(), symbolName.c_str(), funcSize);
    } else {
        ctx.info->prefetchCnt[coreTypeId] = coreTypeId == 0 ? 16 : 8;
        SK_LOGW("split[%zu] Failed to get %s symbol info, default prefetchCnt[%zu]=%u",
                ctx.splitIdx, coreName.c_str(), coreTypeId, ctx.info->prefetchCnt[coreTypeId]);
    }
    if (ctx.splitIdx > 0 && ctx.bindIt->second[ctx.splitIdx] == ctx.bindIt->second[0]) {
        SK_LOGI("InitSingleCoreFunc: split[%zu] %s function is not sk sub op", ctx.splitIdx, coreName.c_str());
    } else {
        validFuncNum++;
    }
    return true;
}

bool InitSingleSplitFunc(ResolvedFunctionInfo &info, size_t splitIdx,
    const SkBindMap &bindMap, SkBindMap::iterator aicIt, SkBindMap::iterator aivIt,
    aclrtBinHandle binHdl, void *binDevAddr, uint32_t &resolvedNum)
{
    bool res = false;
    uint32_t validFuncNum = 0;
    if (aicIt != bindMap.end()) {
        CoreFuncInitContext aicCtx = {&info, aicIt, splitIdx};
        res |= InitSingleCoreFunc<SkNodeCoreType::AIC>(aicCtx, binHdl, binDevAddr, validFuncNum);
    }
    if (aivIt != bindMap.end()) {
        CoreFuncInitContext aivCtx = {&info, aivIt, splitIdx};
        res |= InitSingleCoreFunc<SkNodeCoreType::AIV>(aivCtx, binHdl, binDevAddr, validFuncNum);
    }
    if (!res) {
        SK_LOGE("Failed to initialize kernel function in sk Node split[%zu]", splitIdx);
        return false;
    }
    if (validFuncNum > 0) {
        resolvedNum++;
    }
    return true;
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
    if (bindMap.empty()) {
        SK_LOGI("bindMap is empty for kernel %s", kernelInfos.funcName.c_str());
        return false;
    }
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
    kernelInfos.resolvedNum = 0;
    for (size_t i = 0; i < kMaxSplitBinCount; ++i) {
        ResolvedFunctionInfo info{};
        if (!InitSingleSplitFunc(info, i, bindMap, aicItor, aivItor,
                            binHdl, binDevAddr, kernelInfos.resolvedNum)) {
            SK_LOGE("Failed to initialize kernel function in sk Node split[%zu]", i);
            return false;
        }
        kernelInfos.resolvedFuncs[i] = info;
        SK_LOGI("split[%zu] funcAddr[0]=0x%lx, funcAddr[1]=0x%lx, "
                "prefetchCnt[0]=0x%lx, prefetchCnt[1]=0x%lx",
                i, info.funcAddr[0], info.funcAddr[1], info.prefetchCnt[0], info.prefetchCnt[1]);
    }
    if (kernelInfos.resolvedNum == 2) { // sk balance
        kernelInfos.resolvedFuncs[2] = kernelInfos.resolvedFuncs[0];
        kernelInfos.resolvedFuncs[3] = kernelInfos.resolvedFuncs[1];
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
        SK_LOGE("Origin task is null for %s", Format().c_str());
        return false;
    }
    uint32_t seqId = 0;
    if (aclmdlRITaskGetSeqId(*originTask, &seqId) != ACL_SUCCESS) {
        SK_LOGE("Failed to get nodeId for %s", Format().c_str());
        return false;
    }
    nodeId = static_cast<uint64_t>(seqId);
    return true;
}

bool SuperKernelBaseNode::Update(const UpdateContext &ctx) {
    if (isUpdate) {
        SK_LOGE("Task has already been updated, can't update again, %s", Format().c_str());
        return false;
    }

    isUpdate = true;
    SK_LOGI("Updating node for task : %lu.", nodeId);
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
    int32_t ret = aclrtGetFunctionName(params.funcHandle, sizeof(kernelName), kernelName);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get kernel name for funcHandle, ret: %d", ret);
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
        SK_LOGE("Failed to copy kernel args from device to host, ret: %d, direction=DEVICE_TO_HOST", ret);
        return false;
    }
    parseArgsAddr->name[MAX_SCOPE_NAME_LENN - 1] = '\0';
    size_t nameLen = strlen(parseArgsAddr->name);
    info->scopeName = std::make_unique<char[]>(nameLen + 1);
    errno_t res = memcpy_s(info->scopeName.get(), nameLen + 1, parseArgsAddr->name, nameLen + 1);
    if (res != 0) {
        SK_LOGE("Failed to copy scope name '%s', memcpy_s error code: %d", parseArgsAddr->name, res);
        return false;
    }
    info->isBegin = isBegin;
    info->isEnd = isEnd;
    info->isPlaceholder = isPlaceholder;
    if (strcmp(info->scopeName.get(), defaultScopeName) == 0) {
        info->isFuseEnable = false;
    }
    SK_LOGI("Success parse scope kernel task, kernelName: %s, scopeName: %s, isBegin: %d, isFuseEnable: %d", kernelName,
        info->scopeName.get(), info->isBegin, info->isFuseEnable);
    return true;
}

bool SuperKernelKernelNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        SK_LOGE("Failed to init kernel node for %s", Format().c_str());
        return false;
    }
    nodeType = SkNodeType::NODE_KERNEL;
    aclError aclRet = aclmdlRITaskGetParams(*originTask, &taskParams);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to get kernel params for %s", Format().c_str());
        return false;
    }
    JudgeTaskKernelInfo scopeKernelInfo;
    auto &kernelParams = taskParams.kernelTaskParams;
    if (IsScopeKernel(kernelParams, &scopeKernelInfo)){
        SK_LOGI("Kernel node for task %lu is scope kernel.", nodeId);
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
        SK_LOGI("Kernel task group is not null for task %lu, which cannot be fused in super kernel.", nodeId);
        return true;
    }
    int64_t kernelType = 0;
    int64_t taskRatio = 0;
    CHECK_ACL(aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_TYPE, &kernelType));
    CHECK_ACL(aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_RATIO, &taskRatio));

    const int16_t* taskRatioInt16 = reinterpret_cast<const int16_t*>(&taskRatio);
    uint32_t skTaskTatio[2] = {static_cast<uint32_t>(taskRatioInt16[1]), static_cast<uint32_t>(taskRatioInt16[0])};

    nodeInfos.kernelInfos.taskRatio[0] = skTaskTatio[0];
    nodeInfos.kernelInfos.taskRatio[1] = skTaskTatio[1];
    nodeInfos.kernelInfos.kernelType = NormalizeKernelType((uint32_t)(kernelType), skTaskTatio);
    nodeInfos.kernelInfos.numBlocks = kernelParams.numBlocks;
    nodeInfos.kernelInfos.devArgs = kernelParams.args;
    nodeInfos.kernelInfos.opInfoPtr = taskParams.opInfoPtr;
    nodeInfos.kernelInfos.opInfoSize = taskParams.opInfoSize;
    aclRet = aclrtFunctionGetBinary(kernelParams.funcHandle, &nodeInfos.kernelInfos.binHdl);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to get kernel bin handle for %s, ret=%d",
                 Format().c_str(), aclRet);
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
    if (!isScopeNode && !nodeInfos.kernelInfos.funcName.empty() && nodeInfos.kernelInfos.binHdl != nullptr) {
        isFusible = InitKernelResolvedFuncs(nodeInfos.kernelInfos);
    }
    return true;
}

bool SuperKernelKernelNode::InValidateNode() {
    SK_LOGI("Invalidating kernel node for task %lu, which will be fused in super kernel.", nodeId);
    aclError aclRet = aclmdlRITaskDisable(*originTask);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate kernel node %s", Format().c_str());
        return false;
    }
    SK_LOGI("Invalidated kernel node for task %lu successfully", nodeId);
    return true;
}

std::string SuperKernelKernelNode::Format() const {
    std::ostringstream oss;
    oss << "[nodeId:" << nodeId 
        << ", streamId:" << streamId 
        << ", streamIdxInGraph:" << streamIdxInGraph 
        << ", nodeIdxInStream:" << nodeIdxInStream 
        << "] - " << nodeInfos.kernelInfos.Format();
    return oss.str();
}

std::string KernelInfos::Format() const {
    std::ostringstream oss;
    oss << "KernelInfos{funcName:" << funcName
        << ", kernelType:" << to_string(kernelType)
        << ", taskRatio:[" << taskRatio[0] << "," << taskRatio[1] << "]"
        << ", numBlocks:" << numBlocks
        << ", cubeNum:" << cubeNum
        << ", vecNum:" << vecNum
        << ", resolvedNum:" << resolvedNum;
    if (binHdl != nullptr) {
        oss << ", binHdl:0x" << std::hex << reinterpret_cast<uintptr_t>(binHdl) << std::dec;
    }
    if (funcHdl != nullptr) {
        oss << ", funcHdl:0x" << std::hex << reinterpret_cast<uintptr_t>(funcHdl) << std::dec;
    }
    if (devArgs != nullptr) {
        oss << ", devArgs:0x" << std::hex << reinterpret_cast<uintptr_t>(devArgs) << std::dec;
    }
    oss << "}";
    return oss.str();
}

bool SuperKernelKernelNode::Update(const UpdateContext &ctx) {
    if (!SuperKernelBaseNode::Update(ctx)) {
        SK_LOGE("Failed to update base node for %s", Format().c_str());
        return false;
    }

    if (ctx.customParams != nullptr && ctx.customParams->type != 0) {
        // check update value
        switch (ctx.customParams->type) {
            case ACL_MODEL_RI_TASK_VALUE_WRITE:
            case ACL_MODEL_RI_TASK_VALUE_WAIT:
                if (ctx.customParams->valueWriteTaskParams.devAddr == nullptr) {
                    SK_LOGE("Custom params for kernel node %s has null devAddr, invalid params.", Format().c_str());
                    return false;
                }
                break;
            default:
                SK_LOGI("custom param type : %u not in check list, which will direct update, %s", ctx.customParams->type, Format().c_str());
                break;
        }
        // update kernel with custom params for stream sync
        aclError aclRet = aclmdlRITaskSetParams(*originTask, ctx.customParams);
        if (aclRet != ACL_SUCCESS) {
            SK_LOGE("Failed to set kernel with custom params for %s", Format().c_str());
            return false;
        }
        SK_LOGI("Success to set kernel with custom params for task %lu", nodeId);
        return true;
    } else if (ctx.launchInfo != nullptr && ctx.launchInfo->entryInfo.skEntryFunc != nullptr) {
        taskParams.kernelTaskParams.args = static_cast<void*>(ctx.launchInfo->devArgs.Get());
        taskParams.kernelTaskParams.argsSize = ctx.launchInfo->devArgs.Get()->skHeader.totalSize;
        taskParams.kernelTaskParams.isHostArgs = true;

        taskParams.kernelTaskParams.funcHandle = ctx.launchInfo->entryInfo.skEntryFunc;
        taskParams.kernelTaskParams.numBlocks = ctx.launchInfo->entryInfo.numBlocks;
        taskParams.type = ACL_MODEL_RI_TASK_KERNEL;
        taskParams.opInfoPtr = ctx.launchInfo->cacheInfo;
        taskParams.opInfoSize = ctx.launchInfo->cacheopInfoSize;

        aclError aclRet = aclmdlRITaskSetParams(*originTask, &taskParams);

        if (aclRet != ACL_SUCCESS) {
            SK_LOGE("Failed to update kernel node %s", Format().c_str());
            return false;
        }
        SK_LOGI("Updated kernel node for task %lu with argsHandle", nodeId);
        return true;
    }
    return InValidateNode();
}

bool SuperKernelMemoryNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        SK_LOGE("Failed to init memory node for %s", Format().c_str());
        return false;
    }
    aclError aclRet = aclmdlRITaskGetParams(*originTask, &taskParams);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to get event params for %s", Format().c_str());
        return false;
    }

    if (rtNodeType != ACL_MODEL_RI_TASK_VALUE_WRITE && rtNodeType != ACL_MODEL_RI_TASK_VALUE_WAIT) {
        switch (rtNodeType) {
            case ACL_MODEL_RI_TASK_EVENT_RECORD: {
                auto &eventParam = taskParams.eventRecordTaskParams;
                nodeType = SkNodeType::NODE_NOTIFY;
                nodeInfos.syncInfos.eventId = (uint64_t)eventParam.event;
                nodeInfos.syncInfos.eventFlag = eventParam.eventFlag;
                break;
            }
            case ACL_MODEL_RI_TASK_EVENT_WAIT: {
                auto &eventParam = taskParams.eventWaitTaskParams;
                nodeType = SkNodeType::NODE_WAIT;
                nodeInfos.syncInfos.eventId = (uint64_t)eventParam.event;
                nodeInfos.syncInfos.eventFlag = eventParam.eventFlag;
                break;
            }
            case ACL_MODEL_RI_TASK_EVENT_RESET: {
                auto &eventParam = taskParams.eventResetTaskParams;
                nodeType = SkNodeType::NODE_RESET;
                nodeInfos.syncInfos.eventId = (uint64_t)eventParam.event;
                nodeInfos.syncInfos.eventFlag = eventParam.eventFlag;
                break;
            }
            default:
                SK_LOGE("Unsupported event type %u for %s, which cannot be fused in super kernel.", rtNodeType, Format().c_str());
                return false;
        }
        if ((rtNodeType != ACL_MODEL_RI_TASK_EVENT_RESET) && ((nodeInfos.syncInfos.eventFlag & ACL_EVENT_EXTERNAL) == 0)) {
            isFusible = true;
            SK_LOGI("Event %s: internal to ModelRI, fusible in super kernel",
                    Format().c_str());
        } else {
            SK_LOGI("Event %s: has external dependencies, cannot be fused in super kernel",
                    Format().c_str());
        }

        return true;
    }

    if (rtNodeType == ACL_MODEL_RI_TASK_VALUE_WRITE) {
        auto& memoryParam = taskParams.valueWriteTaskParams;
        nodeType = SkNodeType::NODE_MEMORY_WRITE;
        nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memoryParam.devAddr));
        nodeInfos.syncInfos.addrValue = memoryParam.devAddr;
        nodeInfos.syncInfos.memoryValue = memoryParam.value;
        isFusible = true;
    } else {
        nodeType = SkNodeType::NODE_MEMORY_WAIT;
        auto& memoryParam = taskParams.valueWaitTaskParams;
        nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memoryParam.devAddr));
        nodeInfos.syncInfos.addrValue = memoryParam.devAddr;
        nodeInfos.syncInfos.memoryValue = memoryParam.value;
        nodeInfos.syncInfos.memoryWaitFlag = memoryParam.flag;
        isFusible = false;
    }

    SK_LOGI("Event type of task %lu is memory based, which can be fused in super kernel.", nodeId);

    return true;
}

bool SuperKernelMemoryNode::Update(const UpdateContext &ctx) {
    if (!SuperKernelBaseNode::Update(ctx)) {
        SK_LOGE("Failed to update base node for %s", Format().c_str());
        return false;
    }

    if (ctx.customParams != nullptr && ctx.customParams->type != 0) {
        // check update value
        switch (ctx.customParams->type) {
            case ACL_MODEL_RI_TASK_VALUE_WRITE:
            case ACL_MODEL_RI_TASK_VALUE_WAIT:
                if (ctx.customParams->valueWriteTaskParams.devAddr == nullptr) {
                    SK_LOGE("Custom params for memory node %s has null devAddr, invalid params.", Format().c_str());
                    return false;
                }
                break;
            default:
                SK_LOGI("custom param type : %u not in check list, which will direct update, %s", ctx.customParams->type, Format().c_str());
                break;
        }
        // update memory node with custom params for stream sync
        aclError aclRet = aclmdlRITaskSetParams(*originTask, ctx.customParams);
        if (aclRet != ACL_SUCCESS) {
            SK_LOGE("Failed to set custom params on memory node %s", Format().c_str());
            return false;
        }
        SK_LOGI("Updated memory node via custom params for task %lu", nodeId);
        return true;
    }

    return InValidateNode();
}

bool SuperKernelMemoryNode::InValidateNode() {
    SK_LOGI("Invalidating memory node with eventId %lu for task %lu, which will be fused in super kernel.", nodeInfos.syncInfos.eventId, nodeId);
    aclError aclRet = aclmdlRITaskDisable(*originTask);
    if (aclRet != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate memory node %s", Format().c_str());
        return false;
    }
    return true;
}

std::string SuperKernelMemoryNode::Format() const {
    std::ostringstream oss;
    const char* eventType = nullptr;
    switch (rtNodeType) {
        case ACL_MODEL_RI_TASK_EVENT_RECORD:
            eventType = "EventNotify";
            break;
        case ACL_MODEL_RI_TASK_EVENT_WAIT:
            eventType = "EventWait";
            break;
        case ACL_MODEL_RI_TASK_EVENT_RESET:
            eventType = "EventReset";
            break;
        case ACL_MODEL_RI_TASK_VALUE_WRITE:
            oss << "[nodeId:" << nodeId
                << ", streamId:" << streamId
                << ", streamIdxInGraph:" << streamIdxInGraph
                << ", nodeIdxInStream:" << nodeIdxInStream
                << ", MemoryWrite(value:0x" << std::hex << nodeInfos.syncInfos.memoryValue
                << std::dec << ", eventId:0x" << std::hex << GetEventId() << std::dec << ")]";
            return oss.str();
        case ACL_MODEL_RI_TASK_VALUE_WAIT:
            oss << "[nodeId:" << nodeId
                << ", streamId:" << streamId
                << ", streamIdxInGraph:" << streamIdxInGraph
                << ", nodeIdxInStream:" << nodeIdxInStream
                << ", MemoryWait(flag:0x" << std::hex << nodeInfos.syncInfos.memoryWaitFlag
                << ", value:0x" << std::hex << nodeInfos.syncInfos.memoryValue
                << std::dec << ", eventId:0x" << std::hex << GetEventId() << std::dec << ")]";
            return oss.str();
        default:
            eventType = "Unknown";
            break;
    }
    uint64_t eventId = GetEventId();
    uint64_t eventFlag = nodeInfos.syncInfos.eventFlag;

    oss << "[nodeId:" << nodeId
        << ", streamId:" << streamId
        << ", streamIdxInGraph:" << streamIdxInGraph
        << ", nodeIdxInStream:" << nodeIdxInStream
        << ", " << eventType << "(eventId:0x" << std::hex << eventId
        << ", eventFlag:0x" << eventFlag << std::dec << ")]";

    return oss.str();
}

bool SuperKernelDefaultNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        SK_LOGE("Failed to init default node for %s", Format().c_str());
        return false;
    }
    nodeType = SkNodeType::NODE_DEFAULT;
    SK_LOGI("Default task type for task %lu, which cannot be fused in super kernel.", nodeId);
    return true;
}

bool SuperKernelDefaultNode::InValidateNode() {
    SK_LOGE("Default task type for %s should not be invalidated.", Format().c_str());
    return false;
}

std::string SuperKernelDefaultNode::Format() const {
    std::ostringstream oss;
    oss << "[nodeId:" << nodeId
        << ", streamId:" << streamId
        << ", streamIdxInGraph:" << streamIdxInGraph
        << ", nodeIdxInStream:" << nodeIdxInStream
        << ", type: Default]";
    return oss.str();
}
