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
* \file sk_dfx_exception_handler.cpp
* \brief SuperKernel DFX, handle sk aicore error
*/

#include <cstdarg>
#include <cstring>
#include <set>
#include <map>
#include "sk_dfx_exception_handler.h"
#include "sk_log.h"
#include "sk_types.h"
#include "sk_event_recorder.h"
#include "runtime/kernel.h"

SuperKernelExceptionHandler::SuperKernelExceptionHandler()
    : aicoreNums(DEFAULT_COUNTER_COUNT)
    , skDeviceEntryArgsDev(nullptr)
    , skDeviceEntryArgsPtrLen(0)
    , skDeviceEntryArgsHost(nullptr)
    , skHeaderInfoHost(nullptr)
    , aicTaskQueDevPtr(nullptr)
    , aivTaskQueDevPtr(nullptr)
    , aicTaskCnt(0)
    , aivTaskCnt(0)
    , hasOpTrace_(false)
{
}

void SuperKernelExceptionHandler::HandleException(aclrtExceptionInfo *exceptionInfo) {
    if (exceptionInfo == nullptr) {
        SK_LOGE("Exception info is null");
        return;
    }
    
    // Check if the exception is from sk_entry operator
    if (!IsSuperKernelException(exceptionInfo)) {
        SK_LOGD("Exception is not from sk_entry, skip handling.");
        return;
    }
    
    SK_LOGD("Aclgraph superkernel aicore exception occurred, callback function.");

    if (!ExtractSkEntryArgs(exceptionInfo)) {
        FreeResources();
        return;
    }

    if (!ExtractTaskQueue()) {
        FreeResources();
        return;
    }

    // Print info log
    ExtractAndPrintSkInfo();

    // Print error log
    if (!ParseAndPrintSubKernelSymbols(exceptionInfo)) {
        FreeResources();
        return;
    }

    // Print modelRI index mapping table for debugging
    SkEventRecorder::Instance().PrintModelRIIndexMap();

    // Print op trace for all cores
    PrintAllCoreSymbols();

    FreeResources();
}

bool SuperKernelExceptionHandler::ExtractSkEntryArgs(aclrtExceptionInfo *exceptionInfo) {
    if (!ExtractSkDeviceEntryArgsPtr(exceptionInfo)) {
        return false;
    }

    if (!CopySkDeviceEntryArgsToHost()) {
        return false;
    }

    return true;
}

bool SuperKernelExceptionHandler::CopySkDeviceEntryArgsToHost() {
    // Step 1: First copy SkHeaderInfo to get totalSize
    if (!ExtractSkHeaderInfo()) {
        return false;
    }

    // Step 2: Now that we know totalSize, copy all SkDeviceEntryArgs data to host at once
    SK_LOGI("---Total SkDeviceEntryArgs size: %lu bytes", skHeaderInfoHost->totalSize);
    if (CheckError(aclrtMallocHost((void **)(&skDeviceEntryArgsHost), skHeaderInfoHost->totalSize),
                   "aclrtMallocHost for skDeviceEntryArgsHost") != ACL_SUCCESS) {
        aclrtFreeHost(skHeaderInfoHost);
        skHeaderInfoHost = nullptr;
        return false;
    }

    if (CheckError(aclrtMemcpy(skDeviceEntryArgsHost, skHeaderInfoHost->totalSize,
                                skDeviceEntryArgsDev, skHeaderInfoHost->totalSize,
                                ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy for skDeviceEntryArgs") != ACL_SUCCESS) {
        aclrtFreeHost(skHeaderInfoHost);
        skHeaderInfoHost = nullptr;
        return false;
    }

    // Free temporary SkHeaderInfo, as data is now in skDeviceEntryArgsHost
    if (skHeaderInfoHost != nullptr) {
        aclrtFreeHost(skHeaderInfoHost);
        skHeaderInfoHost = nullptr;
    }

    // Update skHeaderInfoHost to point to skHeaderInfo in skDeviceEntryArgsHost
    skHeaderInfoHost = &(skDeviceEntryArgsHost->skHeader);

    return true;
}

bool SuperKernelExceptionHandler::ExtractSkDeviceEntryArgsPtr(aclrtExceptionInfo *exceptionInfo) {
    auto ret = aclrtGetArgsFromExceptionInfo(exceptionInfo, &skDeviceEntryArgsDev, &skDeviceEntryArgsPtrLen);
    SK_LOGI("---skDeviceEntryArgsDev: %p", skDeviceEntryArgsDev);
    SK_LOGI("---skDeviceEntryArgsPtrLen: %d", skDeviceEntryArgsPtrLen);

    if (skDeviceEntryArgsPtrLen < 8) {
        SK_LOGI("no args, callback return");
        return false;
    }
    return true;
}

bool SuperKernelExceptionHandler::ExtractSkHeaderInfo() {
    // SkHeaderInfo is at the beginning of SkDeviceEntryArgs, allocate temporary memory to copy
    SkHeaderInfo *tempHeader = nullptr;
    if (CheckError(aclrtMallocHost((void **)(&tempHeader), sizeof(SkHeaderInfo)),
                   "aclrtMallocHost for temp SkHeaderInfo") != ACL_SUCCESS) {
        return false;
    }

    if (CheckError(aclrtMemcpy(tempHeader, sizeof(SkHeaderInfo),
                                skDeviceEntryArgsDev, sizeof(SkHeaderInfo),
                                ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy for SkHeaderInfo") != ACL_SUCCESS) {
        aclrtFreeHost(tempHeader);
        return false;
    }

    // Temporarily save SkHeaderInfo data
    skHeaderInfoHost = tempHeader;

    return true;
}

bool SuperKernelExceptionHandler::ExtractTaskQueue() {
    // Data is already in skDeviceEntryArgsHost, accessed via offset
    // Extract task counts
    uint8_t *dataBase = reinterpret_cast<uint8_t*>(skDeviceEntryArgsHost);

    if (skHeaderInfoHost->aicQueOffset > 0) {
        TaskQue *aicTaskQue = reinterpret_cast<TaskQue*>(dataBase + skHeaderInfoHost->aicQueOffset);
        aicTaskCnt = aicTaskQue->taskCnt;
        for (uint32_t i = 0; i < aicTaskCnt; ++i) {
            if (aicTaskQue->taskInfos[i].type == SkTaskType::TYPE_FUNC) {
                funcNodeIndices_.insert(aicTaskQue->taskInfos[i].index);
            }
        }
    }

    if (skHeaderInfoHost->aivQueOffset > 0) {
        TaskQue *aivTaskQue = reinterpret_cast<TaskQue*>(dataBase + skHeaderInfoHost->aivQueOffset);
        aivTaskCnt = aivTaskQue->taskCnt;
        for (uint32_t i = 0; i < aivTaskCnt; ++i) {
            if (aivTaskQue->taskInfos[i].type == SkTaskType::TYPE_FUNC) {
                funcNodeIndices_.insert(aivTaskQue->taskInfos[i].index);
            }
        }
    }

    return true;
}

void SuperKernelExceptionHandler::PrintSkHeaderInfo() const {
    SK_LOGI("=== SkHeaderInfo ===");
    SK_LOGI("aicQueOffset: %u", skHeaderInfoHost->aicQueOffset);
    SK_LOGI("aivQueOffset: %u", skHeaderInfoHost->aivQueOffset);
    SK_LOGI("counterOffset: %u", skHeaderInfoHost->counterOffset);
    SK_LOGI("dfxOffset: %u", skHeaderInfoHost->dfxOffset);
    SK_LOGI("nodeCnt: %u", skHeaderInfoHost->nodeCnt);
    SK_LOGI("totalSize: %lu", skHeaderInfoHost->totalSize);
    uint16_t modelRIIdx = static_cast<uint16_t>((skHeaderInfoHost->modelRIIdAndSkScopeId >> 32) & 0xFFFF);
    uint16_t skScopeId = static_cast<uint16_t>((skHeaderInfoHost->modelRIIdAndSkScopeId >> 16) & 0xFFFF);
    uint64_t originalModelRI = SkEventRecorder::Instance().GetModelRIByIndex(modelRIIdx);
    SK_LOGI("modelRIIdAndSkScopeId: 0x%lx (modelRIIdx=%u, skScopeId=%u, originalModelRI=0x%lx)",
            skHeaderInfoHost->modelRIIdAndSkScopeId, modelRIIdx, skScopeId, originalModelRI);
}

void SuperKernelExceptionHandler::ExtractAndPrintSkInfo() {
    PrintSkHeaderInfo();
    PrintTaskQueue();
    PrintCounterInfo();
    PrintDfxInfo();
}

void SuperKernelExceptionHandler::PrintTaskQueue() const {
    uint8_t *dataBase = reinterpret_cast<uint8_t*>(skDeviceEntryArgsHost);

    if (skHeaderInfoHost->aicQueOffset > 0) {
        TaskQue *aicTaskQue = reinterpret_cast<TaskQue*>(dataBase + skHeaderInfoHost->aicQueOffset);
        SK_LOGI("=== AIC TaskQue (offset=%u) ===", skHeaderInfoHost->aicQueOffset);
        SK_LOGI("taskCnt: %u", aicTaskCnt);
        for (uint32_t i = 0; i < aicTaskCnt; ++i) {
            const TaskInfo &task = aicTaskQue->taskInfos[i];
            SK_LOGI("  [%u] index=%u, type=%s, numBlocks=%u, args=0x%lx",
                    i, task.index, to_string(task.type), task.numBlocks, task.args);
        }
    }

    if (skHeaderInfoHost->aivQueOffset > 0) {
        TaskQue *aivTaskQue = reinterpret_cast<TaskQue*>(dataBase + skHeaderInfoHost->aivQueOffset);
        SK_LOGI("=== AIV TaskQue (offset=%u) ===", skHeaderInfoHost->aivQueOffset);
        SK_LOGI("taskCnt: %u", aivTaskCnt);
        for (uint32_t i = 0; i < aivTaskCnt; ++i) {
            const TaskInfo &task = aivTaskQue->taskInfos[i];
            SK_LOGI("  [%u] index=%u, type=%s, numBlocks=%u, args=0x%lx",
                    i, task.index, to_string(task.type), task.numBlocks, task.args);
        }
    }
}

void SuperKernelExceptionHandler::PrintCounterInfo() const {
    uint8_t *dataBase = reinterpret_cast<uint8_t*>(skDeviceEntryArgsHost);

    if (skHeaderInfoHost->counterOffset > 0) {
        SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo*>(dataBase + skHeaderInfoHost->counterOffset);
        SK_LOGI("=== SkCounterInfo (offset=%u) ===", skHeaderInfoHost->counterOffset);
        for (uint32_t i = 0; i < aicoreNums; ++i) {
            SK_LOGI("  [core %u] index=%u, opState=%u",
                    i, counterInfo[i].index, counterInfo[i].opState);
        }
    }
}

void SuperKernelExceptionHandler::PrintDfxInfo() const {
    uint8_t *dataBase = reinterpret_cast<uint8_t*>(skDeviceEntryArgsHost);

    if (skHeaderInfoHost->dfxOffset > 0) {
        SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo*>(dataBase + skHeaderInfoHost->dfxOffset);
        SK_LOGE("=== SkDfxInfo (offset=%u, nodeCnt=%u) ===",
                skHeaderInfoHost->dfxOffset, skHeaderInfoHost->nodeCnt);
        for (uint32_t i = 0; i < skHeaderInfoHost->nodeCnt; ++i) {
            SK_LOGE("  [node %u] binHdl=0x%lx, funcHdlOri=0x%lx, aicSize=0x%x, aivSize=0x%x, numBlocks=%u, cubeNum=%u, vecNum=%u",
                    i, dfxInfo[i].binHdl, dfxInfo[i].funcHdlOri,
                    dfxInfo[i].aicSize, dfxInfo[i].aivSize,
                    dfxInfo[i].numBlocks, dfxInfo[i].cubeNum, dfxInfo[i].vecNum);
            if (funcNodeIndices_.count(i) > 0) {
                aclrtFuncHandle funcHdl = reinterpret_cast<aclrtFuncHandle>(dfxInfo[i].funcHdlOri);
                char funcName[256] = {0};
                aclError ret = aclrtGetFunctionName(funcHdl, sizeof(funcName), funcName);
                if (ret == ACL_SUCCESS) {
                    SK_LOGE("    Origin function name: %s", funcName);
                } else {
                    SK_LOGE("    Failed to get origin function name for node[%u], ret=%d", i, ret);
                }
            } else {
                SK_LOGE("    Node[%u] is not a FUNC type", i);
            }
            for (int j = 0; j < 4; ++j) {
                if (dfxInfo[i].entryAic[j] != 0) {
                    SK_LOGE("    entryAic[%d]=0x%lx", j, dfxInfo[i].entryAic[j]);
                }
                if (dfxInfo[i].entryAiv[j] != 0) {
                    SK_LOGE("    entryAiv[%d]=0x%lx", j, dfxInfo[i].entryAiv[j]);
                }
            }
        }
    }
}

bool SuperKernelExceptionHandler::GetExceptionRegInfo(const aclrtExceptionInfo &exception,
                                                       ExceptionRegInfo &exceptionRegInfo) {
    // errRegInfo memory allocated by rtGetExceptionRegInfo is managed by RTS, should not be manually freed
    rtError_t rtRet = rtGetExceptionRegInfo(&exception, &exceptionRegInfo.errRegInfo, &exceptionRegInfo.coreNum);
    if (rtRet != 0) {
        SK_LOGE("Call rtGetExceptionRegInfo for error register information failed. ret: %d", static_cast<int32_t>(rtRet));
        return false;
    }
    SK_LOGI("Get error register information. coreNum=%u", exceptionRegInfo.coreNum);
    return true;
}

uint64_t SuperKernelExceptionHandler::GetCondRegValue(const rtExceptionErrRegInfo_t &coreErrRegInfo) {
    // COND register is 64-bit: errReg[20] is low 32 bits, errReg[21] is high 32 bits
    constexpr uint32_t COND_REG_LOW_IDX = 20;
    constexpr uint32_t COND_REG_HIGH_IDX = 21;
    uint64_t condValue = (static_cast<uint64_t>(coreErrRegInfo.errReg[COND_REG_HIGH_IDX]) << 32)
                       | static_cast<uint64_t>(coreErrRegInfo.errReg[COND_REG_LOW_IDX]);
    return condValue;
}

void SuperKernelExceptionHandler::ParseAndPrintCondInfo(uint32_t coreId, rtCoreType_t coreType, uint64_t condValue) {
    // Cond value is set by sk_entry kernel via set_cond instruction.
    // Format: cond = modelRIIdAndSkScopeId | (opState + (task->index << 8))
    //   bits [7:0]   : opState (SkOpTraceType)
    //   bits [15:8]  : task->index (sub-operator index)
    //   bits [63:16] : modelRIIdAndSkScopeId
    // If cond is 0, the runtime/driver does not support this feature yet, skip printing.
    if (condValue == 0) {
        SK_LOGE("[Core %u] Failed to get COND register value, possible reason: driver package not upgraded", coreId);
        return;
    }

    uint8_t opState = static_cast<uint8_t>(condValue & 0xFF);
    uint32_t opIndex = static_cast<uint32_t>((condValue >> 8) & 0xFF);
    const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";

    SK_LOGE("[Core %u] %s COND register value: 0x%lx (opState=%u, opIndex=%u)",
            coreId, coreTypeName, condValue, opState, opIndex);

    PrintCondSubKernelInfo(coreId, condValue);
}

void SuperKernelExceptionHandler::PrintCondSubKernelInfo(uint32_t coreId, uint64_t condValue) {
    uint8_t opState = static_cast<uint8_t>(condValue & 0xFF);
    uint32_t opIndex = static_cast<uint32_t>((condValue >> 8) & 0xFF);

    if (opState == static_cast<uint8_t>(SkOpTraceType::ORIGIN)) {
        SK_LOGE("[Core %u] COND: No SK entry executed yet (opState=ORIGIN).", coreId);
    } else if (opState == static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_LAUNCHED)) {
        SK_LOGE("[Core %u] COND: SK entry launched, no sub-kernel executed yet. Next opIndex=%u", coreId, opIndex);
        if (opIndex < skHeaderInfoHost->nodeCnt) {
            KernelFuncName kernelFuncName = GetOrLoadKernelSymbols(opIndex);
            if (!kernelFuncName.name.empty()) {
                SK_LOGE("[Core %u] COND: Next sub-kernel function name: %s", coreId, kernelFuncName.name.c_str());
            }
        }
    } else if (opState == static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED)) {
        SK_LOGE("[Core %u] COND: Currently running sub-kernel opIndex=%u", coreId, opIndex);
        if (opIndex < skHeaderInfoHost->nodeCnt) {
            KernelFuncName kernelFuncName = GetOrLoadKernelSymbols(opIndex);
            if (!kernelFuncName.name.empty()) {
                SK_LOGE("[Core %u] COND: Current sub-kernel function name: %s", coreId, kernelFuncName.name.c_str());
            }
        }
    } else if (opState == static_cast<uint8_t>(SkOpTraceType::OP_FINISHED)) {
        SK_LOGE("[Core %u] COND: Sub-kernel opIndex=%u finished, about to run next.", coreId, opIndex);
        if (opIndex < skHeaderInfoHost->nodeCnt) {
            KernelFuncName currentKernelFuncName = GetOrLoadKernelSymbols(opIndex);
            if (!currentKernelFuncName.name.empty()) {
                SK_LOGE("[Core %u] COND: Last finished sub-kernel function name: %s", coreId, currentKernelFuncName.name.c_str());
            }
        }
        if (opIndex + 1 < skHeaderInfoHost->nodeCnt) {
            KernelFuncName nextKernelFuncName = GetOrLoadKernelSymbols(opIndex + 1);
            if (!nextKernelFuncName.name.empty()) {
                SK_LOGE("[Core %u] COND: Next sub-kernel function name: %s", coreId, nextKernelFuncName.name.c_str());
            }
        }
    } else if (opState == static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_FINISHED)) {
        SK_LOGE("[Core %u] COND: SK entry execution completed.", coreId);
    } else {
        SK_LOGE("[Core %u] COND: Unknown opState=%u, opIndex=%u", coreId, opState, opIndex);
    }
}

void SuperKernelExceptionHandler::PrintSymbolByCoreId(uint32_t coreId, rtCoreType_t coreType,
                                                     uint64_t startPC, uint64_t currentPC,
                                                     const KernelFuncName &kernelFuncName) {
    const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";

    SK_LOGI("========== coreId=%u, coreType=%s, startPC=0x%lx, currentPC=0x%lx ==========",
            coreId, coreTypeName, startPC, currentPC);

    // Print function name
    if (!kernelFuncName.name.empty()) {
        SK_LOGE("    Function name: %s", kernelFuncName.name.c_str());
    } else {
        SK_LOGE("    No function name available");
    }
}

KernelFuncName SuperKernelExceptionHandler::GetOrLoadKernelSymbols(uint32_t opId) {
    // Check cache
    auto it = opSymbolCache.find(opId);
    if (it != opSymbolCache.end()) {
        SK_LOGI("Using cached function name for opId=%u", opId);
        return it->second;
    }

    // Cache miss, get function name from funcHdl
    if (opId >= skHeaderInfoHost->nodeCnt) {
        SK_LOGE("opId=%u exceeds nodeCnt=%u", opId, skHeaderInfoHost->nodeCnt);
        return KernelFuncName{""};
    }

    // Only TYPE_FUNC nodes have valid function handles
    if (funcNodeIndices_.count(opId) == 0) {
        SK_LOGI("opId=%u is not a FUNC type node", opId);
        return KernelFuncName{""};
    }

    // Access dfxInfo via offset from skDeviceEntryArgsHost
    uint8_t *dataBase = reinterpret_cast<uint8_t*>(skDeviceEntryArgsHost);
    SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo*>(dataBase + skHeaderInfoHost->dfxOffset);

    aclrtFuncHandle funcHdl = reinterpret_cast<aclrtFuncHandle>(dfxInfo[opId].funcHdlOri);
    SK_LOGI("Loading function name for opId=%u, funcHdl=%p", opId, funcHdl);

    char funcName[256] = {0};
    constexpr uint32_t maxLen = sizeof(funcName);
    aclError ret = aclrtGetFunctionName(funcHdl, maxLen, funcName);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("aclrtGetFunctionName failed for opId=%u, funcHdl=%p, ret=%d", opId, funcHdl, ret);
        return KernelFuncName{""};
    }

    SK_LOGI("Got function name for opId=%u: %s", opId, funcName);

    // Construct simple KernelFuncName, only containing function name
    KernelFuncName kernelFuncName{std::string(funcName)};

    // Store in cache
    opSymbolCache[opId] = kernelFuncName;
    return kernelFuncName;
}

void SuperKernelExceptionHandler::PrintMatchedNodeBasicInfo(
        uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC,
        uint32_t nodeIdx, int entryIdx, uint64_t entryAddr, uint64_t endAddr,
        uint32_t funcSize, const SkDfxInfo &dfxNode) const {
    const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";
    uint16_t modelRIIdx = static_cast<uint16_t>((skHeaderInfoHost->modelRIIdAndSkScopeId >> 32) & 0xFFFF);
    uint16_t skScopeId = static_cast<uint16_t>((skHeaderInfoHost->modelRIIdAndSkScopeId >> 16) & 0xFFFF);
    uint64_t originalModelRI = SkEventRecorder::Instance().GetModelRIByIndex(modelRIIdx);

    SK_LOGE("============================================================");
    SK_LOGE("[Core %u] ModelRIIdx=%u, OriginalModelRI=0x%lx, SkScopeId=%u", coreId, modelRIIdx, originalModelRI, skScopeId);
    SK_LOGE("[Core %u] CoreType: %s", coreId, coreTypeName);
    SK_LOGE("[Core %u] StartPC: 0x%lx", coreId, startPC);
    SK_LOGE("[Core %u] CurrentPC: 0x%lx", coreId, currentPC);
    SK_LOGE("[Core %u] Found in node[%u], entry[%d]", coreId, nodeIdx, entryIdx);
    SK_LOGE("[Core %u] Entry address: 0x%lx", coreId, entryAddr);
    SK_LOGE("[Core %u] End address: 0x%lx", coreId, endAddr);
    SK_LOGE("[Core %u] Function size: 0x%x (%u bytes)", coreId, funcSize, funcSize);
    SK_LOGE("[Core %u] numBlocks=%u, cubeNum=%u, vecNum=%u", coreId,
            dfxNode.numBlocks, dfxNode.cubeNum, dfxNode.vecNum);
}

void SuperKernelExceptionHandler::PrintFuncSymbolInfo(
        uint32_t coreId, rtCoreType_t coreType, uint32_t nodeIdx, int entryIdx,
        const uint64_t* entries, const SkDfxInfo &dfxNode) {
    if (funcNodeIndices_.count(nodeIdx) == 0) {
        SK_LOGE("[Core %u] Node[%u] is not a FUNC type", coreId, nodeIdx);
        return;
    }

    // Print origin function name from funcHdlOri
    aclrtFuncHandle funcHdl = reinterpret_cast<aclrtFuncHandle>(dfxNode.funcHdlOri);
    char funcName[256] = {0};
    aclError ret = aclrtGetFunctionName(funcHdl, sizeof(funcName), funcName);
    if (ret == ACL_SUCCESS) {
        SK_LOGE("[Core %u] Origin function name: %s", coreId, funcName);
    } else {
        SK_LOGE("Failed to get origin function name for node[%u], ret=%d", nodeIdx, ret);
    }

    // Resolve actual SK symbol name from bin using funcOffset
    aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(dfxNode.binHdl);
    void *binHostAddr = nullptr;
    uint32_t binHostSize = 0;
    int rtRet = rtGetBinBuffer(binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize);
    if (rtRet != 0 || binHostAddr == nullptr) {
        SK_LOGE("[Core %u] Failed to get bin buffer for node[%u], rtRet=%d", coreId, nodeIdx, rtRet);
        return;
    }

    uint64_t funcOffset = (coreType == RT_CORE_TYPE_AIC)
        ? dfxNode.aicFuncOffset[entryIdx] : dfxNode.aivFuncOffset[entryIdx];
    // Fallback: compute offset from entry address using AIC entry[0] as reference
    if (funcOffset == 0 && dfxNode.aicFuncOffset[0] != 0 && dfxNode.entryAic[0] != 0) {
        uint64_t binDevAddr = dfxNode.entryAic[0] - dfxNode.aicFuncOffset[0];
        funcOffset = entries[entryIdx] - binDevAddr;
        SK_LOGI("[Core %u] Computed funcOffset=0x%lx for node[%u] entry[%d] (binDevAddr=0x%lx)",
                coreId, funcOffset, nodeIdx, entryIdx, binDevAddr);
    }
    if (funcOffset == 0) {
        SK_LOGE("[Core %u] funcOffset is 0 for node[%u] entry[%d], cannot resolve SK symbol name",
                coreId, nodeIdx, entryIdx);
        return;
    }

    std::string symbolName;
    uint64_t symbolSize = 0;
    std::string symbolBind;
    if (GetFuncSymbolInfo(binHdl, static_cast<const char*>(binHostAddr), binHostSize,
                          funcOffset, symbolName, symbolSize, symbolBind)) {
        SK_LOGE("[Core %u] Bound SK symbol name: %s (bind=%s, size=0x%lx)",
                coreId, symbolName.c_str(), symbolBind.c_str(), symbolSize);
    } else {
        SK_LOGE("[Core %u] Failed to get SK symbol name for node[%u] entry[%d], offset=0x%lx",
                coreId, nodeIdx, entryIdx, funcOffset);
    }
}

void SuperKernelExceptionHandler::PrintNodeDevArgs(
        uint32_t coreId, rtCoreType_t coreType, uint32_t nodeIdx) const {
    uint8_t *dataBase = reinterpret_cast<uint8_t*>(skDeviceEntryArgsHost);
    TaskQue *taskQue = (coreType == RT_CORE_TYPE_AIC)
        ? reinterpret_cast<TaskQue*>(dataBase + skHeaderInfoHost->aicQueOffset)
        : reinterpret_cast<TaskQue*>(dataBase + skHeaderInfoHost->aivQueOffset);
    uint32_t taskCnt = (coreType == RT_CORE_TYPE_AIC) ? aicTaskCnt : aivTaskCnt;
    for (uint32_t t = 0; t < taskCnt; ++t) {
        if (taskQue->taskInfos[t].index == nodeIdx && taskQue->taskInfos[t].type == SkTaskType::TYPE_FUNC) {
            SK_LOGE("[Core %u] node[%u] devArgs: 0x%lx", coreId, nodeIdx, taskQue->taskInfos[t].args);
            break;
        }
    }
}

void SuperKernelExceptionHandler::PrintNoMatchInfo(
        uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC) const {
    const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";
    uint16_t modelRIIdx = static_cast<uint16_t>((skHeaderInfoHost->modelRIIdAndSkScopeId >> 32) & 0xFFFF);
    uint16_t skScopeId = static_cast<uint16_t>((skHeaderInfoHost->modelRIIdAndSkScopeId >> 16) & 0xFFFF);
    uint64_t originalModelRI = SkEventRecorder::Instance().GetModelRIByIndex(modelRIIdx);

    SK_LOGE("============================================================");
    SK_LOGE("[Core %u] No sub kernel matched, aicore error occurred in sk entry.", coreId);
    SK_LOGE("[Core %u] ModelRIIdx=%u, OriginalModelRI=0x%lx, SkScopeId=%u", coreId, modelRIIdx, originalModelRI, skScopeId);
    SK_LOGE("[Core %u] CoreType: %s", coreId, coreTypeName);
    SK_LOGE("[Core %u] startPC: 0x%lx", coreId, startPC);
    SK_LOGE("[Core %u] CurrentPC: 0x%lx", coreId, currentPC);
    SK_LOGE("============================================================");
}

void SuperKernelExceptionHandler::IdentifyErrorNodeByPC(uint32_t coreId, rtCoreType_t coreType,
                                                     uint64_t startPC, uint64_t currentPC) {
    if (skHeaderInfoHost->dfxOffset == 0 || skHeaderInfoHost->nodeCnt == 0 || currentPC == 0) {
        return;
    }

    uint8_t *dataBase = reinterpret_cast<uint8_t*>(skDeviceEntryArgsHost);
    SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo*>(dataBase + skHeaderInfoHost->dfxOffset);

    // Iterate through all nodes to find which one's entry range contains currentPC
    for (uint32_t i = 0; i < skHeaderInfoHost->nodeCnt; ++i) {
        uint64_t* entries = (coreType == RT_CORE_TYPE_AIC) ? dfxInfo[i].entryAic : dfxInfo[i].entryAiv;
        uint32_t funcSize = (coreType == RT_CORE_TYPE_AIC) ? dfxInfo[i].aicSize : dfxInfo[i].aivSize;

        for (int j = 0; j < 4; ++j) {
            if (entries[j] == 0) {
                continue;  // Skip invalid entries
            }

            uint64_t entryAddr = entries[j];
            uint64_t endAddr = entryAddr + funcSize;

            // Check if currentPC falls within this entry's range [entryAddr, entryAddr + funcSize)
            if (currentPC >= entryAddr && currentPC < endAddr) {
                PrintMatchedNodeBasicInfo(coreId, coreType, startPC, currentPC, i, j, entryAddr, endAddr, funcSize, dfxInfo[i]);
                PrintFuncSymbolInfo(coreId, coreType, i, j, entries, dfxInfo[i]);
                PrintNodeDevArgs(coreId, coreType, i);
                SK_LOGE("============================================================");
                return;  // Found the error node
            }
        }
    }

    PrintNoMatchInfo(coreId, coreType, startPC, currentPC);
}

void SuperKernelExceptionHandler::PrintCoreSymbols(uint32_t coreId, rtCoreType_t coreType,
                                                  uint64_t startPC, uint64_t currentPC) {
    if (coreId >= aicoreNums) {
        SK_LOGE("coreId=%u exceeds aicoreNums=%u", coreId, aicoreNums);
        return;
    }

    if (!hasOpTrace_) {
        return;
    }

    // Access counterInfo via offset from skDeviceEntryArgsHost
    uint8_t *dataBase = reinterpret_cast<uint8_t*>(skDeviceEntryArgsHost);
    SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo*>(dataBase + skHeaderInfoHost->counterOffset);

    uint32_t opState = counterInfo[coreId].opState;
    uint32_t opId = counterInfo[coreId].index;
    const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";

    SK_LOGE("[Core %u] Type=%s, opState=%u, opId=%u",
            coreId, coreTypeName, opState, opId);

    if (opState == static_cast<uint8_t>(SkOpTraceType::ORIGIN)) {
        // args initialized to 0, if ORIGIN means no SK has been executed yet
        SK_LOGE("[Core %u] No SK entry executed yet.", coreId);
    } else if (opState == static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_LAUNCHED)) {
        // SK operator just started, no sub-kernel executed yet, print next operator symbols
        SK_LOGE("[Core %u] SK started but no sub-kernel executed yet, checking next operators", coreId);

        // Print next operator symbols (starting from first operator)
        if (opId < skHeaderInfoHost->nodeCnt) {
            SK_LOGE("[Core %u] > Next opId=%u (first sub-kernel):", coreId, opId);
            KernelFuncName nextKernelFuncName = GetOrLoadKernelSymbols(opId);
            PrintSymbolByCoreId(coreId, coreType, startPC, currentPC, nextKernelFuncName);
        } else {
            SK_LOGE("[Core %u] > No next operator (opId=%u/%u)", coreId, opId, skHeaderInfoHost->nodeCnt);
        }
    } else if (opState == static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED)) {
        // Sub-kernel is running, print current operator symbols
        SK_LOGE("[Core %u] Currently running opId=%u", coreId, opId);
        KernelFuncName kernelFuncName = GetOrLoadKernelSymbols(opId);
        PrintSymbolByCoreId(coreId, coreType, startPC, currentPC, kernelFuncName);
    } else if (opState == static_cast<uint8_t>(SkOpTraceType::OP_FINISHED)) {
        // Sub-kernel has finished execution, current opId is the last executed operator, print current and next operator symbols
        SK_LOGE("[Core %u] opId=%u finished, checking current and next operators", coreId, opId);

        // Print current finished operator symbols
        SK_LOGE("[Core %u] > Last finished opId=%u:", coreId, opId);
        KernelFuncName currentKernelFuncName = GetOrLoadKernelSymbols(opId);
        PrintSymbolByCoreId(coreId, coreType, startPC, currentPC, currentKernelFuncName);

        // Print next operator symbols (not exceeding nodeCnt)
        if (opId + 1 < skHeaderInfoHost->nodeCnt) {
            SK_LOGE("[Core %u] > Next opId=%u:", coreId, opId + 1);
            KernelFuncName nextKernelFuncName = GetOrLoadKernelSymbols(opId + 1);
            PrintSymbolByCoreId(coreId, coreType, startPC, currentPC, nextKernelFuncName);
        } else {
            SK_LOGE("[Core %u] > No next operator (opId=%u/%u)", coreId, opId, skHeaderInfoHost->nodeCnt);
        }
    } else if (opState == static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_FINISHED)) {
        // SK operator execution completed
        SK_LOGE("[Core %u] SK entry operator execution completed.", coreId);
    } else {
        SK_LOGE("[Core %u] Unknown opState: %u", coreId, opState);
    }
}

void SuperKernelExceptionHandler::PrintAllCoreSymbols() {
    if (!hasOpTrace_) {
        return;
    }

    SK_LOGE("==================================================");
    SK_LOGE("=== Sub-kernel running info on all %u cores ===", aicoreNums);
    SK_LOGE("==================================================");

    for (uint32_t coreId = 0; coreId < aicoreNums; coreId++) {
        // Determine core type
        rtCoreType_t coreType = (coreId < 25) ? RT_CORE_TYPE_AIC : RT_CORE_TYPE_AIV;

        // Temporarily use default values, as real PC values cannot be obtained from counter info
        uint64_t startPC = 0;
        uint64_t currentPC = 0;

        // Call common function to print symbols
        PrintCoreSymbols(coreId, coreType, startPC, currentPC);
    }
}

bool SuperKernelExceptionHandler::ParseAndPrintSubKernelSymbols(aclrtExceptionInfo *exceptionInfo) {
    // Get exception register information
    ExceptionRegInfo exceptionRegInfo{0, nullptr};
    if (!GetExceptionRegInfo(*reinterpret_cast<const aclrtExceptionInfo*>(exceptionInfo), exceptionRegInfo)) {
        // Note: errRegInfo pointer is allocated by RTS, no need to manually free
        return false;
    }

    SK_LOGE("====================================================");
    SK_LOGE("=== Sub-kernel running info on exception cores ===");
    SK_LOGE("====================================================");
    for (uint32_t i = 0; i < exceptionRegInfo.coreNum; i++) {
        rtExceptionErrRegInfo_t coreErrRegInfo = exceptionRegInfo.errRegInfo[i];
        rtCoreType_t coreType = static_cast<rtCoreType_t>(coreErrRegInfo.coreType);
        // Always try to identify error node by PC address first
        IdentifyErrorNodeByPC(coreErrRegInfo.coreId, coreType, coreErrRegInfo.startPC, coreErrRegInfo.currentPC);

        PrintCoreSymbols(coreErrRegInfo.coreId, coreType,
                        coreErrRegInfo.startPC,
                        coreErrRegInfo.currentPC);

        // Parse COND register value from errReg[20](low32) and errReg[21](high32)
        uint64_t condValue = GetCondRegValue(coreErrRegInfo);
        ParseAndPrintCondInfo(coreErrRegInfo.coreId, coreType, condValue);
    }

    return true;
}

void SuperKernelExceptionHandler::FreeResources() {
    if (skDeviceEntryArgsHost != nullptr) {
        aclrtFreeHost(skDeviceEntryArgsHost);
        skDeviceEntryArgsHost = nullptr;
    }

    // skHeaderInfoHost points inside skDeviceEntryArgsHost (normal path, no separate free needed),
    // or has already been freed in error paths of CopySkDeviceEntryArgsToHost().
    skHeaderInfoHost = nullptr;
}

bool SuperKernelExceptionHandler::StartsWith(const char* source, const char* prefix) {
    if (!source || !prefix) {
        return false;
    }

    size_t srcLen = strlen(source);
    size_t prefixLen = strlen(prefix);
    if (prefixLen > srcLen) {
        return false;
    }
    return strncmp(source, prefix, prefixLen) == 0;
}

aclError SuperKernelExceptionHandler::CheckError(aclError ret, const char *errorMsg) {
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Operation failed: %s returned error code %d", errorMsg, static_cast<int32_t>(ret));
    }
    return ret;
}

bool SuperKernelExceptionHandler::IsSuperKernelException(aclrtExceptionInfo *exceptionInfo) {
    constexpr uint32_t MAX_FUNC_NAME_LEN = 256;
    char funcName[MAX_FUNC_NAME_LEN] = {0};

    // Get exception function handle
    aclrtFuncHandle funcHandle = nullptr;
    auto ret = aclrtGetFuncHandleFromExceptionInfo(exceptionInfo, &funcHandle);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get func handle from exception info, ret=%d", ret);
        return false;
    }

    // Get function name
    ret = aclrtGetFunctionName(funcHandle, MAX_FUNC_NAME_LEN, funcName);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get function name, ret=%d", ret);
        return false;
    }

    // Check if function name starts with sk_entry (e.g., sk_entry, sk_entry_aiv, sk_entry_mix11, etc.)
    if (!StartsWith(funcName, "sk_entry")) {
        SK_LOGD("fault kernel_name '%s' does not start with 'sk_entry', skipping", funcName);
        return false;
    }

    // Check if function name contains "op_trace" for additional symbol printing
    hasOpTrace_ = (strstr(funcName, "op_trace") != nullptr);

    SK_LOGE("Exception is from superkernel function '%s', op_trace=%s, proceeding with handling",
            funcName, hasOpTrace_ ? "true" : "false");
    return true;
}

void SuperKernelExceptionCallBackFunc(aclrtExceptionInfo *exceptionInfo) {
    SuperKernelExceptionHandler handler;
    handler.HandleException(exceptionInfo);
}
