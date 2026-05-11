/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for the details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNEsS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_dump_json.cpp
 * \brief Implementation of SuperKernel graph JSON dump utilities
 */

#include "sk_dump_json.h"
#include "sk_common.h"
#include "sk_graph.h"
#include "sk_lock_detector.h"
#include "sk_log.h"
#include "sk_node.h"
#include "sk_options_manager.h"
#include "sk_scope_info.h"
#include "sk_scope_split.h"
#include "sk_types.h"

#include <acl/acl_rt.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// For resolved function info
#include "runtime/kernel.h"

SkBindMap InitSuperKernelBindMap(aclrtBinHandle binHdl);

namespace {

/**
 * @brief Fill symbol info for a single core type (AIC/AIV) in resolved function info
 */
static void FillSymbolInfoForCore(aclrtBinHandle binHdl, uint64_t skFuncOffset,
                                   uint32_t coreIdx, ResolvedFunctionInfo& info) 
{
    void* binHostAddr = nullptr;
    uint32_t binHostSize = 0;
    if (rtGetBinBuffer(binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize) == 0) {
        std::string symbolName;
        uint64_t funcSize = 0;
        std::string symbolBind;
        if (GetFuncSymbolInfo(static_cast<const char*>(binHostAddr), binHostSize,
                              skFuncOffset, symbolName, funcSize, symbolBind)) {
            info.prefetchCnt[coreIdx] = AlignUpAndClamp(funcSize, coreIdx);
            info.symbolBind[coreIdx] = symbolBind;
        }
    }
}

/**
 * @brief Get resolved function info from kernel function handle
 */
void GetResolvedFuncsForDump(aclrtFuncHandle funcHandle, aclrtBinHandle binHdl,
                             ResolvedFunctionInfo resolvedFuncs[],
                             uint32_t& resolvedNum) 
{
    resolvedNum = 0;
    if (binHdl == nullptr || funcHandle == nullptr) {
        return;
    }

    // Get bind map and function address
    SkBindMap bindMap = InitSuperKernelBindMap(binHdl);
    if (bindMap.empty()) {
        return;
    }

    void* addr[2] = {nullptr, nullptr};
    if (aclrtGetFunctionAddr(funcHandle, addr, addr + 1) != ACL_SUCCESS) {
        return;
    }

    // Get binary device address and calculate offsets
    void* binDevAddr = nullptr;
    size_t binDevSize = 0;
    if (aclrtBinaryGetDevAddress(binHdl, &binDevAddr, &binDevSize) != ACL_SUCCESS) {
        return;
    }

    auto aicItor = bindMap.find((uint64_t)addr[0] - (uint64_t)binDevAddr);
    auto aivItor = bindMap.find((uint64_t)addr[1] - (uint64_t)binDevAddr);

    // Process each split bin
    for (size_t i = 0; i < K_MAX_SPLIT_BIN_COUNT; ++i) {
        ResolvedFunctionInfo& info = resolvedFuncs[i];
        uint32_t validFuncNum = 0;

        if (aicItor != bindMap.end()) {
            uint64_t skFuncOffset = aicItor->second.sknlFuncs[i];
            info.funcAddr[0] = skFuncOffset + (uint64_t)binDevAddr;
            info.funcOffset[0] = skFuncOffset;
            FillSymbolInfoForCore(binHdl, skFuncOffset, 0, info);
            validFuncNum += (i == 0 || skFuncOffset != aicItor->second.sknlFuncs[0]) ? 1 : 0;
        }

        if (aivItor != bindMap.end()) {
            uint64_t skFuncOffset = aivItor->second.sknlFuncs[i];
            info.funcAddr[1] = skFuncOffset + (uint64_t)binDevAddr;
            info.funcOffset[1] = skFuncOffset;
            FillSymbolInfoForCore(binHdl, skFuncOffset, 1, info);
            validFuncNum += (i == 0 || skFuncOffset != aivItor->second.sknlFuncs[0]) ? 1 : 0;
        }

        if (validFuncNum > 0) {
            resolvedFuncs[resolvedNum++] = info;
        }
    }
}

/**
 * @brief Add basic node identification info to JSON
 */
void AddBasicNodeInfo(Json& nodeJson, const SuperKernelBaseNode* node) 
{
    nodeJson["taskId"] = node->GetNodeId();
    nodeJson["streamId"] = node->GetStreamId();
    nodeJson["streamIdxInGraph"] = node->GetStreamIdxInGraph();
    nodeJson["nodeIdxInStream"] = node->GetNodeIdxInStream();
    nodeJson["taskType"] = to_string(node->GetNodeType());
}

/**
 * @brief Add scope information to JSON
 */
void AddScopeInfo(Json& nodeJson, const SuperKernelBaseNode* node, const SuperKernelGraph& graph) 
{
    const std::string& scopeName = node->GetScopeName();
    if (scopeName.empty() || scopeName == "(none)") {
        nodeJson["scopeName"] = Json();
    } else {
        nodeJson["scopeName"] = scopeName;
    }
    nodeJson["isScopeNode"] = node->IsScopeNode();
    nodeJson["isScopeBegin"] = node->IsScopeBegin();
    nodeJson["isScopeEnd"] = node->IsScopeEnd();
    nodeJson["isScopePlaceholder"] = node->IsScopePlaceholder();
}

/**
 * @brief Add fusibility information to JSON
 */
void AddFusibilityInfo(Json& nodeJson, const SuperKernelBaseNode* node) 
{
    nodeJson["isFusible"] = node->IsFusible();
    nodeJson["fusionFailReason"] = FusionFailReasonToStr(node->GetFusionFailReasonInfo());
}

/**
 * @brief Add update status and node relationships to JSON
 */
void AddUpdateStatus(Json& nodeJson, const SuperKernelBaseNode* node) 
{
    nodeJson["isUpdated"] = node->IsUpdated();
    nodeJson["isInvalidated"] = node->IsInvalidated();
    
    uint64_t preNodeId = node->GetPreNodeId();
    uint64_t nextNodeId = node->GetNextNodeId();
    if (preNodeId == INVALID_TASK_ID) {
        nodeJson["preNodeId"] = "none";
    } else {
        nodeJson["preNodeId"] = preNodeId;
    }
    if (nextNodeId == INVALID_TASK_ID) {
        nodeJson["nextNodeId"] = "none";
    } else {
        nodeJson["nextNodeId"] = nextNodeId;
    }
    
    if (!node->GetScopeStreamIds().empty()) {
        std::vector<uint32_t> streamIds(node->GetScopeStreamIds().begin(), 
                                        node->GetScopeStreamIds().end());
        nodeJson["scopeStreamIds"] = streamIds;
    }
}

/**
 * @brief Convert pointer to hex string
 */
std::string PtrToHexString(const void* ptr) 
{
    std::stringstream hexStream;
    hexStream << "0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);
    return hexStream.str();
}

/**
 * @brief Convert uint64_t to hex string
 */
std::string Uint64ToHexString(uint64_t value) 
{
    std::stringstream hexStream;
    hexStream << "0x" << std::hex << value;
    return hexStream.str();
}

/**
 * @brief Add basic kernel information to JSON
 */
void AddBasicKernelInfo(Json& kernelInfos, const KernelInfos& kernelInfo) 
{
    kernelInfos["kernelType"] = to_string(kernelInfo.kernelType);
    kernelInfos["taskRatio"][0] = kernelInfo.taskRatio[0];
    kernelInfos["taskRatio"][1] = kernelInfo.taskRatio[1];
    kernelInfos["cap"] = kernelInfo.cap;
    kernelInfos["numBlocks"] = kernelInfo.numBlocks;
    kernelInfos["vecNum"] = kernelInfo.vecNum;
    kernelInfos["cubeNum"] = kernelInfo.cubeNum;
    kernelInfos["devArgs"] = PtrToHexString(kernelInfo.devArgs);
    kernelInfos["opInfoPtr"] = PtrToHexString(kernelInfo.opInfoPtr);
    kernelInfos["opInfoSize"] = kernelInfo.opInfoSize;
    kernelInfos["funcName"] = kernelInfo.funcName;
    kernelInfos["binHandle"] = PtrToHexString(kernelInfo.binHdl);
    kernelInfos["funcHandle"] = PtrToHexString(kernelInfo.funcHdl);
    kernelInfos["launchKernelCfg"] = PtrToHexString(kernelInfo.launchKernelCfg);
}

/**
 * @brief Convert single kernel attribute to JSON
 */
Json KernelAttrToJson(const aclrtLaunchKernelAttr& launchKernelAttr) 
{
    Json KernelAttrJson;
    KernelAttrJson["id"] = static_cast<int>(launchKernelAttr.id);
    switch (launchKernelAttr.id) {
        case ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE:
            KernelAttrJson["schemMode"] = launchKernelAttr.value.schemMode;
            break;
        case ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE:
            KernelAttrJson["dynUBufSize"] = launchKernelAttr.value.dynUBufSize;
            break;
        case ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE:
            KernelAttrJson["engineType"] = launchKernelAttr.value.engineType;
            break;
        case ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH:
            KernelAttrJson["blockTaskPrefetch"] = launchKernelAttr.value.isBlockTaskPrefetch;
            break;
        case ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP:
            KernelAttrJson["dataDump"] = launchKernelAttr.value.isDataDump;
            break;
        case ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT:
            KernelAttrJson["timeout"] = launchKernelAttr.value.timeout;
            break;
        case ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US:
            KernelAttrJson["timeoutUs"] = launchKernelAttr.value.timeout;
            break;
        default:
            KernelAttrJson["rawValue"] = launchKernelAttr.value.rsv[0];
            break;
    }
    return KernelAttrJson;
}

/**
 * @brief Add launch kernel configuration attributes to JSON
 */
void AddLaunchKernelCfgAttrs(Json& kernelInfos, const KernelInfos& kernelInfo) 
{
    if (kernelInfo.launchKernelCfg == nullptr || kernelInfo.launchKernelCfg->attrs == nullptr) {
        return;
    }
    
    Json cfgAttrs = Json::array();
    for (size_t attrIdx = 0; attrIdx < kernelInfo.launchKernelCfg->numAttrs; ++attrIdx) {
        cfgAttrs.push_back(KernelAttrToJson(kernelInfo.launchKernelCfg->attrs[attrIdx]));
    }
    kernelInfos["launchKernelCfgAttrs"] = cfgAttrs;
}

/**
 * @brief Convert single resolved function info to JSON
 */
Json ResolvedFuncToJson(const ResolvedFunctionInfo& resolvedFunctionInfo) 
{
    Json jsonFuncInfo;
    jsonFuncInfo["funcAddr"][0] = Uint64ToHexString(resolvedFunctionInfo.funcAddr[0]);
    jsonFuncInfo["funcAddr"][1] = Uint64ToHexString(resolvedFunctionInfo.funcAddr[1]);
    jsonFuncInfo["prefetchCnt"][0] = resolvedFunctionInfo.prefetchCnt[0];
    jsonFuncInfo["prefetchCnt"][1] = resolvedFunctionInfo.prefetchCnt[1];
    jsonFuncInfo["funcOffset"][0] = Uint64ToHexString(resolvedFunctionInfo.funcOffset[0]);
    jsonFuncInfo["funcOffset"][1] = Uint64ToHexString(resolvedFunctionInfo.funcOffset[1]);
    jsonFuncInfo["symbolBind"][0] = resolvedFunctionInfo.symbolBind[0];
    jsonFuncInfo["symbolBind"][1] = resolvedFunctionInfo.symbolBind[1];
    return jsonFuncInfo;
}

/**
 * @brief Add resolved functions information to JSON
 */
void AddResolvedFuncs(Json& kernelInfos, const KernelInfos& kernelInfo) 
{
    Json resolvedFuncs = Json::array();
    for (size_t i = 0; i < kernelInfo.resolvedNum && i < K_MAX_SPLIT_BIN_COUNT; ++i) {
        resolvedFuncs.push_back(ResolvedFuncToJson(kernelInfo.resolvedFuncs[i]));
    }
    kernelInfos["resolvedFuncs"] = resolvedFuncs;
}

/**
 * @brief Add kernel type-specific information to JSON
 */
void AddKernelInfos(Json& nodeJson, const SuperKernelBaseNode* node) 
{
    const auto& kernelInfo = node->GetNodeInfos().kernelInfos;
    Json kernelInfos;
    
    AddBasicKernelInfo(kernelInfos, kernelInfo);
    AddLaunchKernelCfgAttrs(kernelInfos, kernelInfo);
    
    kernelInfos["isScheModeOn"] = kernelInfo.isScheModeOn;
    kernelInfos["resolvedNum"] = kernelInfo.resolvedNum;
    AddResolvedFuncs(kernelInfos, kernelInfo);
    
    nodeJson["kernelInfos"] = kernelInfos;
}

/**
 * @brief Add common sync information fields to JSON
 */
void AddCommonSyncInfos(Json& syncInfos, const SuperKernelBaseNode* node) 
{
    static_assert(sizeof(void*) == sizeof(uint64_t), "Pointer size must match uint64_t");
    const auto& syncInfo = node->GetNodeInfos().syncInfos;
    syncInfos["addrValue"] = reinterpret_cast<uint64_t>(syncInfo.addrValue);
    
    if (!syncInfo.correspondingWaitNodeIds.empty()) {
        syncInfos["correspondingWaitNodeIds"] = syncInfo.correspondingWaitNodeIds;
    }
    if (!syncInfo.correspondingResetNodeIds.empty()) {
        syncInfos["correspondingResetNodeIds"] = syncInfo.correspondingResetNodeIds;
    }
    if (!syncInfo.correspondingMemoryWriteNodeIds.empty()) {
        syncInfos["correspondingMemoryWriteNodeIds"] = syncInfo.correspondingMemoryWriteNodeIds;
    }
    if (syncInfo.memoryValue != std::numeric_limits<uint64_t>::max()) {
        syncInfos["memoryValue"] = syncInfo.memoryValue;
    }
    if (syncInfo.memoryWaitFlag != std::numeric_limits<uint32_t>::max()) {
        syncInfos["memoryWaitFlag"] = syncInfo.memoryWaitFlag;
    }
    if (syncInfo.eventFlag != std::numeric_limits<uint64_t>::max()) {
        syncInfos["eventFlag"] = "0x" + std::to_string(syncInfo.eventFlag);
    }
}

/**
 * @brief Add wait node type-specific information to JSON
 */
void AddWaitNodeInfos(Json& nodeJson, const SuperKernelBaseNode* node) 
{
    const auto& syncInfo = node->GetNodeInfos().syncInfos;
    Json syncInfos;
    syncInfos["eventId"] = PtrToHexString(reinterpret_cast<const void*>(syncInfo.eventId));
    syncInfos["correspondingNotifyNodeId"] = syncInfo.correspondingNotifyNodeId;
    AddCommonSyncInfos(syncInfos, node);
    nodeJson["syncInfos"] = syncInfos;
}

/**
 * @brief Add notify/memory-write node type-specific information to JSON
 */
void AddNotifyNodeInfos(Json& nodeJson, const SuperKernelBaseNode* node) 
{
    const auto& syncInfo = node->GetNodeInfos().syncInfos;
    Json syncInfos;
    syncInfos["eventId"] = PtrToHexString(reinterpret_cast<const void*>(syncInfo.eventId));
    AddCommonSyncInfos(syncInfos, node);
    
    if (node->GetNodeType() == SkNodeType::NODE_MEMORY_WRITE ||
        node->GetNodeType() == SkNodeType::NODE_MEMORY_WAIT) {
        syncInfos["vecNum"] = node->GetVecNum();
        syncInfos["cubeNum"] = node->GetCubeNum();
    }
    nodeJson["syncInfos"] = syncInfos;
}

/**
 * @brief Override kernelInfos fields from updated task params
 */
void OverrideKernelInfos(Json& nodeJson, const aclmdlRITaskParams& taskParams) 
{
    nodeJson["kernelInfos"]["numBlocks"] = taskParams.kernelTaskParams.numBlocks;
    nodeJson["kernelInfos"]["funcHandle"] = PtrToHexString(taskParams.kernelTaskParams.funcHandle);
    nodeJson["kernelInfos"]["devArgs"] = PtrToHexString(taskParams.kernelTaskParams.args);
    nodeJson["kernelInfos"]["argsSize"] = taskParams.kernelTaskParams.argsSize;
    nodeJson["kernelInfos"]["isHostArgs"] = taskParams.kernelTaskParams.isHostArgs;
    nodeJson["kernelInfos"]["opInfoPtr"] = PtrToHexString(taskParams.opInfoPtr);
    nodeJson["kernelInfos"]["opInfoSize"] = taskParams.opInfoSize;
}

/**
 * @brief Override syncInfos fields from updated task params
 */
void OverrideSyncInfos(Json& nodeJson, const aclmdlRITaskParams& taskParams) 
{
    nodeJson["syncInfos"]["addrValue"] = PtrToHexString(taskParams.valueWriteTaskParams.devAddr);
    nodeJson["syncInfos"]["memoryValue"] = taskParams.valueWriteTaskParams.value;
}

/**
 * @brief Add type-specific information based on node type
 */
void AddTypeSpecificInfo(Json& nodeJson, const SuperKernelBaseNode* node) 
{
    switch (node->GetNodeType()) {
        case SkNodeType::NODE_KERNEL:
            AddKernelInfos(nodeJson, node);
            break;
        case SkNodeType::NODE_WAIT:
        case SkNodeType::NODE_MEMORY_WAIT:
            AddWaitNodeInfos(nodeJson, node);
            break;
        case SkNodeType::NODE_NOTIFY:
        case SkNodeType::NODE_MEMORY_WRITE:
            AddNotifyNodeInfos(nodeJson, node);
            break;
        case SkNodeType::NODE_DEFAULT:
        default:
            break;
    }
}

/**
 * @brief Override fields from updated task params if node has been updated
 */
void AddUpdatedTaskParams(Json& nodeJson, const SuperKernelBaseNode* node) 
{
    if (!node->IsUpdated()) {
        return;
    }
    
    const auto& taskParams = node->GetTaskParams();
    nodeJson["nodeType"] = TaskTypeToString(taskParams.type);
    
    switch (taskParams.type) {
        case ACL_MODEL_RI_TASK_KERNEL:
            OverrideKernelInfos(nodeJson, taskParams);
            break;
        case ACL_MODEL_RI_TASK_VALUE_WRITE:
            OverrideSyncInfos(nodeJson, taskParams);
            break;
        case ACL_MODEL_RI_TASK_VALUE_WAIT:
            nodeJson["syncInfos"]["addrValue"] = PtrToHexString(taskParams.valueWaitTaskParams.devAddr);
            nodeJson["syncInfos"]["memoryValue"] = taskParams.valueWaitTaskParams.value;
            nodeJson["syncInfos"]["memoryWaitFlag"] = taskParams.valueWaitTaskParams.flag;
            break;
        default:
            break;
    }
}

/**
 * @brief Convert node to JSON object with detailed information
 */
Json NodeToJson(const SuperKernelBaseNode* node, const SuperKernelGraph& graph) 
{
    Json nodeJson;
    if (node == nullptr) {
        SK_LOGI("node is null, skipping it");
        return nodeJson;
    }
    
    AddBasicNodeInfo(nodeJson, node);
    AddScopeInfo(nodeJson, node, graph);
    AddFusibilityInfo(nodeJson, node);
    AddUpdateStatus(nodeJson, node);
    AddTypeSpecificInfo(nodeJson, node);
    AddUpdatedTaskParams(nodeJson, node);
    
    return nodeJson;
}

/**
 * @brief Get the JSON output path
 * 
 * Uses the same path logic as super_kernel.cpp DumpGraphJson function
 * Format: {metaDir}/sk_graph_after.json
 * 
 * @param graph Reference to the SuperKernelGraph to get modelRI
 * @return Full path for JSON output
 */
std::string GetJsonOutputPath(const SuperKernelGraph& graph) 
{
    std::string metaDir = CreateSkMetaDirectory(graph.GetModelRI());
    if (metaDir.empty()) {
        SK_LOGE("Failed to create sk_meta directory for JSON dump");
        return "";
    }
    return metaDir + "/sk_graph_before.json";
}

/**
 * @brief Add graph summary information to JSON
 */
void AddGraphSummaryToJson(Json& graphJson, const SuperKernelGraph& graph) 
{
    graphJson["graph"]["totalNodes"] = graph.GetSortedNodeIds().size();
    graphJson["graph"]["totalStreams"] = graph.GetStreams().size();
}

/**
 * @brief Add stream information to JSON
 */
void AddStreamsInfoToJson(Json& graphJson, const SuperKernelGraph& graph) 
{
    const auto& streams = graph.GetStreams();
    const auto& headNodes = graph.GetHeadNodes();
    const auto& nodeSizeInStream = graph.GetNodeSizeInStream();
    
    Json streamsJson = Json::array();
    for (size_t i = 0; i < streams.size(); ++i) {
        Json streamInfo;
        streamInfo["streamIdx"] = i;
        streamInfo["headNodeId"] = (i < headNodes.size()) ? headNodes[i] : 0;
        streamInfo["nodeSize"] = (i < nodeSizeInStream.size()) ? nodeSizeInStream[i] : 0;
        streamsJson.push_back(streamInfo);
    }
    graphJson["graph"]["streams"] = streamsJson;
}

/**
 * @brief Add scope name mappings to JSON
 */
void AddScopeNamesToJson(Json& graphJson, const SuperKernelGraph& graph) 
{
    Json scopeNamesJson = Json::array();
    std::string scopeName;
    uint32_t idx = 0;
    while (graph.GetScopeNameByIdx(idx, scopeName)) {
        Json scopeEntry;
        scopeEntry["scopeIdx"] = idx;
        if (scopeName.empty() || scopeName == "(none)") {
            scopeEntry["scopeName"] = Json();
        } else {
            scopeEntry["scopeName"] = scopeName;
        }
        scopeNamesJson.push_back(scopeEntry);
        idx++;
    }
    graphJson["graph"]["scopeNames"] = scopeNamesJson;
}

/**
 * @brief Add scope name mappings to fused graph JSON (reuses AddScopeNamesToJson)
 */
void AddFusedGraphScopeNames(Json& fusedGraphJson, const SuperKernelGraph& graph) 
{
    AddScopeNamesToJson(fusedGraphJson, graph);
}

/**
 * @brief Collect all nodes to JSON array
 */
void CollectAllNodesToJson(Json& graphJson, const SuperKernelGraph& graph) 
{
    Json nodesArray = Json::array();
    std::vector<uint64_t> sortedNodeIds = graph.GetSortedNodeIds();
    
    for (size_t i = 0; i < sortedNodeIds.size(); ++i) {
        uint64_t nodeId = sortedNodeIds[i];
        SuperKernelBaseNode* node = graph.GetNodeById(nodeId);
        if (node != nullptr) {
            Json nodeJson = NodeToJson(node, graph);
            nodesArray.push_back(nodeJson);
        }
    }
    graphJson["nodes"] = nodesArray;
}

/**
 * @brief Build complete graph JSON object
 */
Json BuildGraphNodesJson(const SuperKernelGraph& graph) 
{
    Json graphNodesJson;
    graphNodesJson["version"] = "1.0";
    graphNodesJson["description"] = "SuperKernel Graph Node Information After Initialization";
    graphNodesJson["modelRI"] = std::to_string(reinterpret_cast<uintptr_t>(graph.GetModelRI()));
    
    AddGraphSummaryToJson(graphNodesJson, graph);
    AddStreamsInfoToJson(graphNodesJson, graph);
    AddScopeNamesToJson(graphNodesJson, graph);
    CollectAllNodesToJson(graphNodesJson, graph);
    
    return graphNodesJson;
}

/**
 * @brief Write JSON to file
 * @param jsonObj JSON object to write
 * @param jsonPath Output file path
 * @return true if successful, false otherwise
 */
bool WriteJsonToFile(const Json& jsonObj, const std::string& jsonPath) 
{
    std::ofstream outFile(jsonPath);
    if (!outFile.is_open()) {
        SK_LOGE("Failed to open JSON file for writing: %s", jsonPath.c_str());
        return false;
    }
    
    outFile << jsonObj.dump(2);  // Pretty print with 2 spaces indent
    outFile.close();
    return true;
}

/**
 * @brief Get the JSON output path for task queue
 * @param graph Reference to the SuperKernelGraph to get modelRI
 * @return Full path for JSON output
 */
bool GetTaskQueueJsonOutputPath(const SuperKernelGraph& graph, std::string& outputPath) 
{
    std::string metaDir = CreateSkMetaDirectory(graph.GetModelRI());
    if (metaDir.empty()) {
        SK_LOGE("Failed to create sk_meta directory for task queue JSON dump");
        outputPath.clear(); 
        return false;
    }
    
    outputPath = metaDir + "/sk_task_queue.json";
    return true;
}

/**
 * @brief Dump SkTask (AIC/AIV task queue) to JSON
 * @param task SkTask reference
 * @param queueName Queue name ("AIC" or "AIV")
 * @return JSON object representing the task queue
 */
Json SkTaskToJson(const SkTask& task) 
{
    Json taskJson;
    taskJson["numBlocks"] = task.numBlocks;
    taskJson["funcCnt"] = task.funcCnt;
    
    const TaskQue* taskQue = task.GetTaskQue();
    if (taskQue != nullptr) {
        taskJson["taskQue"]["taskCnt"] = taskQue->taskCnt;
        
        Json taskInfos = Json::array();
        for (uint32_t i = 0; i < taskQue->taskCnt; ++i) {
            const TaskInfo& singleTaskInfo = taskQue->taskInfos[i];
            Json taskInfo;
            taskInfo["nodeIndex"] = singleTaskInfo.index;  // 原始任务索引
            taskInfo["type"] = to_string(singleTaskInfo.type);
            taskInfo["numBlocks"] = singleTaskInfo.numBlocks;
            taskInfo["entryCnt"] = singleTaskInfo.entryCnt;
            std::stringstream hexStream;
            hexStream << "0x" << std::hex << singleTaskInfo.args;
            taskInfo["args"] = hexStream.str();
            taskInfo["debugOptions"] = singleTaskInfo.debugOptions;
            
            Json entries = Json::array();
            for (uint32_t j = 0; j < singleTaskInfo.entryCnt && j < 4; ++j) {
                std::stringstream hexStream;
                hexStream << "0x" << std::hex << singleTaskInfo.entry[j];
                entries.push_back(hexStream.str());
            }
            taskInfo["entries"] = entries;
            
            taskInfos.push_back(taskInfo);
        }
        taskJson["taskQue"]["taskInfos"] = taskInfos;
    }
    
    return taskJson;
}

/**
 * @brief Get the JSON output path for fused graph
 * @param graph Reference to the SuperKernelGraph to get modelRI
 * @return Full path for JSON output
 */
bool GetFusedGraphJsonOutputPath(const SuperKernelGraph& graph, std::string& outputPath) 
{
    std::string metaDir = CreateSkMetaDirectory(graph.GetModelRI());
    if (metaDir.empty()) {
        SK_LOGE("Failed to create sk_meta directory for fused graph JSON dump");
        outputPath.clear();
        return false;
    }
    
    outputPath = metaDir + "/sk_graph_after.json";
    return true;
}

/**
 * @brief Get a human-readable name for a node
 * @param node Pointer to the node
 * @return Node name string
 */
std::string GetNodeName(const SuperKernelBaseNode* node) 
{
    if (node == nullptr) {
        return "";
    }
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
        return node->GetNodeInfos().kernelInfos.funcName;
    } else {
        return to_string(node->GetNodeType());
    }
}

/**
 * @brief Convert a scope to nested JSON object
 * @param scopeInfo Reference to the scope
 * @param scopeIndex The index of this scope in the scopeInfos vector
 * @param graph Reference to the SuperKernelGraph
 * @return JSON object representing the scope with nested nodes
 */
Json ScopeToNestedJson(const SuperKernelScopeInfo& scopeInfo, size_t scopeIndex, const SuperKernelGraph& graph) 
{
    Json scopeJson;

    // Scope metadata - get scopeId from scopeInfo
    scopeJson["scopeId"] = scopeInfo.GetScopeId();
    const std::string& scopeName = scopeInfo.GetExtInfo().scopeName;
    if (scopeName.empty() || scopeName == "(none)") {
        scopeJson["scopeName"] = Json();
    } else {
        scopeJson["scopeName"] = scopeName;
    }

    // Collect all node IDs in this scope
    std::vector<uint64_t> nodeIds;
    const auto& nodes = scopeInfo.GetNodes();
    for (const auto* node : nodes) {
        if (node != nullptr) {
            nodeIds.push_back(node->GetNodeId());
        }
    }
    scopeJson["nodeIds"] = nodeIds;
    scopeJson["nodeCount"] = nodeIds.size();

    // Add startNodeName and endNodeName for fused scopes
    if (!nodes.empty() && nodes.front() != nullptr && nodes.back() != nullptr) {
        scopeJson["startNodeName"] = GetNodeName(nodes.front());
        scopeJson["endNodeName"] = GetNodeName(nodes.back());
    }

    // Stream info
    const auto& streamInfos = scopeInfo.GetScopeStreamInfos();
    Json streamsJson = Json::array();
    for (const auto& streamInfo : streamInfos) {
        Json streamJson;
        streamJson["streamIdx"] = streamInfo.streamIdx;
        streamJson["headNodeIdx"] = streamInfo.headNodeIdx;
        streamJson["tailNodeIdx"] = streamInfo.tailNodeIdx;
        streamJson["nodeSize"] = streamInfo.nodeSize;
        streamsJson.push_back(streamJson);
    }
    scopeJson["streams"] = streamsJson;

    // Nested nodes array - all nodes inside this scope
    Json nestedNodes = Json::array();
    for (const auto* node : nodes) {
        if (node != nullptr) {
            Json nodeJson = NodeToJson(node, graph);
            nestedNodes.push_back(nodeJson);
        }
    }
    scopeJson["nodes"] = nestedNodes;

    // Failure reason if any
    if (scopeInfo.GetExtInfo().failReason != ScopeFailReason::NONE) {
        scopeJson["failReason"] = ScopeFailReasonToStr(scopeInfo.GetExtInfo().failReason);
    }

    return scopeJson;
}

/**
 * @brief Structure to hold node-to-scope mapping data
 */
struct NodeScopeMapping {
    std::unordered_set<uint64_t> scopeHeaderNodeIds;
    std::unordered_map<uint64_t, const SuperKernelScopeInfo*> nodeIdToScope;
    std::unordered_map<const SuperKernelScopeInfo*, size_t> scopeToIndex;
};

/**
 * @brief Build mapping from nodes to scopes
 */
NodeScopeMapping BuildNodeScopeMapping(const std::vector<SuperKernelScopeInfo>& scopeInfos) 
{
    NodeScopeMapping mapping;
    
    for (size_t scopeIdx = 0; scopeIdx < scopeInfos.size(); ++scopeIdx) {
        const auto& scopeInfo = scopeInfos[scopeIdx];
        mapping.scopeToIndex[&scopeInfo] = scopeIdx;
        const auto& nodes = scopeInfo.GetNodes();
        if (!nodes.empty() && nodes[0] != nullptr) {
            uint64_t headerNodeId = nodes[0]->GetNodeId();
            mapping.scopeHeaderNodeIds.insert(headerNodeId);
            
            for (const auto* node : nodes) {
                if (node != nullptr) {
                    mapping.nodeIdToScope[node->GetNodeId()] = &scopeInfo;
                }
            }
        }
    }
    
    return mapping;
}

/**
 * @brief Add fused graph summary to JSON
 */
void AddFusedGraphSummary(Json& fusedGraphJson, size_t totalNodes, size_t totalScopes, const SuperKernelGraph& graph) 
{
    fusedGraphJson["graph"]["totalNodes"] = totalNodes;
    fusedGraphJson["graph"]["totalScopes"] = totalScopes;
    fusedGraphJson["graph"]["totalStreams"] = graph.GetStreams().size();
}

/**
 * @brief Collect nodes from a scope into processed set
 */
void CollectScopeNodes(const SuperKernelScopeInfo* scopeInfo, std::unordered_set<uint64_t>& processedNodes) 
{
    for (const auto* node : scopeInfo->GetNodes()) {
        if (node != nullptr) {
            processedNodes.insert(node->GetNodeId());
        }
    }
}

/**
 * @brief Add a scope as nested JSON to nodes array
 */
void AddScopeToJson(Json& nodesArray, const SuperKernelScopeInfo* scopeInfo,
                    const NodeScopeMapping& mapping, const SuperKernelGraph& graph) 
                    {
    size_t scopeIdx = mapping.scopeToIndex.at(scopeInfo);
    Json scopeJson = ScopeToNestedJson(*scopeInfo, scopeIdx, graph);
    nodesArray.push_back(scopeJson);
}

/**
 * @brief Find scope info by node ID from mapping
 */
const SuperKernelScopeInfo* FindScopeByNodeId(uint64_t nodeId, const NodeScopeMapping& mapping) 
{
    if (mapping.scopeHeaderNodeIds.count(nodeId) == 0) {
        return nullptr;
    }
    auto it = mapping.nodeIdToScope.find(nodeId);
    return (it != mapping.nodeIdToScope.end()) ? it->second : nullptr;
}

/**
 * @brief Add a single node to JSON array
 */
void AddNodeToJson(Json& nodesArray, uint64_t nodeId, const SuperKernelGraph& graph) 
{
    SuperKernelBaseNode* node = graph.GetNodeById(nodeId);
    if (node != nullptr) {
        Json nodeJson = NodeToJson(node, graph);
        nodesArray.push_back(nodeJson);
    }
}

/**
 * @brief Process a single node/scope entry and add to JSON array
 */
void ProcessNodeEntry(Json& nodesArray, uint64_t nodeId, 
                      const NodeScopeMapping& mapping, 
                      const SuperKernelGraph& graph,
                      std::unordered_set<uint64_t>& processedNodes) 
{
    if (processedNodes.count(nodeId) > 0) {
        return;
    }

    const SuperKernelScopeInfo* scopeInfo = FindScopeByNodeId(nodeId, mapping);
    if (scopeInfo != nullptr) {
        CollectScopeNodes(scopeInfo, processedNodes);
        AddScopeToJson(nodesArray, scopeInfo, mapping, graph);
    } else {
        AddNodeToJson(nodesArray, nodeId, graph);
    }
}

/**
 * @brief Collect nodes and scopes to JSON array
 */
void CollectNodesAndScopesToJson(Json& nodesArray, 
                                  const std::vector<uint64_t>& sortedNodeIds,
                                  const NodeScopeMapping& mapping,
                                  const SuperKernelGraph& graph) 
{
    std::unordered_set<uint64_t> processedNodes;
    
    for (size_t i = 0; i < sortedNodeIds.size(); ++i) {
        uint64_t nodeId = sortedNodeIds[i];
        ProcessNodeEntry(nodesArray, nodeId, mapping, graph, processedNodes);
    }
}

/**
 * @brief Build complete fused graph JSON object
 */
Json BuildFusedGraphJson(const SuperKernelGraph& graph, const std::vector<SuperKernelScopeInfo>& scopeInfos) 
{
    Json fusedGraphJson;
    fusedGraphJson["version"] = "1.0";
    fusedGraphJson["description"] = "SuperKernel Graph After Fusion - Nested Structure";
    fusedGraphJson["modelRI"] = std::to_string(reinterpret_cast<uintptr_t>(graph.GetModelRI()));
    
    std::vector<uint64_t> allSortedNodeIds = graph.GetSortedNodeIds();
    
    AddFusedGraphSummary(fusedGraphJson, allSortedNodeIds.size(), scopeInfos.size(), graph);
    
    NodeScopeMapping mapping = BuildNodeScopeMapping(scopeInfos);
    
    AddFusedGraphScopeNames(fusedGraphJson, graph);
    
    Json nodesArray = Json::array();
    CollectNodesAndScopesToJson(nodesArray, allSortedNodeIds, mapping, graph);
    fusedGraphJson["nodes"] = nodesArray;
    
    return fusedGraphJson;
}

/**
 * @brief Add basic kernel parameters to JSON
 */
static void AddBasicKernelParams(Json& kernelParamsJson, const aclmdlRIKernelTaskParams& kernelParams) 
{
    char funcName[256] = {0};
    if (aclrtGetFunctionName(kernelParams.funcHandle, sizeof(funcName), funcName) == ACL_SUCCESS) {
        kernelParamsJson["funcName"] = std::string(funcName);
    }
    kernelParamsJson["funcHandle"] = PtrToHexString(kernelParams.funcHandle);
    kernelParamsJson["numBlocks"] = kernelParams.numBlocks;
    kernelParamsJson["devargs"] = PtrToHexString(kernelParams.args);
    kernelParamsJson["argsSize"] = kernelParams.argsSize;
    kernelParamsJson["isHostArgs"] = kernelParams.isHostArgs;
    kernelParamsJson["launchKernelCfg"] = PtrToHexString(kernelParams.cfg);

    // Get binary handle
    aclrtBinHandle binHdl = nullptr;
    if (aclrtFunctionGetBinary(kernelParams.funcHandle, &binHdl) == ACL_SUCCESS) {
        kernelParamsJson["binHandle"] = PtrToHexString(binHdl);
    }

    int64_t kernelType = 0;
    int64_t taskRatioVal = 0;
    CHECK_ACL(aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_TYPE, &kernelType));
    CHECK_ACL(aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_RATIO, &taskRatioVal));

    const int16_t* taskRatioInt16 = reinterpret_cast<const int16_t*>(&taskRatioVal);
    uint32_t skTaskRatio[2] = {
        static_cast<uint32_t>(taskRatioInt16[1]),
        static_cast<uint32_t>(taskRatioInt16[0])
    };

    kernelParamsJson["kernelTypeInt"] = static_cast<int>(kernelType);
    kernelParamsJson["kernelType"] = GetKernelTypeString(static_cast<uint32_t>(kernelType), skTaskRatio);
    kernelParamsJson["taskRatio"] = Json::array({ skTaskRatio[0], skTaskRatio[1] });
}

/**
 * @brief Add resolved functions info to JSON
 */
static void AddResolvedFuncsToJson(Json& kernelParamsJson, aclrtFuncHandle funcHandle,
                                   aclrtBinHandle binHdl) 
{
    ResolvedFunctionInfo dumpResolvedFuncs[K_MAX_SPLIT_BIN_COUNT];
    uint32_t resolvedNum = 0;
    GetResolvedFuncsForDump(funcHandle, binHdl, dumpResolvedFuncs, resolvedNum);
    kernelParamsJson["resolvedNum"] = resolvedNum;

    if (resolvedNum == 0) {
        return;
    }

    Json resolvedFuncsJson = Json::array();
    for (uint32_t i = 0; i < resolvedNum && i < K_MAX_SPLIT_BIN_COUNT; ++i) {
        Json rfJson;
        rfJson["funcAddr"][0] = Uint64ToHexString(dumpResolvedFuncs[i].funcAddr[0]);
        rfJson["funcAddr"][1] = Uint64ToHexString(dumpResolvedFuncs[i].funcAddr[1]);
        rfJson["funcOffset"][0] = Uint64ToHexString(dumpResolvedFuncs[i].funcOffset[0]);
        rfJson["funcOffset"][1] = Uint64ToHexString(dumpResolvedFuncs[i].funcOffset[1]);
        rfJson["prefetchCnt"][0] = dumpResolvedFuncs[i].prefetchCnt[0];
        rfJson["prefetchCnt"][1] = dumpResolvedFuncs[i].prefetchCnt[1];
        rfJson["symbolBind"][0] = dumpResolvedFuncs[i].symbolBind[0];
        rfJson["symbolBind"][1] = dumpResolvedFuncs[i].symbolBind[1];
        resolvedFuncsJson.push_back(rfJson);
    }
    kernelParamsJson["resolvedFuncs"] = resolvedFuncsJson;
}

/**
 * @brief Add kernel task parameters to JSON (nested format)
 */
void AddKernelTaskParams(Json& taskJson, const aclmdlRIKernelTaskParams& kernelParams,
                         const aclmdlRITaskParams& taskParams, const aclmdlRITask& task) 
{
    Json kernelParamsJson;
    AddBasicKernelParams(kernelParamsJson, kernelParams);

    // Common task params
    kernelParamsJson["opInfoPtr"] = PtrToHexString(taskParams.opInfoPtr);
    kernelParamsJson["opInfoSize"] = taskParams.opInfoSize;
    kernelParamsJson["taskGrp"] = PtrToHexString(taskParams.taskGrp);

    // Get resolved functions info
    aclrtBinHandle binHdl = nullptr;
    if (aclrtFunctionGetBinary(kernelParams.funcHandle, &binHdl) == ACL_SUCCESS) {
        AddResolvedFuncsToJson(kernelParamsJson, kernelParams.funcHandle, binHdl);
    }

    taskJson["kernelParams"] = kernelParamsJson;
}

/**
 * @brief Add event record task parameters to JSON (nested format)
 */
void AddEventRecordParams(Json& taskJson, const aclmdlRIEventRecordTaskParams& eventParams) 
{
    Json eventParamsJson;
    eventParamsJson["eventId"] = PtrToHexString(eventParams.event);
    eventParamsJson["eventFlag"] = eventParams.eventFlag;
    taskJson["eventRecordParams"] = eventParamsJson;
}

/**
 * @brief Add event wait task parameters to JSON (nested format)
 */
void AddEventWaitParams(Json& taskJson, const aclmdlRIEventWaitTaskParams& eventParams)
{
    Json eventParamsJson;
    eventParamsJson["event"] = PtrToHexString(eventParams.event);
    eventParamsJson["eventFlag"] = eventParams.eventFlag;
    taskJson["eventWaitParams"] = eventParamsJson;
}

/**
 * @brief Add event reset task parameters to JSON (nested format)
 */
void AddEventResetParams(Json& taskJson, const aclmdlRIEventResetTaskParams& eventParams) 
{
    Json eventParamsJson;
    eventParamsJson["event"] = PtrToHexString(eventParams.event);
    eventParamsJson["eventFlag"] = eventParams.eventFlag;
    taskJson["eventResetParams"] = eventParamsJson;
}

/**
 * @brief Add value write task parameters to JSON (nested format)
 */
void AddValueWriteParams(Json& taskJson, const aclmdlRIValueWriteTaskParams& valueParams) 
{
    Json valueParamsJson;
    valueParamsJson["devAddr"] = PtrToHexString(valueParams.devAddr);
    valueParamsJson["value"] = Uint64ToHexString(valueParams.value);
    taskJson["valueWriteParams"] = valueParamsJson;
}

/**
 * @brief Add value wait task parameters to JSON (nested format)
 */
void AddValueWaitParams(Json& taskJson, const aclmdlRIValueWaitTaskParams& valueParams) 
{
    Json valueParamsJson;
    valueParamsJson["devAddr"] = PtrToHexString(valueParams.devAddr);
    valueParamsJson["value"] = Uint64ToHexString(valueParams.value);
    valueParamsJson["flag"] = valueParams.flag;
    taskJson["valueWaitParams"] = valueParamsJson;
}

/**
 * @brief Add task parameters to JSON based on task type
 */
void AddTaskParamsToJson(Json& taskJson, const aclmdlRITaskParams& params, const aclmdlRITask& task) 
{
    switch (params.type) {
        case ACL_MODEL_RI_TASK_KERNEL:
            AddKernelTaskParams(taskJson, params.kernelTaskParams, params, task);
            break;
        case ACL_MODEL_RI_TASK_EVENT_RECORD:
            AddEventRecordParams(taskJson, params.eventRecordTaskParams);
            break;
        case ACL_MODEL_RI_TASK_EVENT_WAIT:
            AddEventWaitParams(taskJson, params.eventWaitTaskParams);
            break;
        case ACL_MODEL_RI_TASK_EVENT_RESET:
            AddEventResetParams(taskJson, params.eventResetTaskParams);
            break;
        case ACL_MODEL_RI_TASK_VALUE_WRITE:
            AddValueWriteParams(taskJson, params.valueWriteTaskParams);
            break;
        case ACL_MODEL_RI_TASK_VALUE_WAIT:
            AddValueWaitParams(taskJson, params.valueWaitTaskParams);
            break;
        default:
            break;
    }
}

/**
 * @brief Convert a single task to JSON object
 */
Json TaskToJson(uint32_t taskIdx, int32_t streamId,
                aclmdlRITaskType taskType, const aclmdlRITaskParams* params,
                const aclmdlRITask& task) 
{
    Json taskJson;
    // Get the real taskId (nodeId) from modelRI using aclmdlRITaskGetSeqId
    uint32_t seqId = 0;
    aclError ret = aclmdlRITaskGetSeqId(task, &seqId);
    if (ret == ACL_SUCCESS) {
        taskJson["taskId"] = static_cast<uint64_t>(seqId);
    }
    // taskIdx is the index within the stream (from enumeration)
    taskJson["streamId"] = streamId;
    taskJson["taskType"] = TaskTypeToString(taskType);
    taskJson["taskTypeInt"] = static_cast<int>(taskType);

    

    // Add type-specific parameters (nested in separate fields)
    if (params != nullptr) {
        AddTaskParamsToJson(taskJson, *params, task);
    }

    return taskJson;
}

/**
 * @brief Print nodeIds in batches to avoid exceeding maxLineLength
 */
void PrintNodeIdsInBatches(const std::vector<uint64_t>& nodeIds, const char* prefix = "    nodeIds: [") 
{
    constexpr size_t BATCH_SIZE = 100;
    constexpr size_t MAX_LINE_LEN = 4096;
    const std::string suffix = "]";

    std::string linePrefix = prefix;
    std::string currentBatch;

    for (size_t i = 0; i < nodeIds.size(); ++i) {
        std::string nodeIdStr = (currentBatch.empty() ? "" : ", ") + std::to_string(nodeIds[i]);
        size_t estimatedLen = linePrefix.length() + currentBatch.length() + nodeIdStr.length() + suffix.length();

        if (estimatedLen > MAX_LINE_LEN && !currentBatch.empty()) {
            SK_LOGI("%s%s%s", linePrefix.c_str(), currentBatch.c_str(), suffix.c_str());
            currentBatch.clear();
            linePrefix = "    continued: ";
        }
        currentBatch += nodeIdStr;
    }

    if (!currentBatch.empty()) {
        SK_LOGI("%s%s%s", linePrefix.c_str(), currentBatch.c_str(), suffix.c_str());
    }
}

/**
 * @brief Collect kernel node IDs from a node list, skipping scope nodes
 */
std::unordered_set<uint64_t> CollectKernelIdsFromNodes(const SuperKernelBaseNode* const* nodes, size_t nodeCount) 
{
    std::unordered_set<uint64_t> kernelIds;
    for (size_t i = 0; i < nodeCount; ++i) {
        const auto* node = nodes[i];
        if (node == nullptr || node->IsScopeBegin() || node->IsScopeEnd() || node->IsScopePlaceholder()) {
            continue;
        }
        if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
            kernelIds.insert(node->GetNodeId());
        }
    }
    return kernelIds;
}

/**
 * @brief Collect kernel node IDs from a scope
 */
std::unordered_set<uint64_t> CollectKernelIdsFromScope(const SuperKernelScopeInfo& scopeInfo) 
{
    const auto& nodes = scopeInfo.GetNodes();
    return CollectKernelIdsFromNodes(nodes.data(), nodes.size());
}

/**
 * @brief Build original kernel sets grouped by scopeBitFlags
 */
std::unordered_map<std::string, std::unordered_set<uint64_t>> BuildOriginalKernelSets(
    const SuperKernelGraph& graph, const std::vector<OriginalScopeInfo>& originalScopes) 
{
    std::unordered_map<std::string, std::unordered_set<uint64_t>> originalKernelSets;

    for (const auto& origScope : originalScopes) {
        std::unordered_set<uint64_t> kernelIds;
        for (uint64_t nodeId : origScope.nodeIds) {
            SuperKernelBaseNode* node = graph.GetNodeById(nodeId);
            if (node == nullptr || node->IsScopeBegin() || node->IsScopeEnd() || node->IsScopePlaceholder()) {
                continue;
            }
            if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
                kernelIds.insert(nodeId);
            }
        }
        originalKernelSets[graph.BitsetToString(origScope.GetScopeBitFlags())] = std::move(kernelIds);
    }
    return originalKernelSets;
}

/**
 * @brief Check if fused scope kernel set matches original scope
 */
bool IsKernelSetMatch(const SuperKernelScopeInfo& scope,
                      const std::unordered_map<std::string, std::unordered_set<uint64_t>>& originalKernelSets,
                      const SuperKernelGraph& graph) 
{
    const auto& scopeBitFlags = scope.GetScopeBitFlags();
    std::string bitFlagStr = graph.BitsetToString(scopeBitFlags);

    auto it = originalKernelSets.find(bitFlagStr);
    if (it == originalKernelSets.end()) {
        return false;
    }

    std::unordered_set<uint64_t> fusedKernelIds = CollectKernelIdsFromScope(scope);
    return it->second == fusedKernelIds;
}

/**
 * @brief Print fused node IDs with line length limit
 */
void PrintFusedNodeIds(const std::vector<uint64_t>& fusedNodeIds) 
{
    constexpr size_t MAX_LINE_LEN = 4096;
    std::string prefix = "    fusedNodeIds=[";
    std::string suffix = "];";
    std::string currentLine;

    for (size_t i = 0; i < fusedNodeIds.size(); ++i) {
        std::string nodeIdStr = (currentLine.empty() ? "" : ", ") + std::to_string(fusedNodeIds[i]);

        if (currentLine.length() + nodeIdStr.length() + suffix.length() > MAX_LINE_LEN) {
            SK_LOGI("%s%s%s", prefix.c_str(), currentLine.c_str(), suffix.c_str());
            currentLine.clear();
            prefix = "    continued: ";
        }
        currentLine += nodeIdStr;
    }

    if (!currentLine.empty()) {
        SK_LOGI("%s%s%s", prefix.c_str(), currentLine.c_str(), suffix.c_str());
    }
}

/**
 * @brief Collect fused node IDs and check if scope has kernel
 */
void CollectFusedNodes(const ScopeExtInfo& extInfo,
                       std::vector<uint64_t>& fusedNodeIds, bool& hasKernel) 
{
    fusedNodeIds.clear();
    hasKernel = false;

    for (const auto* node : extInfo.filteredNodes) {
        if (node == nullptr || node->IsScopeBegin() || node->IsScopeEnd() || node->IsScopePlaceholder()) {
            continue;
        }
        fusedNodeIds.push_back(node->GetNodeId());
        if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
            hasKernel = true;
        }
    }
}

} // anonymous namespace

void PrintFusedScopes(const SuperKernelGraph& graph,
                      const std::vector<SuperKernelScopeInfo>& processedScopeInfos) 
{
    // Build scopeId -> scope index map for root tracing
    std::unordered_map<uint16_t, size_t> scopeIdToIdx;
    for (size_t i = 0; i < processedScopeInfos.size(); ++i) {
        scopeIdToIdx[processedScopeInfos[i].GetScopeId()] = i;
    }

    // Build original scope kernel node sets
    const auto& originalScopes = graph.GetOriginalScopeInfos();
    auto originalKernelSets = BuildOriginalKernelSets(graph, originalScopes);

    // Print scopes after fusion
    SK_LOGI("scopes after fusion:");
    for (const auto& scopeInfo : processedScopeInfos) {
        const auto& extInfo = scopeInfo.GetExtInfo();

        std::vector<uint64_t> fusedNodeIds;
        bool hasKernel = false;
        CollectFusedNodes(extInfo, fusedNodeIds, hasKernel);

        // Skip scope without kernel
        if (!hasKernel) {
            continue;
        }

        ScopeBreakInfo rootScopeBreakInfo = FindRootBreakInfo(scopeInfo, scopeIdToIdx, processedScopeInfos);
        std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(scopeInfo.GetScopeBitFlags(), graph);

        // Line 0: scopeId and scopeBitFlag
        SK_LOGI("  scopeId=%u, scopeBitFlag=%s", scopeInfo.GetScopeId(),
                graph.BitsetToString(scopeInfo.GetScopeBitFlags()).c_str());

        // Line 1: fusedNodeIds
        PrintFusedNodeIds(fusedNodeIds);

        // Line 2: fusionStatus (failReason if not success)
        if (extInfo.fusionStatus == ScopeFusionStatus::SUCCESS) {
            SK_LOGI("    fusionStatus=SUCCESS");
        } else {
            SK_LOGI("    fusionStatus=%s, failReason=%s",
                    ScopeFusionStatusToStr(extInfo.fusionStatus), ScopeFailReasonToStr(extInfo.failReason));
        }

        // Line 3: breakReason (if kernel set differs from original scope)
        if (rootScopeBreakInfo.GetReason() != ScopeBreakReason::NONE && !IsKernelSetMatch(scopeInfo, originalKernelSets, graph)) {
            SK_LOGI("    breakReason=[%s], scopeName=[%s]", rootScopeBreakInfo.Format().c_str(), scopeNames.c_str());
        }
    }
}

void PrintOriginalScopes(const SuperKernelGraph& graph) 
{
    const auto& originalScopeInfos = graph.GetOriginalScopeInfos();
    SK_LOGI("original scopes before fusion:");
    for (const auto& origScope : originalScopeInfos) {
        std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(origScope.GetScopeBitFlags(), graph);
        SK_LOGI("  scopeId=%u, scopeFlags=%s, scopeNames=[%s], totalNodes=%zu",
                origScope.scopeId, graph.BitsetToString(origScope.GetScopeBitFlags()).c_str(),
                scopeNames.c_str(), origScope.nodeIds.size());
        PrintNodeIdsInBatches(origScope.nodeIds);
    }
}

bool DumpFusedGraphToJson(const SuperKernelGraph& graph, const std::vector<SuperKernelScopeInfo>& scopeInfos) 
{
    if (!sk::logger::FileLogger::Instance().IsEnabled()) {
        return true;  // Kernel meta save is disabled, skip
    }
    SK_LOGI("Starting to dump fused SuperKernel graph to JSON with nested structure");

    std::string jsonPath;
    bool pathOk = GetFusedGraphJsonOutputPath(graph, jsonPath);
    if (!pathOk) {
        SK_LOGE("Failed to get fused graph json output path");
        return false;
    }
    if (jsonPath.empty()) {
        SK_LOGE("Failed to get fused graph JSON output path");
        return false;
    }

    try {
        Json fusedGraphJson = BuildFusedGraphJson(graph, scopeInfos);

        if (!WriteJsonToFile(fusedGraphJson, jsonPath)) {
            return false;
        }

        SK_LOGI("Successfully dumped fused graph with %zu nodes (%zu scopes) to JSON file: %s",
                fusedGraphJson["nodes"].size(), scopeInfos.size(), jsonPath.c_str());
        return true;

    } catch (const std::exception& e) {
        SK_LOGE("Exception while dumping fused graph to JSON: %s", e.what());
        return false;
    }
}

Json SkTaskToQueueJson(const SkTask& aicTask, const SkTask& aivTask, uint16_t scopeId) 
{
    Json taskQueueJson;
    taskQueueJson["scopeId"] = scopeId;
    taskQueueJson["taskQueues"]["aic"] = SkTaskToJson(aicTask);
    taskQueueJson["taskQueues"]["aiv"] = SkTaskToJson(aivTask);
    return taskQueueJson;
}

bool DumpAllTaskQueuesToJson(const SuperKernelGraph& graph, 
                             const std::unordered_map<std::string, Json>& taskQueueJsons)
{
    if (!sk::logger::FileLogger::Instance().IsEnabled()) {
        return true;  // Kernel meta save is disabled, skip
    }
    
    std::string metaDir = CreateSkMetaDirectory(graph.GetModelRI());
    if (metaDir.empty()) {
        SK_LOGE("Failed to create sk_meta directory for task queue JSON dump");
        return false;
    }
    std::string jsonPath = metaDir + "/sk_task_queue.json";
    
    try {
        Json rootJson;
        rootJson["version"] = "1.0";
        rootJson["description"] = "SuperKernel Task Queue Information";
        rootJson["modelRI"] = std::to_string(reinterpret_cast<uintptr_t>(graph.GetModelRI()));
        rootJson["scopeCount"] = taskQueueJsons.size();
        
        Json scopesArray = Json::array();
        for (const auto& [scopeId, taskQueueJson] : taskQueueJsons) {
            scopesArray.push_back(taskQueueJson);
        }
        rootJson["scopes"] = scopesArray;
        
        std::ofstream outFile(jsonPath);
        if (!outFile.is_open()) {
            SK_LOGE("Failed to open task queue JSON file for writing: %s", jsonPath.c_str());
            return false;
        }
        
        outFile << rootJson.dump(2);
        outFile.close();
        
        SK_LOGI("Successfully dumped %zu task queues to JSON file: %s", taskQueueJsons.size(), jsonPath.c_str());
        return true;
        
    } catch (const std::exception& e) {
        SK_LOGE("Exception while dumping task queues to JSON: %s", e.what());
        return false;
    }
}

bool DumpGraphNodesToJson(const SuperKernelGraph& graph) 
{
    if (!sk::logger::FileLogger::Instance().IsEnabled()) {
        return true;  // Kernel meta save is disabled, skip
    }
    SK_LOGI("Starting to dump SuperKernel graph nodes to JSON");
    
    std::string jsonPath = GetJsonOutputPath(graph);
    if (jsonPath.empty()) {
        SK_LOGE("Failed to get JSON output path");
        return false;
    }
    
    try {
        Json graphNodesJson = BuildGraphNodesJson(graph);
        
        if (!WriteJsonToFile(graphNodesJson, jsonPath)) {
            SK_LOGE("Failed to write JSON to file: %s", jsonPath.c_str());
            return false;
        }
        
        SK_LOGI("Successfully dumped %zu nodes to JSON file: %s",
                graphNodesJson["nodes"].size(), jsonPath.c_str());
        return true;
        
    } catch (const std::exception& e) {
        SK_LOGE("Exception while dumping graph to JSON: %s", e.what());
        return false;
    }
}

bool DumpRawTaskJson(aclmdlRI model, const SuperKernelOptionsManager& opts, const std::string& metaDir, const std::string& filename)
{
    if (!sk::logger::FileLogger::Instance().IsEnabled()) {
        return true;  // Kernel meta save is disabled, skip
    }

    if (filename.empty()) {
        SK_LOGE("DumpRawTaskJson failed: filename is empty");
        return false;
    }

    SK_LOGI("Start creating temp graph for %s dump...", filename.c_str());
    SuperKernelGraph tempGraph(model, opts);
    if (!tempGraph.InitFromModelRI()) {
        SK_LOGE("Failed to init temp graph for %s dump", filename.c_str());
        return false;
    }
    SK_LOGI("End creating temp graph for %s dump", filename.c_str());

    std::string fullFilename = filename + ".json";
    std::string testJsonPath = metaDir + "/" + fullFilename;
    std::ofstream testJsonFile(testJsonPath);
    if (testJsonFile.is_open()) {
        nlohmann::ordered_json graphJson = tempGraph.ToJson();
        graphJson["options"] = opts.ToJson();
        testJsonFile << graphJson.dump(2);  // Pretty print with 2-space indentation
        testJsonFile.close();
        SK_LOGI("Successfully dumped graph to %s: %s", fullFilename.c_str(), testJsonPath.c_str());
    } else {
        SK_LOGE("Failed to open %s for writing: %s", fullFilename.c_str(), testJsonPath.c_str());
        return false;
    }
    return true;
}
