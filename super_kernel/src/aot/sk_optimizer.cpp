/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>
#include "sk_optimizer.h"
#include "sk_scope_split.h"
#include "sk_scope_postprocess.h"
#include "sk_task_builder.h"
#include "sk_log.h"
#include "aprof_pub.h"
#include "securec.h"

bool SuperKernelOptimizer::Update(SuperKernelProcessedScopeInfo& processedScopeInfo, SuperKernelGraph& graph,
                                  const SkLaunchInfo& launchInfo)
{
    SK_LOGI("scope update begin: streamCount=%zu", processedScopeInfo.updateStreamInfos.size());
    bool skMainNodeUpdated = false;
    size_t updateTotalCount = 0;

    for (auto& streamInfo : processedScopeInfo.updateStreamInfos) {
        SK_LOGI("update stream begin: streamId=%u, headNodeId=%lu, tailNodeId=%lu, nodeSize=%lu, customParamSize=%zu",
                streamInfo.streamIdx, streamInfo.headNodeIdx, streamInfo.tailNodeIdx, streamInfo.nodeSize,
                streamInfo.customParams.size());
        size_t customParamSize = streamInfo.customParams.size();
        if (streamInfo.nodeSize < customParamSize) {
            SK_LOGE("node size is less than custom params size: nodeSize=%lu, customParamSize=%zu", streamInfo.nodeSize,
                    customParamSize);
            return false;
        }

        uint64_t curNodeId = streamInfo.headNodeIdx;
        size_t eventCnt = 0;

        while (curNodeId != INVALID_TASK_ID) {
            auto* node = graph.GetNodeById(curNodeId);
            if (node == nullptr) {
                SK_LOGE("node not found during stream-based update: nodeId=%lu, streamIdx=%u", curNodeId,
                        streamInfo.streamIdx);
                return false;
            }
            UpdateContext ctx;
            if (eventCnt < customParamSize) {
                // set front node for stream sync
                auto& customParams = streamInfo.customParams[eventCnt++];
                ctx.customParams = &customParams;
            } else if (curNodeId == processedScopeInfo.skMainNodeId) {
                // search node for sk launch after front node
                if (!skMainNodeUpdated) {
                    skMainNodeUpdated = true;
                    ctx.launchInfo = const_cast<SkLaunchInfo*>(&launchInfo);
                } else {
                    SK_LOGI("repeat find sk launch node, skip update kernel and set invalid node");
                }
            }

            ++updateTotalCount;
            if (!node->Update(ctx)) { // default invalid
                SK_LOGE("node update failed: nodeId=%lu, streamIdx=%u", curNodeId, streamInfo.streamIdx);
                return false;
            }

            if (curNodeId == streamInfo.tailNodeIdx) {
                break;
            }
            curNodeId = node->GetNextNodeId();
        }
        SK_LOGI("update stream end: streamId=%u, visitedNodes=%zu", streamInfo.streamIdx, eventCnt);
    }

    if (!skMainNodeUpdated) {
        SK_LOGE("not find sk launch node, sk optimize failed");
        return false;
    }

    SK_LOGI("scope update finished: update total nodes=%zu", updateTotalCount);

    return true;
}

bool SuperKernelOptimizer::UpdateScopeNode(SuperKernelProcessedScopeInfo& processedScopeInfo, SuperKernelGraph& graph)
{
    if (processedScopeInfo.nodes.empty()) {
        for (auto& streamInfo : processedScopeInfo.updateStreamInfos) {
            uint64_t curNodeId = streamInfo.headNodeIdx;
            while (curNodeId != INVALID_TASK_ID) {
                auto* node = graph.GetNodeById(curNodeId);
                if (!node->IsScopeNode()) {
                    continue;
                }
                UpdateContext ctx;
                if (!node->Update(ctx)) { // default invalid
                    SK_LOGE("node update failed: nodeId=%lu, streamIdx=%u", curNodeId, streamInfo.streamIdx);
                    return false;
                }
            }
        }
    }
    return true;
}

// Schedule task-stream nodes for super-kernel launch.
bool SuperKernelOptimizer::Schedule(SuperKernelProcessedScopeInfo& processedScopeInfo, SuperKernelGraph& graph,
                                    SkTaskBuilder& builder)
{
    const auto& taskNodes = processedScopeInfo.nodes;
    if (taskNodes.empty()) {
        SK_LOGI("no tasks for super kernel optimization: scope has 0 nodes for optimization, skip scheduling and updating");
        if (!UpdateScopeNode(processedScopeInfo, graph)) {
            SK_LOGE("scope node update failed for empty task list");
            return false;
        }
        return true;
    }

    std::vector<SuperKernelBaseNode*> customTasks;
    customTasks.reserve(processedScopeInfo.eventNodes.size());
    for (const auto& eventNode : processedScopeInfo.eventNodes) {
        customTasks.emplace_back(eventNode.get());
    }

    SK_LOGI("schedule scope: taskCount=%zu, customTaskCount=%zu, updateStreamCount=%zu", taskNodes.size(),
            customTasks.size(), processedScopeInfo.updateStreamInfos.size());

    SkLaunchInfo launchInfo = builder.Build(taskNodes, customTasks);

    if (!SkProfiling(processedScopeInfo, launchInfo, graph)) {
        SK_LOGE("SkProfiling failed");
        return false;
    }

    if (launchInfo.entryInfo.skEntryFunc == nullptr || launchInfo.devArgs.Get() == nullptr) {
        SK_LOGE("schedule failed: build launch info failed");
        return false;
    }
    SK_LOGI("schedule scope: build finished, entryType=%s, entryFuncHandle=%p",
            to_string(launchInfo.entryInfo.entryType), launchInfo.entryInfo.skEntryFunc);

    if (!Update(processedScopeInfo, graph, launchInfo)) {
        SK_LOGE("schedule failed: scope update failed");
        return false;
    }
    return true;
}


const char* GetEntryFuncNameByOpType(SkKernelType& opType) {
    // sk_entry_aiv
    if (opType == SkKernelType::AIV_ONLY || opType == SkKernelType::MIX_AIV_1_0) {
        return "sk_entry_aiv";
    }

    // sk_entry_aic
    if (opType == SkKernelType::AIC_ONLY || opType == SkKernelType::MIX_AIC_1_0) {
        return "sk_entry_aic";
    }
    // sk_entry_mix11
    if (opType == SkKernelType::MIX_AIC_1_1) {
        return "sk_entry_mix11";
    }
    // sk_entry_mix12
    if (opType == SkKernelType::MIX_AIC_1_2) {
        return "sk_entry_mix12";
    }

    // Unknown opType
    SK_LOGE("opType is not in the enum class SkKernelType");
    return nullptr;
}

bool SuperKernelOptimizer::SkProfiling(const SuperKernelProcessedScopeInfo &scopeInfo, SkLaunchInfo &launchInfo,
                                        SuperKernelGraph& graph) {
    SkHostEntryInfo& skEntryInfo = launchInfo.entryInfo;
    
    uint32_t taskType = MSPROF_GE_TASK_TYPE_MIX_AIC; //全部填mix_aic 4
    uint32_t opFlag = 0; //记录op属性标记的bitmap，bit0代表是否使能了HF32
    std::string combinedAttrIdStr;
    uint32_t maxTensorNum = SHAPE_MAX_TENSOR_NUM;

    // ====== 第一遍遍历：计算总 tensor 数量，并收集 NODE_KERNEL 类型的节点 ======
    uint32_t totalTensorNum = 0;
    std::vector<SuperKernelBaseNode*> kernelNodes;
    for (size_t i = 0; i < scopeInfo.nodes.size(); ++i) {
        SuperKernelBaseNode* node = scopeInfo.nodes[i];
        if (node == nullptr) {
            SK_LOGE("[sk profiling] Failed to get node, node is nullptr");
            return false;
        }
        SkNodeType nodeType = node->GetNodeType();
        if (nodeType == SkNodeType::NODE_KERNEL) {
            const NodeInfos& nodeInfos = node->GetNodeInfos();
            const KernelInfos& kernelInfos = nodeInfos.kernelInfos;
            if (kernelInfos.opInfoPtr != nullptr) {
                const CacheopInfoBasic* cacheInfo = 
                    static_cast<const CacheopInfoBasic*>(kernelInfos.opInfoPtr);
                totalTensorNum += cacheInfo->tensorNum;
            }

            if (totalTensorNum > maxTensorNum) {
                totalTensorNum = maxTensorNum; // 截断到最大值
                break;
            }
            kernelNodes.push_back(node);
        }
    }

    size_t totalSize = sizeof(CacheopInfoBasic) + totalTensorNum * sizeof(MsrofTensorData);
    auto ptr = std::make_unique<uint8_t[]>(totalSize);
    if (ptr == nullptr) {
        SK_LOGE("[sk profiling] Failed to allocate memory for launchInfo.cacheInfo\n");
        return false;
    }

    // ptr生命周期跟随graph，因为runtime在aclmdlRIUpdate才会把shape信息copy走
    launchInfo.cacheInfo = ptr.get();
    graph.AddShapeInfoPtr(std::move(ptr));
    errno_t ret = memset_s(launchInfo.cacheInfo, totalSize, 0, totalSize);
    if (ret != EOK) {
        SK_LOGE("[sk profiling] memset_s launchInfo.cacheInfo failed, ret=%d\n",  ret);
    }
    launchInfo.cacheopInfoSize = totalSize;
    uint8_t *dest = static_cast<uint8_t *>(launchInfo.cacheInfo);
    uint64_t destOffset = sizeof(CacheopInfoBasic); // 预留 CacheopInfoBasic 的空间，后续填充

    // ====== 第二遍遍历：从 kernelNodes 收集数据 ======
    for (size_t i = 0; i < kernelNodes.size(); ++i) {
        SuperKernelBaseNode* node = kernelNodes[i];
        
        const NodeInfos& nodeInfos = node->GetNodeInfos();
    
        const KernelInfos& kernelInfos = nodeInfos.kernelInfos;
        if (kernelInfos.opInfoPtr != nullptr) {
            const CacheopInfoBasic* cacheInfo = 
                static_cast<const CacheopInfoBasic*>(kernelInfos.opInfoPtr);
            if (kernelInfos.opInfoSize >= (sizeof(CacheopInfoBasic) + sizeof(MsrofTensorData) * cacheInfo->tensorNum)) {
                // attrId
                char* attrIdStr = MsprofId2Str(cacheInfo->attrId);
                if (!combinedAttrIdStr.empty()) {
                    combinedAttrIdStr += "|";
                }
                combinedAttrIdStr += attrIdStr;
                
                // opFlag 取值只有0和1
                opFlag = opFlag || cacheInfo->opFlag;
                
                // ====== 复制 tensorData 到 launchInfo.cacheInfo ======
                for (uint32_t t = 0; t < cacheInfo->tensorNum; ++t) {
                    const MsrofTensorData& tensor = cacheInfo->tensorData[t];
                    MsrofTensorData msTensor;
                    msTensor.tensorType = tensor.tensorType;
                    msTensor.format = tensor.format;
                    msTensor.dataType = tensor.dataType;
                    for (int s = 0; s < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++s) {
                        msTensor.shape[s] = tensor.shape[s];
                    }
                    errno_t ret = memcpy_s(dest + destOffset, totalSize - destOffset,
                                            &msTensor, sizeof(MsrofTensorData));
                    if (ret != EOK) {
                        SK_LOGE("[sk profiling] memcpy_s failed, ret=%d\n",  ret);
                        return false;
                    }
                    destOffset += sizeof(MsrofTensorData);
                }
            }
            else {
                SK_LOGE("[sk profiling] warning: kernelInfos.opInfoSize should be greater than or equal to kernelInfos.opInfoPtr \n");
            }
        }
    }

    // 将拼接后的字符串转换回 uint64_t
    uint64_t combinedAttrId = combinedAttrIdStr.empty() ? 0 : 
        MsprofStr2Id(combinedAttrIdStr.c_str(), combinedAttrIdStr.length());

    CacheopInfoBasic cacheopInfoBasic;
    cacheopInfoBasic.taskType = taskType;
    // numBlocks 编码：高16位表示mix模式类型，低16位为实际numBlocks
    // sk_entry_mix11 -> 高16位=1, sk_entry_mix12 -> 高16位=2
    uint32_t numBlocks = skEntryInfo.numBlocks;
    if (skEntryInfo.entryType == SkKernelType::MIX_AIC_1_1) {
        cacheopInfoBasic.numBlocks = (1U << 16) | (numBlocks & 0xFFFF);
    } else if (skEntryInfo.entryType == SkKernelType::MIX_AIC_1_2) {
        cacheopInfoBasic.numBlocks = (2U << 16) | (numBlocks & 0xFFFF);
    } else {
        cacheopInfoBasic.numBlocks = numBlocks;
    }
    const char* skEntryFuncName = GetEntryFuncNameByOpType(skEntryInfo.entryType);
    if (skEntryFuncName != nullptr) {
        cacheopInfoBasic.nodeId = MsprofStr2Id(skEntryFuncName, strlen(skEntryFuncName));
    } else {
        SK_LOGE("[sk profiling] Failed to get entry func name\n");
        return false;
    }
    const char* opTypeStr = "SuperKernel";
    cacheopInfoBasic.opType = MsprofStr2Id(opTypeStr, strlen(opTypeStr));
    cacheopInfoBasic.attrId = combinedAttrId;
    cacheopInfoBasic.opFlag = opFlag;
    cacheopInfoBasic.tensorNum = totalTensorNum; 
    ret = memcpy_s(dest, sizeof(CacheopInfoBasic), &cacheopInfoBasic, sizeof(CacheopInfoBasic));
    if (ret != EOK) {
        SK_LOGE("[sk profiling] memcpy_s cacheopInfoBasic failed, ret=%d\n",  ret);
    }
    return true;
}

bool SuperKernelOptimizer::Process(SuperKernelGraph& graph)
{
    // Split graph into multiple scopes.
    SuperKernelScopeSplitter splitter(graph);
    if (splitter.SplitGraph()) {
        SK_LOGI("graph split into %zu scopes", splitter.GetScopeInfos().size());
    } else {
        SK_LOGE("graph split failed or no scopes found: cannot proceed with super kernel optimization");
        return false;
    }
    auto& scopeInfos = splitter.GetScopeInfos();

    SkTaskBuilder builder(opts, graph);
    SuperKernelScopePostProcessor postProcessor(graph);

    // Process each scope sequentially.
    size_t scopeIndex = 0;
    for (auto& scopeInfo : scopeInfos) {
        SK_LOGI("process scope begin: scopeIndex=%zu", scopeIndex);
        SuperKernelProcessedScopeInfo processedScopeInfo = postProcessor.PostProcess(scopeInfo);
        if (!Schedule(processedScopeInfo, graph, builder)) {
            SK_LOGE("process scope failed: scopeIndex=%zu, schedule/update returned false", scopeIndex);
            return false;
        }
        ++scopeIndex;
    }
    SK_LOGI("super kernel process finished: scopeCount=%zu", scopeInfos.size());
    return true;
}
