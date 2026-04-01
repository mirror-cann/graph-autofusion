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
 * \file sk_graph.h
 * \brief
 */

#ifndef __SK_GRAPH_H__
#define __SK_GRAPH_H__

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdint>

#include "sk_log.h"
#include "sk_types.h"
#include "sk_node.h"
#include "acl/acl.h"

class SuperKernelNodeFactory {
public:
    static std::unique_ptr<SuperKernelBaseNode> CreateNode(std::unique_ptr<aclmdlRITask> task, aclmdlRITaskType taskType,
                                                           uint64_t nodeIdx, uint64_t streamIdxInGraph, int32_t streamId, uint64_t preNodeId);
};

struct EventInfos {
    uint64_t notifyNodeId = INVALID_TASK_ID;
    std::unordered_set<uint64_t> resetNodeIdList;
    std::unordered_set<uint64_t> waitNodeIdList;
};

struct MemoryInfos {
    std::unordered_set<uint64_t> writeNodeIdList;
    std::unordered_set<uint64_t> waitNodeIdList;
};

class SuperKernelGraph {
public:
    aclError Update();

    SuperKernelGraph() = default;
    ~SuperKernelGraph() = default;
    SuperKernelGraph(const SuperKernelGraph&) = delete;
    SuperKernelGraph& operator=(const SuperKernelGraph&) = delete;
    SuperKernelGraph(SuperKernelGraph&&) = default;
    SuperKernelGraph& operator=(SuperKernelGraph&&) = default;
    SuperKernelGraph(aclmdlRI modelRI) : modelRI(modelRI) {}
    bool InitSKGraph();

    SuperKernelBaseNode* GetNodeById(uint64_t nodeId) const;
    aclmdlRI GetModelRI() const { return modelRI; }
    const std::vector<uint64_t>& GetHeadNodes() const
    {
        return headNodes;
    }
    const std::vector<uint64_t>& GetNodeSizeInStream() const
    {
        return nodeSizeInStream;
    }
    const std::vector<aclrtStream>& GetStreams() const
    {
        return streams;
    }

    aclrtStream GetStreamByIndex(uint32_t streamIdx) const;

    // Get all nodeIds sorted in ascending order
    std::vector<uint64_t> GetSortedNodeIds() const;

    // Get event info by event id
    const EventInfos* GetEventInfo(uint64_t eventId) const
    {
        auto it = eventToNodes.find(eventId);
        return it != eventToNodes.end() ? &it->second : nullptr;
    }

    // Get scope name by scope index
    bool GetScopeNameByIdx(uint32_t scopeIdx, std::string& scopeName) const
    {
        auto it = scopeIdxToName.find(scopeIdx);
        if (it != scopeIdxToName.end()) {
            scopeName = it->second;
            return true;
        }
        return false;
    }

    // Add shape info memory block, managed by graph lifecycle
    void AddShapeInfoPtr(std::unique_ptr<uint8_t[]> ptr) {
        shapeInfoPtrList.emplace_back(std::move(ptr));
    }

    // Clear all shape info memory blocks
    void ClearShapeInfoPtrList() {
        shapeInfoPtrList.clear();
    }
    std::string BitsetToString(const std::bitset<MAX_SCOPE_NUM>& bitset) const;
    // Expand update nodes
    bool ExpandUpdateNodes(std::vector<SuperKernelBaseNode*>& needUpdateNodes);
    void SetUpdateFlag(bool flag)
    {
        needUpdate = flag;
    }

private:
    bool AddNode(std::unique_ptr<SuperKernelBaseNode> node);
    bool AddEventAssociateNotify(uint64_t eventId, SuperKernelBaseNode* node);
    bool AddEventAssociateWait(uint64_t eventId, SuperKernelBaseNode* node);
    bool AddEventAssociateReset(uint64_t eventId, SuperKernelBaseNode* node);
    bool AddMemoryAssociateWrite(uint64_t eventId, SuperKernelBaseNode* node);
    bool AddMemoryAssociateWait(uint64_t eventId, SuperKernelBaseNode* node);
    bool AddEventAssociate();
    void BuildEventNodeAssociations();
    bool PostProcessMemoryNode();
    bool ProcessMemoryWriteNodes(const uint64_t eventId, const MemoryInfos& memoryInfo,
                               const uint64_t memoryWaitValue, const uint32_t waitFlag);
    void UpdateNodeScopeBitFlags();
    std::unordered_map<uint64_t, std::unique_ptr<SuperKernelBaseNode>> graphMap;
    std::unordered_map<uint64_t, EventInfos> eventToNodes;
    std::unordered_map<uint64_t, MemoryInfos> memoryToNodes;
    std::vector<uint64_t> headNodes;
    std::vector<uint64_t> nodeSizeInStream;
    std::vector<aclrtStream> streams;
    aclmdlRI modelRI;
    friend class SuperKernelOptimizer;
    std::unordered_map<std::string, uint32_t> scopeNameToIdx;    ///< scopeName -> scopeIdx
    std::unordered_map<uint32_t, std::string> scopeIdxToName;    ///< scopeIdx -> scopeName (reverse mapping)
    std::vector<std::unique_ptr<uint8_t[]>> shapeInfoPtrList;    ///< profiling sk shape info memory, lifecycle follows graph
    std::unordered_set<SuperKernelBaseNode*> needUpdateNodes;      ///< event nodes that need to be marked with scope bit flags after scope processing 
    bool needUpdate = false;
};

#endif // __SK_GRAPH_H__
