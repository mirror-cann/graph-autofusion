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

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>

#include "sk_log.h"
#include "sk_types.h"
#include "sk_node.h"
#include "rt_sk_intf.h"

class SuperKernelNodeFactory {
public:
    static std::unique_ptr<SuperKernelBaseNode> CreateNode(std::unique_ptr<aclrtTask> task, aclrtTaskType taskType, uint64_t nodeIdx, uint64_t streamId, uint64_t preNodeId);
};

struct EventInfos {
    uint64_t notifyNodeId = INVALID_TASK_ID;
    uint64_t resetNodeId = INVALID_TASK_ID;
    std::unordered_set<uint64_t> waitNodeIdList;
};

class SuperKernelGraph {
public:
    void Update();

    SuperKernelGraph() = default;
    ~SuperKernelGraph() = default;
    SuperKernelGraph(const SuperKernelGraph&) = delete;
    SuperKernelGraph& operator=(const SuperKernelGraph&) = delete;
    SuperKernelGraph(SuperKernelGraph&&) = default;
    SuperKernelGraph& operator=(SuperKernelGraph&&) = default;
    SuperKernelGraph(aclmdlRI modelRI) : modelRI(modelRI) {}
    bool InitSKGraph();

    SuperKernelBaseNode* GetNodeById(uint64_t nodeId) const;
    const std::vector<uint64_t>& GetHeadNodes() const { return headNodes; }
    const std::vector<uint64_t>& GetNodeSizeInStream() const { return nodeSizeInStream; }
    const std::vector<aclrtStream>& GetStreams() const { return streams; }

    aclrtStream GetStreamByIndex(uint32_t streamIdx) const;

private:
    uint64_t AllocateNodeId() {
        return nextNodeId++;
    }

    bool AddNode(std::unique_ptr<SuperKernelBaseNode> node);
    bool AddEventAssociateNotify(uint64_t eventId, uint64_t nodeId);
    bool AddEventAssociateWait(uint64_t eventId, uint64_t nodeId);
    bool AddEventAssociateReset(uint64_t eventId, uint64_t nodeId);

    std::unordered_map<uint64_t, std::unique_ptr<SuperKernelBaseNode>> graphMap;
    std::unordered_map<uint64_t, EventInfos> eventToNodes;
    std::vector<uint64_t> headNodes;
    std::vector<uint64_t> nodeSizeInStream;
    std::vector<aclrtStream> streams;
    aclmdlRI modelRI;
    uint64_t nextNodeId = 0;
    friend class SuperKernelScopeSplitter;
};
