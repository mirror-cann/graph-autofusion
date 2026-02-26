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
 * \file sk_node.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "sk_log.h"
#include "sk_types.h"
#include "rt_sk_intf.h"


// Forward declaration
class SuperKernelGraph;

struct ResolvedFunctionInfo {
    uint64_t funcAddr[2] = {0, 0};
    uint64_t prefetchCnt[2] = {0, 0};
    aclrtFuncHandle funcHdl = nullptr;
    aclrtFuncHandle oriFuncHdl = nullptr;
};

constexpr size_t kMaxSplitBinCount = 4;

struct KernelInfos {
    SkKernelType kernelType = SkKernelType::DEFAULT;
    uint32_t skKernelType = 0;
    uint32_t taskRatio[2] = {0, 0};
    uint32_t numBlocks = 0;
    const void *devArgs = nullptr;
    std::string funcName;
    aclrtBinHandle binHdl = nullptr;
    aclrtFuncHandle funcHdl = nullptr;
    aclrtFuncHandle oriFuncHdl = nullptr;
    ResolvedFunctionInfo resolvedFuncs[4];
};

struct SyncInfos {
    uint64_t eventId = INVALID_TASK_ID;
    uint64_t waitNodeId = INVALID_TASK_ID;
    uint64_t notifyNodeId = INVALID_TASK_ID;
    void* addrValue = nullptr;
};


struct NodeInfos {
    union {
        KernelInfos kernelInfos;
        SyncInfos syncInfos;
    };

    NodeInfos() : kernelInfos() {}
    ~NodeInfos() {}
};


// Base Node Class
class SuperKernelBaseNode {
public:
    SuperKernelBaseNode(std::unique_ptr<aclrtTask> inputOriginTask, SkNodeType inputNodeType, uint64_t inputNodeIdxInStream, uint64_t inputStreamIdxInGraph, uint64_t inputPreNodeId)
        : originTask(std::move(inputOriginTask)),
          nodeType(inputNodeType),
          nodeIdxInStream(inputNodeIdxInStream),
          streamIdxInGraph(inputStreamIdxInGraph),
          preNodeId(inputPreNodeId),
          nextNodeId(INVALID_TASK_ID),
          isVisited(false),
          isFusible(false) { }
    virtual ~SuperKernelBaseNode() = default;

    virtual bool InitNode();

    // Accessors
    uint32_t GetStreamIdxInGraph() const { return streamIdxInGraph; }
    uint64_t GetNodeIdxInStream() const { return nodeIdxInStream; }
    uint64_t GetNodeId() const { return nodeId; }
    bool IsFusible() const { return isFusible; }
    void SetNodeId(uint64_t inputNodeId) { nodeId = inputNodeId; }

    // Node Relationships
    void SetPreNodeId(uint64_t inputPreNodeId) { preNodeId = inputPreNodeId; }
    void SetNextNodeId(uint64_t inputNextNodeId) { nextNodeId = inputNextNodeId; }
    uint64_t GetPreNodeId() const { return preNodeId; }

    uint64_t GetNextNodeId() const { return nextNodeId; }

    // SuperKernelKernelNode specific accessors
    virtual uint32_t GetNumBlocks() const { return 0; }
    virtual SkKernelType GetKernelType() const { return SkKernelType::DEFAULT; }

    // SuperKernelEventNode/SuperKernelMemoryNode specific accessors
    virtual uint64_t GetEventId() const { return INVALID_TASK_ID; }

    // SuperKernelEventNotifyNode/SuperKernelMemoryNotifyNode specific accessors
    virtual uint64_t GetNotifyNodeId() const { return INVALID_TASK_ID; }

    // SuperKernelEventWaitNode/SuperKernelMemoryWaitNode specific accessors
    virtual uint64_t GetWaitNodeId() const { return INVALID_TASK_ID; }

    virtual const NodeInfos& GetNodeInfos() const { return nodeInfos; }

    virtual bool InValidateNode() = 0;

    // Task Type
    SkNodeType GetNodeType() const { return nodeType; }

    // Visitation State
    bool IsVisited() const { return isVisited; }
    void SetVisited(bool inputIsVisited) { isVisited = inputIsVisited; }

protected:
    uint32_t streamIdxInGraph;
    uint64_t nodeIdxInStream;
    uint64_t nodeId;
    uint64_t preNodeId;
    uint64_t nextNodeId;
    std::unique_ptr<aclrtTask> originTask;
    SkNodeType nodeType;
    bool isVisited;
    bool isFusible = false;
    NodeInfos nodeInfos;
};


// Derived Node Classes

class SuperKernelKernelNode : public SuperKernelBaseNode {
public:
    using SuperKernelBaseNode::SuperKernelBaseNode;
    bool InitNode() override;
    uint32_t GetNumBlocks() const override { return nodeInfos.kernelInfos.numBlocks; }
    SkKernelType GetKernelType() const override { return nodeInfos.kernelInfos.kernelType; }
    bool InValidateNode() override;
private:
    aclrtTaskKernelParams kernelParams;
};

class SuperKernelEventNode : public SuperKernelBaseNode {
public:
    using SuperKernelBaseNode::SuperKernelBaseNode;
    uint64_t GetEventId() const override { return nodeInfos.syncInfos.eventId; }
    bool InitNode() override;
    bool InValidateNode() override;
private:
    aclrtTaskEventParams eventParams;
};

class SuperKernelEventNotifyNode : public SuperKernelEventNode {
public:
    using SuperKernelEventNode::SuperKernelEventNode;
    uint64_t GetWaitNodeId() const override { return nodeInfos.syncInfos.waitNodeId; }
};

class SuperKernelEventWaitNode : public SuperKernelEventNode {
public:
    using SuperKernelEventNode::SuperKernelEventNode;
    uint64_t GetNotifyNodeId() const override { return nodeInfos.syncInfos.notifyNodeId; }
};

class SuperKernelMemoryNode : public SuperKernelBaseNode {
public:
    using SuperKernelBaseNode::SuperKernelBaseNode;
    uint64_t GetEventId() const override { return nodeInfos.syncInfos.eventId; }
    bool InitNode() override;
    bool InValidateNode() override;
private:
     aclrtTaskMemValueParams memValueParams;
};

class SuperKernelMemoryNotifyNode : public SuperKernelMemoryNode {
public:
    using SuperKernelMemoryNode::SuperKernelMemoryNode;
    uint64_t GetWaitNodeId() const override { return nodeInfos.syncInfos.waitNodeId; }
};

class SuperKernelMemoryWaitNode : public SuperKernelMemoryNode {
public:
    using SuperKernelMemoryNode::SuperKernelMemoryNode;
    uint64_t GetNotifyNodeId() const override { return nodeInfos.syncInfos.notifyNodeId; }
};

class SuperKernelMemoryResetNode : public SuperKernelMemoryNode {
public:
    using SuperKernelMemoryNode::SuperKernelMemoryNode;
};

class SuperKernelDefaultNode : public SuperKernelBaseNode {
public:
    using SuperKernelBaseNode::SuperKernelBaseNode;
    bool InitNode() override;
    bool InValidateNode() override;
};
