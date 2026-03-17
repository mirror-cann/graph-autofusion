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

#ifndef __SK_NODE_H__
#define __SK_NODE_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <bitset>

#include "sk_log.h"
#include "sk_types.h"
#include "acl/acl.h"

// Forward declaration
class SuperKernelGraph;
struct SkLaunchInfo;


// Update context for node update operations
struct UpdateContext {
    SkLaunchInfo* launchInfo = nullptr;
    // Feature(aclmdIRITaskParams): Replace this raw aclrtTaskEventParams pointer
    // with aclmdIRITaskParams carrier when post-process switches to IR-task flow.
    // Optional event update payload for stream-based update path.
    aclmdlRITaskParams* customParams = nullptr;
};

struct SknlMapInfo {
    uint64_t cap;
    void* globalFunc;
    void* sknlFunc[4];
};

struct ResolvedFunctionInfo {
    uint64_t funcAddr[2] = {0, 0};
    uint64_t prefetchCnt[2] = {0, 0};
};

constexpr size_t kMaxSplitBinCount = 4;

struct KernelInfos {
    SkKernelType kernelType = SkKernelType::DEFAULT;
    uint32_t taskRatio[2] = {0, 0};
    uint32_t numBlocks = 0;
    uint32_t vecNum = 0;      ///< Number of vector cores required
    uint32_t cubeNum = 0;     ///< Number of cube cores required
    const void *devArgs = nullptr;
    std::string funcName;
    aclrtBinHandle binHdl = nullptr;
    aclrtFuncHandle funcHdl = nullptr;
    ResolvedFunctionInfo resolvedFuncs[4];
};

struct SyncInfos {
    uint64_t eventId = INVALID_TASK_ID;
    // For notify nodes: empty (not used)
    // For wait nodes: this is the ID of the notify node this wait node waits on
    uint64_t correspondingNotifyNodeId = INVALID_TASK_ID;
    // For notify nodes: list of all wait node IDs that wait on this notify
    // For wait nodes: empty (not used)
    std::vector<uint64_t> correspondingWaitNodeIds;
    void* addrValue = nullptr;
};

struct NodeInfos {
    KernelInfos kernelInfos;
    SyncInfos syncInfos;
};

// Base Node Class
class SuperKernelBaseNode {
public:
    SuperKernelBaseNode(std::unique_ptr<aclmdlRITask> inputOriginTask, aclmdlRITaskType inputRtNodeType, uint64_t inputNodeIdxInStream, uint64_t inputStreamIdxInGraph, uint64_t inputPreNodeId)
        : originTask(std::move(inputOriginTask)),
          rtNodeType(inputRtNodeType),
          nodeIdxInStream(inputNodeIdxInStream),
          streamIdxInGraph(inputStreamIdxInGraph),
          preNodeId(inputPreNodeId),
          nextNodeId(INVALID_TASK_ID),
          isVisited(false),
          isFusible(false),
          isScopeNode(false),
          notifyExpandVecNum(0),
          notifyExpandCubeNum(0) { }
    virtual ~SuperKernelBaseNode() = default;
    virtual bool InitNode();

    virtual const std::string GetNodeName() const = 0;

    // Accessors
    uint32_t GetStreamIdxInGraph() const
    {
        return streamIdxInGraph;
    }
    uint64_t GetNodeIdxInStream() const
    {
        return nodeIdxInStream;
    }
    uint64_t GetNodeId() const
    {
        return nodeId;
    }
    bool IsFusible() const
    {
        return isFusible;
    }
    void SetIsFusible(bool fusible)
    {
        isFusible = fusible;
    }
    void SetNodeId(uint64_t inputNodeId)
    {
        nodeId = inputNodeId;
    }

    // Node Relationships
    void SetPreNodeId(uint64_t inputPreNodeId)
    {
        preNodeId = inputPreNodeId;
    }
    void SetNextNodeId(uint64_t inputNextNodeId)
    {
        nextNodeId = inputNextNodeId;
    }
    uint64_t GetPreNodeId() const
    {
        return preNodeId;
    }

    uint64_t GetNextNodeId() const
    {
        return nextNodeId;
    }

    // SuperKernelKernelNode specific accessors
    virtual uint32_t GetNumBlocks() const { return 0; }
    virtual SkKernelType GetKernelType() const { return SkKernelType::DEFAULT; }
    virtual uint32_t GetVecNum() const { return 0; }
    virtual uint32_t GetCubeNum() const { return 0; }

    // SuperKernelEventNode/SuperKernelMemoryNode specific accessors
    virtual uint64_t GetEventId() const
    {
        return INVALID_TASK_ID;
    }

    // SuperKernelEventNotifyNode/SuperKernelMemoryNotifyNode specific accessors
    // Get all wait node IDs that wait on this notify node's event (one-to-many relationship)
    virtual std::vector<uint64_t> GetCorrespondingWaitNodeIds() const
    {
        return std::vector<uint64_t>();
    }

    // SuperKernelEventWaitNode/SuperKernelMemoryWaitNode specific accessors
    // Get the notify node ID that this wait node waits on (many-to-one relationship)
    virtual uint64_t GetCorrespondingNotifyNodeId() const
    {
        return INVALID_TASK_ID;
    }

    // Setter for corresponding wait node IDs (used by SuperKernelGraph to build associations)
    virtual void SetCorrespondingWaitNodeIds(const std::vector<uint64_t>& waitIds) {}

    // Setter for notify node ID (used by SuperKernelGraph to build associations for wait nodes)
    virtual void SetCorrespondingNotifyNodeId(uint64_t notifyId) {}

    virtual const NodeInfos& GetNodeInfos() const
    {
        return nodeInfos;
    }

    virtual bool Update(const UpdateContext& ctx = {})
    {
        return InValidateNode();
    }

    virtual bool InValidateNode() = 0;

    // Task Type
    SkNodeType GetNodeType() const
    {
        return nodeType;
    }

    void SetNodeType(SkNodeType inputNodeType)
    {
        nodeType = inputNodeType;
    }
    // Visitation State
    bool IsVisited() const
    {
        return isVisited;
    }
    void SetVisited(bool inputIsVisited)
    {
        isVisited = inputIsVisited;
    }

    // scope
    virtual const std::string GetScopeName() const { return ""; }
    virtual bool IsScopeBegin() const { return false; }
    virtual bool IsScopeEnd() const { return false; }
    virtual bool IsScopePlaceholder() const { return false; }

    const std::bitset<MAX_SCOPE_NUM>& GetScopeBitFlags() const
    {
        return scopeBitFlags;
    }
    void SetScopeBitFlags(const std::bitset<MAX_SCOPE_NUM>& flags)
    {
        scopeBitFlags = flags;
    }
    void SetIsScopeNode(bool isScope) { isScopeNode = isScope; }
    bool IsScopeNode() const { return isScopeNode; }
    void ClearScopeBitFlags() { scopeBitFlags.reset(); }
    void MarkEventNodeToScope(SuperKernelBaseNode* node);
    
    // Notify node expand number setters
    void SetNotifyExpandVecNum(uint32_t vecNum) { notifyExpandVecNum = vecNum; }
    void SetNotifyExpandCubeNum(uint32_t cubeNum) { notifyExpandCubeNum = cubeNum; }
public:
    NodeInfos nodeInfos;
    std::unique_ptr<aclmdlRITask> originTask;

protected:
    aclmdlRITaskParams taskParams;
    uint32_t notifyExpandVecNum;
    uint32_t notifyExpandCubeNum;
    uint32_t streamIdxInGraph;
    uint64_t nodeIdxInStream;
    uint64_t nodeId;
    uint64_t preNodeId;
    uint64_t nextNodeId;
    SkNodeType nodeType;
    aclmdlRITaskType rtNodeType;
    bool isVisited;
    bool isFusible;
    bool isScopeNode;
    std::bitset<MAX_SCOPE_NUM> scopeBitFlags;
};

// Derived Node Classes

class SuperKernelKernelNode : public SuperKernelBaseNode {
public:
    using SuperKernelBaseNode::SuperKernelBaseNode;
    bool InitNode() override;
    uint32_t GetNumBlocks() const override { return nodeInfos.kernelInfos.numBlocks; }
    SkKernelType GetKernelType() const override { return nodeInfos.kernelInfos.kernelType; }
    uint32_t GetVecNum() const override { return nodeInfos.kernelInfos.vecNum; }
    uint32_t GetCubeNum() const override { return nodeInfos.kernelInfos.cubeNum; }

    const std::string GetNodeName() const override
    {
        return nodeInfos.kernelInfos.funcName;
    }
    bool InValidateNode() override;
    bool Update(const UpdateContext& ctx) override;
    const std::string GetScopeName() const override
    {
        return scopeName;
    }
    bool IsScopeBegin() const override { return isScopeBegin; }
    bool IsScopeEnd() const override { return isScopeEnd; }
    bool IsScopePlaceholder() const override { return isPlaceholder; }
private:
    bool isScopeBegin = false;
    bool isScopeEnd = false;
    bool isPlaceholder = false;
    std::string scopeName;
};

class SuperKernelMemoryNode : public SuperKernelBaseNode {
public:
    using SuperKernelBaseNode::SuperKernelBaseNode;
    uint64_t GetEventId() const override
    {
        return nodeInfos.syncInfos.eventId;
    }
    std::vector<uint64_t> GetCorrespondingWaitNodeIds() const override
    {
        return nodeInfos.syncInfos.correspondingWaitNodeIds;
    }
    void SetCorrespondingWaitNodeIds(const std::vector<uint64_t>& waitIds) override
    {
        nodeInfos.syncInfos.correspondingWaitNodeIds.assign(waitIds.begin(), waitIds.end());
    }
    uint64_t GetCorrespondingNotifyNodeId() const override
    {
        return nodeInfos.syncInfos.correspondingNotifyNodeId;
    }
    void SetCorrespondingNotifyNodeId(uint64_t notifyId) override
    {
        nodeInfos.syncInfos.correspondingNotifyNodeId = notifyId;
    }

    const std::string GetNodeName() const override
    {
        std::string memoryNodeName;
        if (rtNodeType == ACL_MODEL_RI_TASK_VALUE_WRITE) {
            memoryNodeName = "NotifyNode_";
        } else if (rtNodeType == ACL_MODEL_RI_TASK_VALUE_WAIT) {
            memoryNodeName = "WaitNode_";
        } else {
            memoryNodeName = "UnknownMemoryNode_";
        }
        memoryNodeName += std::to_string(GetEventId());
        return memoryNodeName;
    }
    bool InitNode() override;
    bool Update(const UpdateContext& ctx) override;
    bool InValidateNode() override;
    uint32_t GetVecNum() const override { return notifyExpandVecNum; }
    uint32_t GetCubeNum() const override { return notifyExpandCubeNum; }
};

class SuperKernelDefaultNode : public SuperKernelBaseNode {
public:
    using SuperKernelBaseNode::SuperKernelBaseNode;
    bool InitNode() override;
    bool InValidateNode() override;

    const std::string GetNodeName() const override
    {
        static const std::string defaultNodeName = "DefaultNode";
        return defaultNodeName;
    }
};

#endif // __SK_NODE_H__
