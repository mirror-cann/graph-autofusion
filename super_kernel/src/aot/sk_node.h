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
#include <set>

#include "sk_log.h"
#include "sk_types.h"
#include "acl/acl.h"

// Forward declaration
class SuperKernelGraph;
struct SkLaunchInfo;

// Forward declaration for ScopeFailReason (defined in sk_scope_info.h)
enum class ScopeFailReason : uint8_t;

// Forward declaration for DeadlockFailReason (defined in sk_lock_detector.h)
enum class DeadlockFailReason : uint8_t;

// unfused reason
enum class FusionFailReason {
    CAN_FUSE,               // 0：可以融合（默认）
    BINDMAP_EMPTY,      // 1：bindmap为空
    TASK_GROUP_EMPTY,   // 2：task group为空
    RESET_TYPE_NODE,    // 3: reset 类型会放到最后
    NOT_IN_SCOPE,       // 4: 不在融合范围
    IN_UNFUSIBLE_SCOPE, // 5: 标定不可融
    ISOLATED_EVENT,     // 6：存在孤立事件
    EXIST_DEADLOCK,     // 7：存在死锁
    SCOPE_FUSE_PART,    // 8：scope融合失败（具体原因见 ScopeFailReason）
    EXTERNAL_DEPEND,    // 9: event存在外部依赖
    UNSUPPORT_EVENT_TYPE, // 10: event类型不支持
};

// Fusion fail reason with optional scope/deadlock detail
// Note: scopeDetailValue stores ScopeFailReason as uint8_t to avoid circular dependency
// Note: deadlockDetailValue stores DeadlockFailReason as uint8_t to avoid circular dependency
struct FusionFailReasonInfo {
    FusionFailReason primary = FusionFailReason::CAN_FUSE;
    uint8_t scopeDetailValue = 0;       // ScopeFailReason::NONE
    uint8_t deadlockDetailValue = 0;    // DeadlockFailReason::NOT_FIND_DEADLOCK
    
    FusionFailReasonInfo() = default;
    explicit FusionFailReasonInfo(FusionFailReason p) : primary(p) {}
    FusionFailReasonInfo(FusionFailReason p, ScopeFailReason s);
    FusionFailReasonInfo(FusionFailReason p, DeadlockFailReason d);
    
    ScopeFailReason GetScopeDetail() const;
    void SetScopeDetail(ScopeFailReason s);
    
    DeadlockFailReason GetDeadlockDetail() const;
    void SetDeadlockDetail(DeadlockFailReason d);
    
    bool operator==(FusionFailReason p) const { return primary == p; }
    bool operator!=(FusionFailReason p) const { return primary != p; }
};

inline const char* FusionFailReasonToStr(FusionFailReason reason) {
    switch (reason) {
        case FusionFailReason::CAN_FUSE:              return "node can fuse";
        case FusionFailReason::BINDMAP_EMPTY:     return "bindMap is empty, please check config SK_BIND";
        case FusionFailReason::TASK_GROUP_EMPTY:   return "Kernel task group is not null";
        case FusionFailReason::RESET_TYPE_NODE:    return "reset type node in end";
        case FusionFailReason::NOT_IN_SCOPE:      return "node not in fusion scope";
        case FusionFailReason::IN_UNFUSIBLE_SCOPE: return "node in unfusible scope";
        case FusionFailReason::ISOLATED_EVENT:    return "Isolated event node may cause resource shortage in multi-stream scenarios";
        case FusionFailReason::EXIST_DEADLOCK:    return "exist deadlock";
        case FusionFailReason::SCOPE_FUSE_PART:   return "scope fuse failed";
        case FusionFailReason::EXTERNAL_DEPEND:   return "event node has external dependency";
        case FusionFailReason::UNSUPPORT_EVENT_TYPE: return "unsupport event type";
        default:                                  return "UNKNOWN_REASON";
    }
}

// Declaration - implementation in sk_node.cpp after including sk_scope_info.h
std::string FusionFailReasonToStr(const FusionFailReasonInfo& info);

// Update context for node update operations
struct UpdateContext {
    SkLaunchInfo* launchInfo = nullptr;
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
    uint64_t funcOffset[2] = {0, 0};  // Offset within the bin file for AIC/AIV
};

constexpr size_t kMaxSplitBinCount = 4;

struct KernelInfos {
    SkKernelType kernelType = SkKernelType::DEFAULT;
    uint32_t taskRatio[2] = {0, 0};
    uint32_t resolvedNum = 0;
    uint32_t numBlocks = 0;
    uint32_t vecNum = 0;      ///< Number of vector cores required
    uint32_t cubeNum = 0;     ///< Number of cube cores required
    const void *devArgs = nullptr;
    void* opInfoPtr = nullptr;
    size_t opInfoSize = 0;
    std::string funcName = "Unknown";
    aclrtBinHandle binHdl = nullptr;
    aclrtFuncHandle funcHdl = nullptr;
    aclrtLaunchKernelCfg* launchKernelCfg = nullptr;
    bool isScheModeOn = false;
    ResolvedFunctionInfo resolvedFuncs[4];

    std::string Format() const;
};

struct SyncInfos {
    uint64_t eventId = INVALID_TASK_ID;
    void* addrValue = nullptr;
    // For notify nodes: empty (not used)
    // For wait nodes: this is the ID of the notify node this wait node waits on
    uint64_t correspondingNotifyNodeId = INVALID_TASK_ID;
    // For notify nodes: list of all wait node IDs that wait on this notify
    // For wait nodes: empty (not used)
    std::vector<uint64_t> correspondingWaitNodeIds;
    // For event nodes, the corresponding reset node ID
    std::vector<uint64_t> correspondingResetNodeIds;
    std::vector<uint64_t> correspondingMemoryWriteNodeIds;
    uint64_t memoryValue = std::numeric_limits<uint64_t>::max();
    uint32_t memoryWaitFlag = std::numeric_limits<uint32_t>::max();
    uint64_t eventFlag = std::numeric_limits<uint64_t>::max();
};

struct NodeInfos {
    KernelInfos kernelInfos;
    SyncInfos syncInfos;
};

// Base Node Class
class SuperKernelBaseNode {
public:
    SuperKernelBaseNode(std::unique_ptr<aclmdlRITask> inputOriginTask, aclmdlRITaskType inputRtNodeType,
                        uint64_t inputNodeIdxInStream, uint64_t inputStreamIdxInGraph, int32_t inputStreamId, uint64_t inputPreNodeId)
        : originTask(std::move(inputOriginTask)),
          taskParams({}),
          rtNodeType(inputRtNodeType),
          notifyExpandVecNum(0),
          notifyExpandCubeNum(0),
          streamIdxInGraph(inputStreamIdxInGraph),
          streamId(inputStreamId),
          nodeIdxInStream(inputNodeIdxInStream),
          nodeId(INVALID_TASK_ID),
          preNodeId(inputPreNodeId),
          nextNodeId(INVALID_TASK_ID),
          nodeType(SkNodeType::NODE_DEFAULT),
          isVisited(false),
          isFusible(false),
          isScopeNode(false),
          isUpdate(false) { }
    virtual ~SuperKernelBaseNode() = default;
    virtual bool InitNode();

    /**
     * @brief Format complete node information for logging
     * @return Formatted string with nodeId, streamIdxInGraph, nodeIdxInStream, and node-specific info
     *
     * Format: [nodeId:lu, streamIdxInGraph:u, nodeIdxInStream:lu] - {node-specific-info}
     * Examples:
     *   Kernel: [nodeId:123, streamIdxInGraph:0, nodeIdxInStream:5] - Kernel:func_name
     *   Notify: [nodeId:124, streamIdxInGraph:0, nodeIdxInStream:6] - Event:Notify(eventId:0x7ff8a0)
     *   Wait:    [nodeId:125, streamIdxInGraph:0, nodeIdxInStream:7] - Event:Wait(eventId:0x7ff8a0)
     *   Default: [nodeId:126, streamIdxInGraph:0, nodeIdxInStream:8] - Default
     */
    virtual std::string Format() const = 0;

    // Accessors
    uint32_t GetStreamIdxInGraph() const
    {
        return streamIdxInGraph;
    }
    int32_t GetStreamId() const
    {
        return streamId;
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
    virtual bool IsScheModeOn() const { return false; }
    virtual bool GetScheMode() const { return false; }
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

    virtual std::vector<uint64_t> GetCorrespondingMemoryWriteNodeIds() const
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

    virtual void SetCorrespondingMemoryWriteNodeId(const std::vector<uint64_t>& memortWriteIds) {}

    virtual const NodeInfos& GetNodeInfos() const
    {
        return nodeInfos;
    }

    virtual bool Update(const UpdateContext& ctx = {});

    virtual aclError InValidateNode();

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
    void SetScopeStreamIds(const std::unordered_set<uint32_t>& streamIds) { scopeStreamIds = streamIds; }
    const std::unordered_set<uint32_t>& GetScopeStreamIds() const { return scopeStreamIds; }

    // for update
    bool IsUpdated() const { return isUpdate; }
    void SetUpdate(bool update) { isUpdate = update; }

    // Fusion fail reason setters
    void SetFusionFailReason(FusionFailReason reason, ScopeFailReason scopeReason = static_cast<ScopeFailReason>(0)) {
        fusionFailReason_.primary = reason;
        fusionFailReason_.SetScopeDetail(scopeReason);
    }
    void SetFusionFailReason(FusionFailReason reason, DeadlockFailReason deadlockReason) {
        fusionFailReason_.primary = reason;
        fusionFailReason_.SetDeadlockDetail(deadlockReason);
    }
    void SetFusionFailReason(const FusionFailReasonInfo& info) { fusionFailReason_ = info; }
    
    // Fusion fail reason getters
    FusionFailReason GetFusionFailReason() const { return fusionFailReason_.primary; }
    const FusionFailReasonInfo& GetFusionFailReasonInfo() const { return fusionFailReason_; }

public:
    NodeInfos nodeInfos;
    std::unique_ptr<aclmdlRITask> originTask;
    std::unordered_set<uint64_t> sendToNodeId;
    std::unordered_set<uint64_t> receiveNodeId;
    FusionFailReasonInfo fusionFailReason_;

protected:
    aclmdlRITaskParams taskParams;
    void LogNodeUpdateResult(const aclmdlRITaskParams* taskParams) const;
    const char* GetUpdateTargetTypeName(aclmdlRITaskType type) const;
    uint32_t notifyExpandVecNum;
    uint32_t notifyExpandCubeNum;
    uint32_t streamIdxInGraph;
    int32_t streamId;
    uint64_t nodeIdxInStream;
    uint64_t nodeId;
    uint64_t preNodeId;
    uint64_t nextNodeId;
    SkNodeType nodeType;
    aclmdlRITaskType rtNodeType;
    bool isVisited;
    bool isFusible;
    bool isScopeNode;
    bool isUpdate;
    std::unordered_set<uint32_t> scopeStreamIds;
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
    bool GetScheMode() const override;
    std::string Format() const override;
    bool Update(const UpdateContext& ctx) override;
    const std::string GetScopeName() const override
    {
        return scopeName;
    }
    bool IsScopeBegin() const override { return isScopeBegin; }
    bool IsScopeEnd() const override { return isScopeEnd; }
    bool IsScopePlaceholder() const override { return isPlaceholder; }
    bool IsScheModeOn() const override { return nodeInfos.kernelInfos.isScheModeOn; }
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

    void SetCorrespondingMemoryWriteNodeId(const std::vector<uint64_t>& memortWriteIds) {
        nodeInfos.syncInfos.correspondingMemoryWriteNodeIds.assign(memortWriteIds.begin(), memortWriteIds.end());
    }

    std::vector<uint64_t> GetCorrespondingMemoryWriteNodeIds() const override
    {
        return nodeInfos.syncInfos.correspondingMemoryWriteNodeIds;
    }

    std::string Format() const override;
    bool InitNode() override;
    bool Update(const UpdateContext& ctx) override;
    uint32_t GetVecNum() const override { return notifyExpandVecNum; }
    uint32_t GetCubeNum() const override { return notifyExpandCubeNum; }
};

class SuperKernelDefaultNode : public SuperKernelBaseNode {
public:
    using SuperKernelBaseNode::SuperKernelBaseNode;
    bool InitNode() override;
    aclError InValidateNode() override;

    std::string Format() const override;
};

// Kernel binary dump function
class SuperKernelGraph;
bool DumpKernelBinaries(const SuperKernelGraph& graph, const std::string& binPath);

#endif // __SK_NODE_H__
