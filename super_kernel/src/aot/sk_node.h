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
#include <unordered_map>
#include <array>

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

// SkBindMap type for kernel function binding
using SkBindMap = std::unordered_map<uint64_t, std::array<uint64_t, 4>>;

// Unfused reason
enum class FusionFailReason {
    CAN_FUSE,               // 0: Can fuse (default)
    BINDMAP_IS_EMPTY,      // 1: Operator does not support SuperKernel fusion
    TASK_GROUP_NOT_EMPTY,   // 2: Operator dynamically refreshes task info at runtime, SK does not support fusing dynamically changing tasks
    NOT_IN_SCOPE,       // 3: Operator is not within user-marked fusion range
    IN_UNFUSIBLE_SCOPE, // 4: User actively marked this operator as unfusible
    EXCEED_DEVICE_MAX,  // 5: Operator requires more cores than device maximum
    RESET_TYPE_NODE,    // 3: reset type node placed at end
    ISOLATED_EVENT,     // 6: Isolated event exists
    EXIST_DEADLOCK,     // 7: Deadlock exists
    SCOPE_FUSE_PART,    // 8: Scope fusion failed (see ScopeFailReason for details)
    EXTERNAL_DEPEND,    // 9: Event has external dependency
    UNSUPPORT_EVENT_TYPE, // 10: Unsupported event type
    NOTIFY_NO_WAIT_NODE,  // 11: notify node has no wait node in modelRI, mark as unfusible
    MEMORY_WAIT_NODE_ONLY, // 12: No memory write exists, meaning the memory write is outside modelRI,
    MEMORY_WRITE_NODE_ONLY,  // 13: only exists memory write nodes, mask it as unfusible
    DEFAULT_NODE, // default node uses aicpu resources, mask it as unfusible
};

// Bindmap related fail reason detail
enum class BindmapFailReason : uint8_t {
    NONE,                    // 0: No error
    BINDMAP_INIT_EMPTY,      // 1: Bindmap empty after initialization
    BINHDL_NULL,             // 2: binHdl is null
    FUNCHDL_NULL,            // 3: funcHdl is null
    FUNC_NOT_FOUND,          // 4: Failed to initialize kernel function in sk
};

// Fusion fail reason with optional scope/deadlock detail
// Note: scopeDetailValue stores ScopeFailReason as uint8_t to avoid circular dependency
// Note: deadlockDetailValue stores DeadlockFailReason as uint8_t to avoid circular dependency
struct FusionFailReasonInfo {
    FusionFailReason primary = FusionFailReason::CAN_FUSE;
    uint8_t scopeDetailValue = 0;       // ScopeFailReason::NONE
    uint8_t deadlockDetailValue = 0;    // DeadlockFailReason::NOT_FIND_DEADLOCK
    uint8_t bindmapDetailValue = 0;     // BindmapFailReason::NONE
    
    FusionFailReasonInfo() = default;
    explicit FusionFailReasonInfo(FusionFailReason reason) : primary(reason) {}
    FusionFailReasonInfo(FusionFailReason reason, ScopeFailReason scopeReason);
    FusionFailReasonInfo(FusionFailReason reason, DeadlockFailReason deadlockReason);
    FusionFailReasonInfo(FusionFailReason reason, BindmapFailReason bindmapReason);
    
    ScopeFailReason GetScopeDetail() const;
    void SetScopeDetail(ScopeFailReason scopeReason);
    
    DeadlockFailReason GetDeadlockDetail() const;
    void SetDeadlockDetail(DeadlockFailReason deadlockReason);

    BindmapFailReason GetBindmapDetail() const;
    void SetBindmapDetail(BindmapFailReason bindmapReason);
    
    bool operator==(FusionFailReason reason) const { return primary == reason; }
    bool operator!=(FusionFailReason reason) const { return primary != reason; }
};

// Declaration for BindmapFailReasonToStr
const char* BindmapFailReasonToStr(BindmapFailReason reason);

// Declaration for AlignUpAndClamp
size_t AlignUpAndClamp(size_t value, size_t coreIdx);

inline const char* FusionFailReasonToStr(FusionFailReason reason) {
    switch (reason) {
        case FusionFailReason::CAN_FUSE:              
            return "node can fuse";
        case FusionFailReason::BINDMAP_IS_EMPTY:     
            return "The operator does not support the operation of fusing SuperKernel";
        case FusionFailReason::TASK_GROUP_NOT_EMPTY:   
            return "The operator will refresh task information at runtime, but SK does not support fusing dynamically changing tasks";
        case FusionFailReason::NOT_IN_SCOPE:      
            return "The user actively marked that this operator is not fused";
        case FusionFailReason::IN_UNFUSIBLE_SCOPE: 
            return "This operator is not within the fusion range marked by the user";
        case FusionFailReason::EXCEED_DEVICE_MAX:  
            return "The number of kernels required by the operator exceeds the maximum number of kernels that the device can provide";
        case FusionFailReason::RESET_TYPE_NODE:    
            return "reset type node in end";
        case FusionFailReason::ISOLATED_EVENT:    
            return "There is no kernel node on the stream where the current node is located, and this stream is within the scope";
        case FusionFailReason::EXIST_DEADLOCK:    
            return "exist deadlock";
        case FusionFailReason::SCOPE_FUSE_PART:   
            return "scope fuse failed";
        case FusionFailReason::EXTERNAL_DEPEND:   
            return "event node has external dependency";
        case FusionFailReason::UNSUPPORT_EVENT_TYPE: 
            return "unsupport event type";
        case FusionFailReason::NOTIFY_NO_WAIT_NODE: 
            return "notify node has no wait node in modelRI, mark as unfusible";
        case FusionFailReason::MEMORY_WAIT_NODE_ONLY: 
            return "No memory write exists, meaning the memory write is outside modelRI. Therefore change all waits to event semantics, but they cannot be fused.";
        case FusionFailReason::MEMORY_WRITE_NODE_ONLY: 
            return "only exists memory write nodes, mask it as unfusible";
        case FusionFailReason::DEFAULT_NODE: 
            return "default node uses aicpu resources, mask it as unfusible";
        default:                                  
            return "UNKNOWN_REASON";
    }
}

// Declaration for dump json
const char* GetKernelTypeString(uint32_t kernelType, const uint32_t taskRatio[2]);
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
    std::string symbolBind[2] = {"", ""};  // Symbol binding type: GLOBAL/WEAK/LOCAL
};

constexpr size_t K_MAX_SPLIT_BIN_COUNT = 4;

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
    BindmapFailReason bindmapFailReason = BindmapFailReason::NONE;

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
    void SetFusionFailReason(FusionFailReason reason, BindmapFailReason bindmapReason) {
        fusionFailReason_.primary = reason;
        fusionFailReason_.SetBindmapDetail(bindmapReason);
    }    
    void SetFusionFailReason(const FusionFailReasonInfo& info) { fusionFailReason_ = info; }
    
    // Fusion fail reason getters
    FusionFailReason GetFusionFailReason() const { return fusionFailReason_.primary; }
    const FusionFailReasonInfo& GetFusionFailReasonInfo() const { return fusionFailReason_; }

    // Task params accessor (for dump after update)
    const aclmdlRITaskParams& GetTaskParams() const { return taskParams; }

    // Invalidation status accessor
    bool IsInvalidated() const { return isInvalidated; }
    void SetInvalidated(bool invalidated) { isInvalidated = invalidated; }

    SkBindMap InitSuperKernelBindMap(aclrtBinHandle binHdl);

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
    bool isInvalidated = false;
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

// Get bind map for a binary handle
const SkBindMap& GetSkBindMap(aclrtBinHandle binHdl);

#endif // __SK_NODE_H__
