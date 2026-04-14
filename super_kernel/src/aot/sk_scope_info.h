/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_scope_info.h
 * \brief Super Kernel Scope Info - data structures for scope splitting
 */

#ifndef __SK_SCOPE_INFO_H__
#define __SK_SCOPE_INFO_H__

#include <bitset>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "sk_types.h"
#include "sk_node.h"

/*!
 * \struct ScopeStreamInfo
 * \brief Information about a single stream within a scope
 */
struct ScopeStreamInfo {
    uint32_t streamIdx = 0;              ///< Stream index in the graph
    uint64_t headNodeIdx = INVALID_TASK_ID;  ///< First node of this stream in scope
    uint64_t tailNodeIdx = INVALID_TASK_ID;  ///< Last node of this stream in scope
    uint64_t nodeSize = 0;               ///< Number of nodes from this stream in scope
};

/*!
 * \enum ScopeFailReason
 * \brief Failure reasons for scope processing
 */
enum class ScopeFailReason : uint8_t {
    NONE = 0,
    VALIDATION_FAILED,
    NO_TASK,
    NO_KERNEL,
    EVENT_MEMORY_APPLY_FAILED,
    STREAM_BOUNDARY_INVALID,
    STREAM_SELECT_FAILED,
    SUB_STREAM_SYNC_FAILED,
};

/*!
 * \struct ScopeExtInfo
 * \brief Extended information for scope post-processing and scheduling
 */
struct ScopeExtInfo {
    std::vector<std::vector<aclmdlRITaskParams>> customParamsList; ///< Custom parameters for each stream
    std::vector<SuperKernelBaseNode*> filteredNodes;               ///< Post-processed nodes used for scheduling
    std::vector<std::unique_ptr<SuperKernelBaseNode>> eventNodes;  ///< Synthesized event nodes for stream sync
    uint64_t skMainNodeId = INVALID_TASK_ID;                       ///< Main launch node ID for this scope
    uint32_t scopeIdx = 0;
    std::string scopeName;
    ScopeFailReason failReason = ScopeFailReason::NONE;

    ScopeExtInfo() = default;
    ScopeExtInfo(const ScopeExtInfo&) = delete;
    ScopeExtInfo& operator=(const ScopeExtInfo&) = delete;
    ScopeExtInfo(ScopeExtInfo&&) = default;
    ScopeExtInfo& operator=(ScopeExtInfo&&) = default;
};

/*!
 * \brief Convert ScopeFailReason to string
 */
inline const char* to_string(ScopeFailReason reason)
{
    switch (reason) {
        case ScopeFailReason::NONE:
            return "NONE";
        case ScopeFailReason::VALIDATION_FAILED:
            return "VALIDATION_FAILED";
        case ScopeFailReason::NO_TASK:
            return "NO_TASK";
        case ScopeFailReason::NO_KERNEL:
            return "NO_KERNEL";
        case ScopeFailReason::EVENT_MEMORY_APPLY_FAILED:
            return "EVENT_MEMORY_APPLY_FAILED";
        case ScopeFailReason::STREAM_BOUNDARY_INVALID:
            return "STREAM_BOUNDARY_INVALID";
        case ScopeFailReason::STREAM_SELECT_FAILED:
            return "STREAM_SELECT_FAILED";
        case ScopeFailReason::SUB_STREAM_SYNC_FAILED:
            return "SUB_STREAM_SYNC_FAILED";
        default:
            return "UNKNOWN";
    }
}

/*!
 * \class SuperKernelScopeInfo
 * \brief Information about a single scope in Super Kernel fusion
 *
 * A scope represents a collection of nodes from potentially multiple streams
 * that can be fused into a single Super Kernel.
 */
class SuperKernelScopeInfo {
public:
    SuperKernelScopeInfo() = default;
    ~SuperKernelScopeInfo() = default;
    SuperKernelScopeInfo(const SuperKernelScopeInfo&) = delete;
    SuperKernelScopeInfo& operator=(const SuperKernelScopeInfo&) = delete;
    SuperKernelScopeInfo(SuperKernelScopeInfo&&) = default;
    SuperKernelScopeInfo& operator=(SuperKernelScopeInfo&&) = default;

    // ============ ScopeStreamInfos ============
    const std::vector<ScopeStreamInfo>& GetScopeStreamInfos() const { return scopeStreamInfos; }
    void SetScopeStreamInfos(std::vector<ScopeStreamInfo> infos) { scopeStreamInfos = std::move(infos); }
    void AddScopeStreamInfo(const ScopeStreamInfo& info) { scopeStreamInfos.push_back(info); }

    // ============ Nodes ============
    const std::vector<SuperKernelBaseNode*>& GetNodes() const { return nodes; }
    void SetNodes(std::vector<SuperKernelBaseNode*> nodeList) { nodes = std::move(nodeList); }
    void AddNode(SuperKernelBaseNode* node) { nodes.push_back(node); }

    // ============ ScopeBitFlags ============
    const std::bitset<MAX_SCOPE_NUM>& GetScopeBitFlags() const { return scopeBitFlags; }
    void SetScopeBitFlags(const std::bitset<MAX_SCOPE_NUM>& flags) { scopeBitFlags = flags; }

    // ============ ExtInfo ============
    const ScopeExtInfo& GetExtInfo() const { return extInfo_; }
    void SetExtInfo(ScopeExtInfo&& extInfo) { extInfo_ = std::move(extInfo); }
    ScopeExtInfo& MutableExtInfo() { return extInfo_; }

private:
    std::vector<ScopeStreamInfo> scopeStreamInfos;  ///< Per-stream information (public for compatibility)
    std::vector<SuperKernelBaseNode*> nodes;        ///< All nodes in this scope (public for compatibility)
    std::bitset<MAX_SCOPE_NUM> scopeBitFlags;       ///< Scope bit flags (public for compatibility)
    ScopeExtInfo extInfo_;  ///< Extended info for post-processing and scheduling
};

#endif // __SK_SCOPE_INFO_H__
