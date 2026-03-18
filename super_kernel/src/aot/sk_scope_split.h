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
 * \file sk_scope_split.h
 * \brief Super Kernel Scope Splitter - splits multi-stream graph into scopes
 * 
 * This module implements the scope splitting algorithm for multi-stream graphs.
 * A "scope" represents a group of nodes that can be fused into a single Super Kernel.
 * 
 * Architecture:
 * The scope splitting is implemented as a multi-pass pipeline, similar to compiler passes:
 * - Pass 1: Initial scope splitting based on fusibility and scopeBitFlags
 * - Pass 2: Deadlock detection and scope refinement
 * - Future passes can be added for additional optimizations
 * 
 * Key Concepts:
 * - Scope: A collection of nodes from potentially multiple streams that can be fused together
 * - ScopeBitFlags: Bit flags that identify which scope a node belongs to
 * - Deadlock Detection: Prevents circular dependencies between streams
 */

#ifndef __SK_SCOPE_SPLIT_H__
#define __SK_SCOPE_SPLIT_H__

#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

#include "sk_graph.h"
#include "sk_log.h"
#include "sk_lock_detector.h"

struct ScopeStreamInfo {
    uint32_t streamIdx = 0;              ///< Stream index in the graph
    uint64_t headNodeIdx = INVALID_TASK_ID;  ///< First node of this stream in scope
    uint64_t tailNodeIdx = INVALID_TASK_ID;  ///< Last node of this stream in scope
    uint64_t nodeSize = 0;               ///< Number of nodes from this stream in scope
};

struct SuperKernelScopeInfo {
    std::vector<ScopeStreamInfo> scopeStreamInfos;  ///< Per-stream information
    std::vector<SuperKernelBaseNode*> nodes;        ///< All nodes in this scope (ordered by node ID)
    std::bitset<MAX_SCOPE_NUM> scopeBitFlags;       ///< Scope bit flags (all nodes must have matching flags)
};

/*!
 * \struct StreamState
 * \brief Processing state for a single stream during multi-stream splitting
 */
struct StreamState {
    uint64_t currentNodeIdx;              ///< Current node being processed
    bool isSuspended;                     ///< True if waiting for Notify event
    uint64_t waitingForNotify;            ///< Event ID being waited for (if suspended)
    bool isTerminated;                    ///< True if stream cannot continue in current scope

    StreamState()
        : currentNodeIdx(INVALID_TASK_ID),
          isSuspended(false),
          waitingForNotify(INVALID_TASK_ID),
          isTerminated(false) {}
};

// ============ Pass Base Class ============

/*!
 * \class ScopeSplitPass
 * \brief Base class for scope splitting passes
 */
class ScopeSplitPass {
public:
    explicit ScopeSplitPass(SuperKernelGraph& inputGraph) : graph_(inputGraph) {}
    virtual ~ScopeSplitPass() = default;
    
    /*!
     * \brief Execute the pass on the given scopes
     * \param scopes Input/output scope list
     * \return true on success
     */
    virtual bool Run(std::vector<SuperKernelScopeInfo>& scopes) = 0;
    
    virtual std::string GetName() const = 0;

    // ============ Debug/Logging Utilities (public static for reuse) ============
    
    /*!
     * \brief Print scope splitting results for debugging
     * \param scopes Scope list to print
     * \param graph Graph for scope name lookup
     */
    static void PrintScopeResults(const std::vector<SuperKernelScopeInfo>& scopes, 
                                   const SuperKernelGraph& graph);
    
    /*!
     * \brief Get scope names string from scopeBitFlags
     * \param scopeBitFlags Bit flags to convert
     * \param graph Graph for scope name lookup
     * \return Comma-separated scope names string
     */
    static std::string GetScopeNamesFromBitFlags(const std::bitset<MAX_SCOPE_NUM>& scopeBitFlags,
                                                  const SuperKernelGraph& graph);
    
    /*!
     * \brief Print nodes in a scope
     * \param scopeIdx Scope index
     * \param scope Scope info
     */
    static void PrintScopeNodes(size_t scopeIdx, const SuperKernelScopeInfo& scope);
    
    /*!
     * \brief Print stream infos in a scope
     * \param scopeIdx Scope index
     * \param scope Scope info
     */
    static void PrintScopeStreamInfos(size_t scopeIdx, const SuperKernelScopeInfo& scope);

protected:
    SuperKernelGraph& graph_;
};

// ============ Pass 1: Initial Scope Split ============

/*!
 * \class InitialScopeSplitPass
 * \brief Pass 1: Split graph into initial scopes based on fusibility and scopeBitFlags
 * 
 * This pass performs coarse-grained scope splitting:
 * - Groups nodes by scopeBitFlags
 * - Respects fusibility constraints
 * - Handles Wait/Notify synchronization
 * - Does NOT perform deadlock detection
 */
class InitialScopeSplitPass : public ScopeSplitPass {
public:
    explicit InitialScopeSplitPass(SuperKernelGraph& inputGraph);
    ~InitialScopeSplitPass() = default;
    
    bool Run(std::vector<SuperKernelScopeInfo>& scopes) override;
    std::string GetName() const override { return "InitialScopeSplitPass"; }

private:
    // ============ Stream State Management ============
    void InitStreamStates();
    void ResetStreamStates();
    void SkipUnfusibleNodes();
    bool SkipUnfusibleNodesForStream(uint32_t streamIdx);
    bool ProcessUnfusibleWaitNode(uint32_t streamIdx, SuperKernelBaseNode* waitNode);
    bool AllStreamsFinished() const;
    bool DetermineCurrentScopeBitFlags();
    
    // ============ Node Processing ============
    void InitNodeHeap();
    void TryAddNodeToHeap(uint32_t streamIdx);
    void HandleWaitNode(SuperKernelBaseNode* waitNode, uint32_t streamIdx);
    void ProcessNotifyNode(SuperKernelBaseNode* notifyNode);
    void ProcessResetNode(SuperKernelBaseNode* resetNode);
    
    // ============ Scope Building ============
    void AddStreamInfoToScope(SuperKernelScopeInfo& scopeInfo, SuperKernelBaseNode* node);
    bool BuildCurrentScope(SuperKernelScopeInfo& scopeInfo);
    
    // ============ Member Variables ============
    std::unordered_map<uint32_t, StreamState> streamStates_;
    std::set<uint64_t> visitedNotifies_;
    std::set<uint64_t> processedNodes_;
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> nodeHeap_;
    std::bitset<MAX_SCOPE_NUM> currentScopeBitFlags_;
};

// ============ Pass 2: Deadlock Detection and Refinement ============

/*!
 * \class DeadlockRefinePass
 * \brief Pass 2: Detect deadlocks and refine scopes
 * 
 * This pass performs fine-grained scope refinement:
 * - Detects deadlocks within each scope
 * - Splits scopes at Wait nodes that would cause deadlocks
 * - The Wait node that triggers deadlock is NOT included in either split scope
 */
class DeadlockRefinePass : public ScopeSplitPass {
public:
    explicit DeadlockRefinePass(SuperKernelGraph& inputGraph);
    ~DeadlockRefinePass() = default;
    
    bool Run(std::vector<SuperKernelScopeInfo>& scopes) override;
    std::string GetName() const override { return "DeadlockRefinePass"; }

private:
    /*!
     * \brief Find deadlock point in a scope
     * \param scope Scope to check
     * \param deadlockNode Output: node that causes deadlock (nullptr if no deadlock)
     * \param deadlockWaitNode Output: first Wait node before deadlock (nullptr if no deadlock)
     * \return true if deadlock found
     */
    bool FindDeadlockInScope(const SuperKernelScopeInfo& scope, 
                             SuperKernelBaseNode** deadlockNode,
                             SuperKernelBaseNode** deadlockWaitNode);
    
    /*!
     * \brief Split a scope at a Wait node
     * \param scope Original scope
     * \param waitNode Wait node to split at (not included in either result)
     * \param scopeBefore Output: scope before Wait node
     * \param scopeAfter Output: scope after Wait node
     */
    void SplitScopeAtWaitNode(const SuperKernelScopeInfo& scope,
                              SuperKernelBaseNode* waitNode,
                              SuperKernelScopeInfo& scopeBefore,
                              SuperKernelScopeInfo& scopeAfter);
    
    /*!
     * \brief Rebuild stream infos for a scope
     * \param scope Scope to rebuild
     */
    void RebuildStreamInfos(SuperKernelScopeInfo& scope);
    
    LockDetector lockDetector_;
};

// ============ Main Splitter Class ============

/*!
 * \class SuperKernelScopeSplitter
 * \brief Main class that orchestrates scope splitting passes
 * 
 * Usage:
 *   SuperKernelScopeSplitter splitter(graph);
 *   splitter.SplitGraph();
 *   const auto& scopes = splitter.GetScopeInfos();
 */
class SuperKernelScopeSplitter {
public:
    explicit SuperKernelScopeSplitter(SuperKernelGraph& inputGraph);
    ~SuperKernelScopeSplitter() = default;
    SuperKernelScopeSplitter(const SuperKernelScopeSplitter&) = delete;
    SuperKernelScopeSplitter& operator=(const SuperKernelScopeSplitter&) = delete;
    SuperKernelScopeSplitter(SuperKernelScopeSplitter&&) = default;
    SuperKernelScopeSplitter& operator=(SuperKernelScopeSplitter&&) = default;

    /*!
     * \brief Split graph into scopes using multi-pass pipeline
     * \return true on success
     */
    bool SplitGraph();

    /*!
     * \brief Get the resulting scope information
     * \return Reference to vector of scope information
     */
    std::vector<SuperKernelScopeInfo>& GetScopeInfos() noexcept { return scopeInfos_; }

    /*!
     * \brief Set Notify nodes' expand numbers to max kernel vec/cube in a scope
     * Should be called immediately after each scope is generated
     * \param scope Scope to process
     */
    static void SetNotifyNodesExpandNumForScope(SuperKernelScopeInfo& scope);

    /*!
     * \brief Print final scope results
     */
    void PrintFinalResults() const;

    SuperKernelGraph& graph_;
    std::vector<SuperKernelScopeInfo> scopeInfos_;
    std::vector<std::unique_ptr<ScopeSplitPass>> passes_;
};

#endif // __SK_SCOPE_SPLIT_H__
