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
 * \file sk_scope_split.h
 * \brief Super Kernel Scope Splitter - splits multi-stream graph into scopes
 * 
 * This module implements the scope splitting algorithm for multi-stream graphs.
 * A "scope" represents a group of nodes that can be fused into a single Super Kernel.
 * 
 * Key Concepts:
 * - Scope: A collection of nodes from potentially multiple streams that can be fused together
 * - ScopeBitFlags: Bit flags that identify which scope a node belongs to
 *   - Nodes with the same ScopeBitFlags can be fused into the same scope
 *   - Nodes with different ScopeBitFlags cannot be fused together
 * - Deadlock Detection: Prevents circular dependencies between streams that would cause hangs
 * 
 * Splitting Algorithm:
 * 1. For each scope, determine ScopeBitFlags from the node with minimum node ID
 * 2. Add nodes to heap based on their node ID (priority queue ensures ordered processing)
 * 3. For each node, check:
 *    - ScopeBitFlags match (nodes with different flags go to different scopes)
 *    - Node is fusible (not permanently marked unfusible)
 *    - No deadlock would occur (checked via LockDetector)
 * 4. Special handling for event nodes (Wait/Notify/Reset):
 *    - Wait: Suspend stream until corresponding Notify is processed
 *    - Notify: Resume suspended streams waiting for this event
 *    - Reset: Clear event state for reuse
 * 5. When no more nodes can be added, start a new scope
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

/*!
 * \class SuperKernelScopeSplitter
 * \brief Splits a multi-stream graph into scopes for Super Kernel fusion
 * 
 * Usage:
 *   SuperKernelScopeSplitter splitter(graph);
 *   splitter.SplitGraph();
 *   const auto& scopes = splitter.GetScopeInfos();
 */
class SuperKernelScopeSplitter {
public:
    explicit SuperKernelScopeSplitter(SuperKernelGraph& graph) : graph(graph) {}
    ~SuperKernelScopeSplitter() = default;
    SuperKernelScopeSplitter(const SuperKernelScopeSplitter&) = delete;
    SuperKernelScopeSplitter& operator=(const SuperKernelScopeSplitter&) = delete;
    SuperKernelScopeSplitter(SuperKernelScopeSplitter&&) = default;
    SuperKernelScopeSplitter& operator=(SuperKernelScopeSplitter&&) = default;

    /*!
     * \brief Split graph into scopes
     * \return true on success
     */
    bool SplitGraph();

    /*!
     * \brief Get the resulting scope information
     * \return Reference to vector of scope information
     */
    std::vector<SuperKernelScopeInfo>& GetScopeInfos() noexcept { return scopeInfos; }

private:
    // ============ Single Stream Methods ============
    
    /*!
     * \brief Find the first fusible kernel node starting from given index
     * \param curNodeIdx Starting node index
     * \return Index of first fusible kernel node, or INVALID_TASK_ID if none found
     */
    uint64_t FindSingleStreamAvailableHeadNode(uint64_t curNodeIdx) const;

    /*!
     * \brief Generate scope info starting from a head node (single stream)
     * \param curNodeIdx Starting node index
     * \return Next node index after the generated scope
     */
    uint64_t GenerateSingleStreamScopeInfosByNodeIdx(uint64_t curNodeIdx);

    // ============ Multi-Stream Methods ============
    
    /*!
     * \brief Initialize the node heap by trying to add current nodes from all streams
     */
    void InitNodeHeap();

    /*!
     * \brief Try to add the current node of a stream to the heap
     * \param streamIdx Stream index to process
     * 
     * This method checks:
     * - Stream state (not terminated/suspended)
     * - Node not already processed
     * - ScopeBitFlags match
     * - Node is fusible
     * - No deadlock would occur
     */
    void TryAddNodeToHeap(uint32_t streamIdx);

    /*!
     * \brief Process a Notify node - resume suspended streams
     * \param notifyNode The Notify node being processed
     */
    void ProcessNotifyNode(SuperKernelBaseNode* notifyNode);

    /*!
     * \brief Process a Reset node - clear event state
     * \param resetNode The Reset node being processed
     */
    void ProcessResetNode(SuperKernelBaseNode* resetNode);

    /*!
     * \brief Handle Wait node in TryAddNodeToHeap
     * \param waitNode The Wait node to process
     * \param streamIdx Stream index
     */
    void HandleWaitNode(SuperKernelBaseNode* waitNode, uint32_t streamIdx);

    /*!
     * \brief Add stream info to a scope when a node is added
     * \param scopeInfo Scope to update
     * \param node Node being added
     */
    void AddStreamInfoToScope(SuperKernelScopeInfo& scopeInfo, SuperKernelBaseNode* node);

    /*!
     * \brief Skip over permanently unfusible nodes, Wait nodes, and Reset nodes in all streams
     * 
     * These nodes should not be processed as scope starting points:
     * - Unfusible nodes: cannot participate in scope fusion
     * - Wait nodes: depend on Notify nodes from other streams
     * - Reset nodes: event cleanup, not computational
     * Notify nodes are fusible and will not be skipped.
     */
    void SkipUnfusibleAndEventNodes();

    /*!
     * \brief Reset stream states for the next scope iteration
     * 
     * Resets isTerminated and isSuspended flags while keeping currentNodeIdx.
     * Also skips unfusible nodes and event nodes.
     */
    void ResetStreamStates();

    /*!
     * \brief Check if all streams have finished processing
     * \return true if all streams are terminated, suspended, or finished
     */
    bool AllStreamsFinished() const;

    /*!
     * \brief Determine currentScopeBitFlags from the node with minimum currentNodeIdx
     * \return false if no fusible node found (should end splitting)
     * 
     * The scope's bit flags are determined by the node with the smallest ID
     * among all streams' current nodes. This ensures consistent scope boundaries.
     */
    bool DetermineCurrentScopeBitFlags();

    /*!
     * \brief Reset all splitting state (called before starting splitting)
     */
    void ResetSplittingState();

    /*!
     * \brief Print scope splitting results for debugging
     */
    void PrintScopeResults() const;

    /*!
     * \brief Set Notify nodes' expand numbers to max kernel vec/cube in scope
     * \param scopeInfo Scope to process
     * 
     * This ensures the scope runs before the next scope (which contains waiting Wait nodes).
     * Without this, deadlock detector would block the Wait nodes incorrectly.
     */
    void SetNotifyNodesExpandNum(SuperKernelScopeInfo& scopeInfo);

    /*!
     * \brief Get scope names string from scopeBitFlags
     * \param scopeBitFlags Bit flags to convert
     * \return Comma-separated scope names string
     */
    std::string GetScopeNamesFromBitFlags(const std::bitset<MAX_SCOPE_NUM>& scopeBitFlags) const;

    /*!
     * \brief Print nodes in a scope
     * \param scopeIdx Scope index
     * \param scope Scope info
     */
    void PrintScopeNodes(size_t scopeIdx, const SuperKernelScopeInfo& scope) const;

    /*!
     * \brief Print stream infos in a scope
     * \param scopeIdx Scope index
     * \param scope Scope info
     */
    void PrintScopeStreamInfos(size_t scopeIdx, const SuperKernelScopeInfo& scope) const;

    // ============ Member Variables ============
    
    SuperKernelGraph& graph;                              ///< Reference to the graph being split
    std::vector<SuperKernelScopeInfo> scopeInfos;         ///< Resulting scope information

    // Multi-stream splitting state (used as member variables to simplify function signatures)
    std::unordered_map<uint32_t, StreamState> streamStates_;  ///< Per-stream processing state
    std::set<uint64_t> visitedNotifies_;                      ///< Set of visited Notify node IDs
    std::set<uint64_t> processedNodes_;                       ///< Set of processed node IDs
    std::priority_queue<uint64_t, std::vector<uint64_t>, 
                        std::greater<uint64_t>> nodeHeap_;    ///< Min-heap of candidate node IDs
    LockDetector lockDetector;                                ///< Deadlock detection utility
    std::bitset<MAX_SCOPE_NUM> currentScopeBitFlags_;         ///< Current scope's bit flags
};

#endif // __SK_SCOPE_SPLIT_H__
