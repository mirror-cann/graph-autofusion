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
 * - Pass 0: Initial scope splitting based on fusibility and scopeBitFlags
 * - Pass 1: Deadlock detection and scope refinement
 * - Pass 2: SchoMode kernel core trend based split refinement
 * - Pass 3: Remove event-only streams from scopes
 * 
 * Key Concepts:
 * - Scope: A collection of nodes from potentially multiple streams that can be fused together
 * - ScopeBitFlags: Bit flags that identify which scope a node belongs to
 * - Deadlock Detection: Prevents circular dependencies between streams
 */

#ifndef __SK_SCOPE_SPLIT_H__
#define __SK_SCOPE_SPLIT_H__

#include <optional>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "sk_candidate_heap.h"
#include "sk_graph.h"
#include "sk_log.h"
#include "sk_lock_detector.h"
#include "sk_options_manager.h"
#include "sk_scope_info.h"

/*!
 * \struct StreamState
 * \brief Processing state for a single stream during multi-stream splitting
 */
struct StreamState {
    uint64_t currentNodeId;              ///< Current node being processed
    bool isSuspended;                     ///< True if waiting for Notify event
    uint64_t waitingForNotify;            ///< Event ID being waited for (if suspended)
    bool isTerminated;                    ///< True if stream cannot continue in current scope

    StreamState()
        : currentNodeId(INVALID_TASK_ID),
          isSuspended(false),
          waitingForNotify(INVALID_TASK_ID),
          isTerminated(false) {}

    /*!
     * \brief Format stream state information for logging
     * \return Formatted string describing the stream state
     */
    std::string FormatStreamStateInfo() const {
        std::string info = "currentNodeId=";
        if (currentNodeId == INVALID_TASK_ID) {
            info += "INVALID";
        } else {
            info += std::to_string(currentNodeId);
        }

        info += ", isSuspended=";
        info += (isSuspended ? "true" : "false");

        info += ", waitingForNotify=";
        if (waitingForNotify == INVALID_TASK_ID) {
            info += "INVALID";
        } else {
            info += "0x" + std::to_string(waitingForNotify);
        }

        info += ", isTerminated=";
        info += (isTerminated ? "true" : "false");

        return info;
    }
};

// Forward declaration
class SuperKernelScopeSplitter;

// ============ Pass Base Class ============

/*!
 * \class ScopeSplitPass
 * \brief Base class for scope splitting passes
 */
class ScopeSplitPass {
public:
    explicit ScopeSplitPass(SuperKernelGraph& inputGraph) : graph_(inputGraph), splitter_(nullptr) {}
    virtual ~ScopeSplitPass() = default;

    /*!
     * \brief Execute the pass on the given scopes
     * \param scopes Input/output scope list
     * \return true on success
     */
    virtual bool Run(std::vector<SuperKernelScopeInfo>& scopes) = 0;

    virtual std::string GetName() const = 0;

    /*!
     * \brief Set the splitter reference for re-split requests
     * \param splitter Reference to the splitter
     */
    void SetSplitter(SuperKernelScopeSplitter* splitter) { splitter_ = splitter; }

    /*!
     * \brief Request a re-split of the graph
     * Called by passes when they modify node fusibility and need the pipeline to re-run
     */
    void RequestResplit();

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

    /*!
     * \brief Rebuild stream infos for a scope
     * \param scope Scope to rebuild
     */
    static void RebuildStreamInfos(SuperKernelScopeInfo& scope);

private:
    /*!
     * \brief Print detailed scope information to current log context
     * \param scopes Scope list to print
     * \param graph Graph for scope name lookup
     */
    static void PrintScopeDetails(const std::vector<SuperKernelScopeInfo>& scopes,
                                   const SuperKernelGraph& graph);

protected:
    SuperKernelGraph& graph_;
    SuperKernelScopeSplitter* splitter_;  ///< Reference to splitter for re-split requests
};

// ============ Pass 3: Event-Only Stream Remove (after SchoModeKernelSplit) ============

/*!
 * \class EventOnlyStreamRemovePass
 * \brief Pass 3: Mark event-only stream nodes as non-fusible to trigger re-split
 *
 * This pass runs after SchoModeKernelSplitPass to detect scopes that contain streams
 * with only event nodes (NODE_NOTIFY, NODE_WAIT, NODE_RESET, NODE_MEMORY_WRITE,
 * NODE_MEMORY_WAIT). Instead of removing such streams directly, it marks all event
 * nodes in those streams as non-fusible and signals the pipeline to re-run scope
 * splitting from the beginning.
 *
 * This approach prevents potential deadlocks that could occur if nodes are removed
 * from scopes after deadlock detection has already been performed.
 *
 * Processing logic:
 * - For each scope, collect nodes by stream
 * - If a stream's nodes are ALL event nodes, mark them as non-fusible
 * - Clear scopes to signal re-split is needed
 * - The pipeline will re-run from Pass 0 with updated fusibility
 */
class EventOnlyStreamRemovePass : public ScopeSplitPass {
public:
    explicit EventOnlyStreamRemovePass(SuperKernelGraph& inputGraph);
    ~EventOnlyStreamRemovePass() = default;

    bool Run(std::vector<SuperKernelScopeInfo>& scopes) override;
    std::string GetName() const override { return "EventOnlyStreamRemovePass"; }

private:
    /*!
     * \brief Check if a node is an event node type
     * \param node Node to check
     * \return true if the node is an event node type
     */
    bool IsEventNode(SuperKernelBaseNode* node) const;

    /*!
     * \brief Collect nodes grouped by stream index for a scope
     * \param scope Scope to analyze
     * \param streamNodes Output: map from stream index to list of nodes
     */
    void CollectNodesPerStream(const SuperKernelScopeInfo& scope,
                               std::unordered_map<uint32_t, std::vector<SuperKernelBaseNode*>>& streamNodes);

    /*!
     * \brief Check if all nodes in a vector are event nodes
     * \param nodes Vector of nodes to check
     * \return true if all nodes are event nodes (and vector is non-empty)
     */
    bool IsStreamAllEventNodes(const std::vector<SuperKernelBaseNode*>& nodes) const;

    /*!
     * \brief Process a single scope and mark event-only stream nodes as non-fusible
     * \param scope Scope to process
     * \return Number of nodes marked as non-fusible
     */
    uint32_t ProcessScope(SuperKernelScopeInfo& scope);

    uint32_t markedCount_;    ///< Total number of nodes marked as non-fusible
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
    explicit InitialScopeSplitPass(SuperKernelGraph& inputGraph, SkHeapType heapType);
    ~InitialScopeSplitPass() = default;

    bool Run(std::vector<SuperKernelScopeInfo>& scopes) override;
    std::string GetName() const override { return "InitialScopeSplitPass"; }

private:
    // ============ Stream State Management ============
    void InitStreamStates();
    bool ResetStreamStates();  // Changed from void to bool to propagate errors
    bool SkipUnfusibleNodes();  // Changed from void to bool to propagate errors
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
    bool HandleUnfusibleNotifyNode(SuperKernelBaseNode* notifyNode, uint32_t streamIdx);
    bool ResumeSuspendedWaitStreams(SuperKernelBaseNode* notifyNode, uint32_t notifyStreamIdx);
    
    // ============ Scope Building ============
    void AddStreamInfoToScope(SuperKernelScopeInfo& scopeInfo, SuperKernelBaseNode* node);
    bool BuildCurrentScope(SuperKernelScopeInfo& scopeInfo);

    // ============ Diagnostic Logging ============
    void LogFusibleNodeSearchResult();
    
    // ============ Member Variables ============
    std::unordered_map<uint32_t, StreamState> streamStates_;
    std::set<uint64_t> visitedNotifies_;
    std::set<uint64_t> processedNodes_;
    SkCandidateHeap nodeHeap_;
    std::bitset<MAX_SCOPE_NUM> currentScopeBitFlags_;
};

// ============ Pass 2: Deadlock Detection and Refinement ============

/*!
 * \enum ScopeProcessResult
 * \brief Result of processing a single scope for deadlock detection
 */
enum class ScopeProcessResult {
    NO_DEADLOCK,           ///< No deadlock found, scope is safe and added to outputScopes
    DEADLOCK_RESOLVED,     ///< Deadlock found and successfully split, remaining part returned via pendingScope
    DEADLOCK_UNRESOLVED    ///< Deadlock found but cannot split (no Wait node), caller should treat as fatal and abort
};

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
    * \brief Process a single scope with deadlock detection and one-step splitting.
    *
    * This function either:
    * - Detects no deadlock and moves the whole scope into outputScopes; or
    * - Detects a deadlock and splits at the nearest preceding Wait node, moving
    *   the front part into outputScopes and returning the remaining part via
    *   pendingScope; or
    * - Detects an unresolvable deadlock (no suitable Wait node), in which case
    *   the caller should treat this as a fatal error and abort refinement.
    *
    * \param scopeToProcess The scope to process (moved)
    * \param outputScopes Vector to store deadlock-free scopes produced from the front part
    * \param pendingScope Optional: remaining part of the scope that still needs processing
    * \return Result of the processing (NO_DEADLOCK, DEADLOCK_RESOLVED, or DEADLOCK_UNRESOLVED)
    */
    ScopeProcessResult ProcessSingleScope(
        SuperKernelScopeInfo&& scopeToProcess,
        std::vector<SuperKernelScopeInfo>& outputScopes,
        std::optional<SuperKernelScopeInfo>& pendingScope);

    LockDetector lockDetector_;
};

// ============ Pass 3: SchoMode Kernel Core Split Refinement ============

/*!
 * \enum SchoModeScopeProcessResult
 * \brief Result of processing a single scope for SchoMode kernel split refinement
 */
enum class SchoModeScopeProcessResult {
    NO_SPLIT,          ///< No split required, scope is output as-is
    SPLIT_RESOLVED     ///< Split required and completed, remaining part returned via pendingScope
};

/*!
 * \class SchoModeKernelSplitPass
 * \brief Pass 3: Refine scopes based on SchoMode kernel core trend
 *
 * Rules:
 * - Traverse nodes in a scope and merge kernel core requirement using max(cube), max(vec)
 * - When a kernel node with IsSchoModeOn()==true is encountered:
 *   - If its core requirement is greater than merged previous requirement, keep merging
 *   - If its core requirement is smaller than merged previous requirement, split at this node
 * - Split point kernel is included in the "after" scope
 */
class SchoModeKernelSplitPass : public ScopeSplitPass {
public:
    explicit SchoModeKernelSplitPass(SuperKernelGraph& inputGraph);
    ~SchoModeKernelSplitPass() = default;

    bool Run(std::vector<SuperKernelScopeInfo>& scopes) override;
    std::string GetName() const override { return "SchoModeKernelSplitPass"; }

private:
    /*!
     * \brief Process a single scope and split if SchoMode rule requires
     * \param scopeToProcess Input scope to process (moved)
     * \param outputScopes Output refined scopes
     * \param pendingScope Remaining part after split, if any
     * \return Processing result
     */
    SchoModeScopeProcessResult ProcessSingleScope(
        SuperKernelScopeInfo&& scopeToProcess,
        std::vector<SuperKernelScopeInfo>& outputScopes,
        std::optional<SuperKernelScopeInfo>& pendingScope);

    /*!
     * \brief Split a scope at the split node (split node belongs to scopeAfter)
     * \param scope Original scope
     * \param splitNode Node where split happens
     * \param scopeBefore Output: nodes before splitNode
     * \param scopeAfter Output: nodes from splitNode to end
     */
    void SplitScopeAtNode(const SuperKernelScopeInfo& scope,
                          SuperKernelBaseNode* splitNode,
                          SuperKernelScopeInfo& scopeBefore,
                          SuperKernelScopeInfo& scopeAfter);
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
    explicit SuperKernelScopeSplitter(SuperKernelGraph& inputGraph, SuperKernelOptionsManager& opts);
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
     * \brief Request a re-split of the graph
     * Called by passes when they modify node fusibility and need the pipeline to re-run
     */
    void RequestResplit() { needResplit_ = true; }

    /*!
     * \brief Check if a re-split has been requested
     * \return true if re-split is needed
     */
    bool NeedsResplit() const { return needResplit_; }

    /*!
     * \brief Print final scope results
     */
    void PrintFinalResults() const;

    SuperKernelGraph& graph_;
    std::vector<SuperKernelScopeInfo> scopeInfos_;
    std::vector<std::unique_ptr<ScopeSplitPass>> passes_;
    bool needResplit_ = false;  ///< Flag to signal re-split is needed
};

#endif // __SK_SCOPE_SPLIT_H__
