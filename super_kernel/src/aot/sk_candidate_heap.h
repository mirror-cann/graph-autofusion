/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_CANDIDATE_HEAP_H__
#define __SK_CANDIDATE_HEAP_H__

#include <set>
#include <vector>
#include <queue>
#include "sk_graph.h"
#include "sk_node.h"


enum class SkHeapType : uint8_t { 
    PRIORITY_QUEUE = 0,
    CUSTOMIZE_QUEUE = 1,
    DEFAULT_QUEUE = 0xff
};

/*!
 * \class SkCandidateHeap
 * \brief A candidate node selector with priority-based selection for scope splitting.
 *
 * This class provides push/pop operations for managing candidate nodes during scope splitting.
 * It supports multiple node types: KERNEL, NOTIFY, WAIT, RESET.
 *
 * Selection rules:
 * 1. When there are non-KERNEL nodes (NOTIFY/WAIT/RESET) in the heap:
 *    - Non-KERNEL nodes have priority and are selected by nodeId (smallest first)
 * 2. When there are no non-KERNEL nodes:
 *    - KERNEL nodes are selected according to SkTaskSorter::SelectNextNode rules:
 *      - First selection: choose node with smallest nodeId
 *      - After MIX: prefer MIX (prioritize different stream)
 *      - After VEC: prefer CUBE (prioritize different stream)
 *      - After CUBE: prefer VEC (prioritize different stream)
 *      - Default: prefer node from different stream, then smallest nodeId
 *
 * The class tracks the previous kernel's type and stream index to implement the selection rules.
 */
class SkCandidateHeap {
public:
    /*!
     * \brief Kernel type classification for selection rules
     */
    enum class KernelTypeClass { MIX, VEC, CUBE, OTHER };

    SkCandidateHeap(SuperKernelGraph& inputGraph, SkHeapType heapModeIn);
    ~SkCandidateHeap() = default;

    // Disable copy
    SkCandidateHeap(const SkCandidateHeap&) = delete;
    SkCandidateHeap& operator=(const SkCandidateHeap&) = delete;

    // Enable move
    SkCandidateHeap(SkCandidateHeap&&) = default;
    SkCandidateHeap& operator=(SkCandidateHeap&&) = default;

    /*!
     * \brief push a node into the candidate heap
     * \param node The node to add (can be KERNEL, NOTIFY, WAIT, or RESET type)
     *
     * The node is automatically categorized into kernel or non-kernel collection
     * based on its node type.
     */
    void push(uint64_t nodeId);

    /*!
     * \brief pop and return the next node according to selection rules
     * \return The selected node, or nullptr if heap is empty
     *
     * Selection priority:
     * 1. If there are KERNEL nodes, select according to kernel selection rules
     * 2. Otherwise, select the non-kernel node with smallest nodeId
     */
    uint64_t pop();

    /*!
     * \brief Check if the heap is empty
     * \return true if both kernel and non-kernel collections are empty
     */
    bool empty() const;

    /*!
     * \brief Get total number of nodes in the heap
     * \return Total count of kernel and non-kernel nodes
     */
    size_t size() const;

    /*!
     * \brief Check if there are any kernel nodes
     * \return true if there is at least one kernel node
     */
    bool HasKernelNodes() const;

    /*!
     * \brief clear all nodes and reset state
     */
    void clear();

    /*!
     * \brief reset all variables and state to initial values
     * 
     * This is equivalent to clear() - clears all nodes and resets selection state.
     * Use this for a more intuitive name when reinitializing the heap.
     */
    void reset();

    /*!
     * \brief Get the previous kernel type class
     * \return The kernel type class of the last popped kernel node
     */
    KernelTypeClass GetPrevKernelTypeClass() const { return prevKernelTypeClass_; }

    /*!
     * \brief Get the previous stream index
     * \return The stream index of the last popped node
     */
    uint32_t GetPrevStreamIdx() const { return prevStreamIdx_; }

    /*!
     * \brief Set the previous kernel type class (for initialization or external control)
     * \param typeClass The kernel type class to set
     */
    void SetPrevKernelTypeClass(KernelTypeClass typeClass) { prevKernelTypeClass_ = typeClass; }

    /*!
     * \brief Set the previous stream index (for initialization or external control)
     * \param streamIdx The stream index to set
     */
    void SetPrevStreamIdx(uint32_t streamIdx) { prevStreamIdx_ = streamIdx; }

    /*!
     * \brief reset selection state (but keep nodes)
     * 
     * This resets the previous kernel type class to OTHER and marks the next
     * selection as the first selection.
     */
    void ResetSelectionState();

private:
    /*!
     * \brief Select next kernel node according to rules
     * \return The selected kernel node, or nullptr if no kernel nodes
     */
    SuperKernelBaseNode* SelectNextKernelNode();

    /*!
     * \brief Select next non-kernel node (smallest nodeId)
     * \return The selected non-kernel node, or nullptr if no non-kernel nodes
     */
    SuperKernelBaseNode* SelectNextNonKernelNode();

    // Helper functions for kernel type classification
    static bool IsMixKernelType(SkKernelType type);
    static bool IsVecKernelType(SkKernelType type);
    static bool IsCubeKernelType(SkKernelType type);
    static KernelTypeClass GetKernelTypeClass(SkKernelType type);
    // Comparator for sorting nodes by nodeId
    static bool CompareByNodeId(SuperKernelBaseNode* a, SuperKernelBaseNode* b);

    SuperKernelGraph& graph_;
    SkHeapType heapMode = SkHeapType::PRIORITY_QUEUE;
    // Node collections
    std::set<SuperKernelBaseNode*, bool(*)(SuperKernelBaseNode*, SuperKernelBaseNode*)> kernelNodes_;
    std::set<SuperKernelBaseNode*, bool(*)(SuperKernelBaseNode*, SuperKernelBaseNode*)> nonKernelNodes_;

    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> nodeHeap_;

    // State tracking for selection rules
    KernelTypeClass prevKernelTypeClass_;
    uint32_t prevKernelStreamIdx_;
    uint32_t prevStreamIdx_;
    bool isFirstSelection_;
};

#endif // __SK_CANDIDATE_HEAP_H__
