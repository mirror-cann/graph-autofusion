/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Source Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_candidate_heap.h"
#include "sk_log.h"

SkCandidateHeap::SkCandidateHeap(SuperKernelGraph& inputGraph, SkHeapType heapModeIn)
    : graph_(inputGraph)
    , heapMode(heapModeIn)
    , defaultNodes_(CompareByNodeId)
    , kernelNodes_(CompareByNodeId)
    , nonKernelNodes_(CompareByNodeId)
    , prevKernelTypeClass_(KernelTypeClass::OTHER)
    , prevKernelStreamIdx_(0)
    , prevStreamIdx_(0)
    , isFirstSelection_(true)
{}

void SkCandidateHeap::push(uint64_t nodeId)
{
    if (heapMode == SkHeapType::PRIORITY_QUEUE) {
        nodeHeap_.push(nodeId);
        return;
    }
    auto node = graph_.GetNodeById(nodeId);
    if (node == nullptr) {
        SK_LOGW("SkCandidateHeap::push: attempting to push null node, ignored");
        return;
    }

    SkNodeType nodeType = node->GetNodeType();
    if (nodeType == SkNodeType::NODE_DEFAULT) {
        defaultNodes_.insert(node);
        SK_LOGD("SkCandidateHeap::push: added default node %s, defaultCount=%zu",
            node->Format().c_str(), defaultNodes_.size());
    } else if (nodeType == SkNodeType::NODE_KERNEL) {
        if (node->IsScopeNode()) {
            nonKernelNodes_.insert(node);
            SK_LOGD("SkCandidateHeap::push: added non-kernel node %s (type=%d), nonKernelCount=%zu",
                node->Format().c_str(), static_cast<int>(nodeType), nonKernelNodes_.size());
        } else {
            kernelNodes_.insert(node);
            SK_LOGD("SkCandidateHeap::push: added kernel node %s, kernelCount=%zu",
                node->Format().c_str(), kernelNodes_.size());
        }
    } else if (nodeType == SkNodeType::NODE_NOTIFY ||
               nodeType == SkNodeType::NODE_WAIT ||
               nodeType == SkNodeType::NODE_RESET) {
        nonKernelNodes_.insert(node);
        SK_LOGD("SkCandidateHeap::push: added non-kernel node %s (type=%d), nonKernelCount=%zu",
            node->Format().c_str(), static_cast<int>(nodeType), nonKernelNodes_.size());
    } else {
        SK_LOGW("SkCandidateHeap::push: unsupported node type %d for node %s, ignored",
            static_cast<int>(nodeType), node->Format().c_str());
    }
}

uint64_t SkCandidateHeap::pop()
{
    if (heapMode == SkHeapType::PRIORITY_QUEUE) {
        uint64_t nodeId = nodeHeap_.top();
        nodeHeap_.pop();
        return nodeId;
    }
    if (empty()) {
        SK_LOGE("SkCandidateHeap::pop: heap size is zero");
        return 0;
    }

    SuperKernelBaseNode* selectedNode = nullptr;

    if (!defaultNodes_.empty()) {
        selectedNode = *defaultNodes_.begin();
        defaultNodes_.erase(selectedNode);

        SK_LOGD("SkCandidateHeap::pop: selected default node %s, remaining defaultCount=%zu",
            selectedNode->Format().c_str(), defaultNodes_.size());
        return selectedNode->GetNodeId();
    }

    if (!nonKernelNodes_.empty()) {
        selectedNode = SelectNextNonKernelNode();
        if (selectedNode != nullptr) {
            nonKernelNodes_.erase(selectedNode);

            // For non-kernel nodes, update stream index but keep kernel type class
            prevStreamIdx_ = selectedNode->GetStreamIdxInGraph();
            // Note: non-kernel nodes don't change the kernel type class

            SK_LOGD("SkCandidateHeap::pop: selected non-kernel node %s, remaining nonKernelCount=%zu",
                selectedNode->Format().c_str(), nonKernelNodes_.size());
            return selectedNode->GetNodeId();
        }
    }

    if (!kernelNodes_.empty()) {
        selectedNode = SelectNextKernelNode();
        if (selectedNode != nullptr) {
            kernelNodes_.erase(selectedNode);

            // Update previous kernel state
            prevKernelTypeClass_ = GetKernelTypeClass(selectedNode->GetKernelType());
            prevKernelStreamIdx_ = selectedNode->GetStreamIdxInGraph();
            prevStreamIdx_ = selectedNode->GetStreamIdxInGraph();
            isFirstSelection_ = false;

            SK_LOGD("SkCandidateHeap::pop: selected kernel node %s, remaining kernelCount=%zu",
                    selectedNode->Format().c_str(), kernelNodes_.size());
            return selectedNode->GetNodeId();
        }
    }

    SK_LOGE("SkCandidateHeap::pop: cannot select node, both heaps are empty");
    return 0;
}

bool SkCandidateHeap::empty() const
{    
    if (heapMode == SkHeapType::PRIORITY_QUEUE) {
        return nodeHeap_.empty();
    }
    return defaultNodes_.empty() && kernelNodes_.empty() && nonKernelNodes_.empty();
}

size_t SkCandidateHeap::size() const
{
    if (heapMode == SkHeapType::PRIORITY_QUEUE) {
        return nodeHeap_.size();
    }
    return defaultNodes_.size() + kernelNodes_.size() + nonKernelNodes_.size();
}

bool SkCandidateHeap::HasKernelNodes() const
{
    return !kernelNodes_.empty();
}

void SkCandidateHeap::clear()
{
    while (!nodeHeap_.empty()) {
        nodeHeap_.pop();
    }
    defaultNodes_.clear();
    kernelNodes_.clear();
    nonKernelNodes_.clear();
    ResetSelectionState();
}

void SkCandidateHeap::reset()
{
    clear();
}

void SkCandidateHeap::ResetSelectionState()
{
    prevKernelTypeClass_ = KernelTypeClass::OTHER;
    prevKernelStreamIdx_ = 0;
    prevStreamIdx_ = 0;
    isFirstSelection_ = true;
}

SuperKernelBaseNode* SkCandidateHeap::SelectNextKernelNode()
{
    if (kernelNodes_.empty()) {
        return nullptr;
    }

    SuperKernelBaseNode* selectedNode = nullptr;

    if (isFirstSelection_) {
        // First selection: prefer MIX kernel, otherwise choose node with smallest nodeId
        for (auto* node : kernelNodes_) {
            if (IsMixKernelType(node->GetKernelType())) {
                selectedNode = node;
                SK_LOGD("SkCandidateHeap::SelectNextKernelNode: first selection, chose MIX node %s",
                        selectedNode->Format().c_str());
                return selectedNode;
            }
        }
        // No MIX kernel found, choose node with smallest nodeId
        selectedNode = *kernelNodes_.begin();
        SK_LOGD("SkCandidateHeap::SelectNextKernelNode: first selection, no MIX found, chose node %s",
                selectedNode->Format().c_str());
        return selectedNode;
    }

    // Apply selection rules based on previous kernel type
    bool foundByRule = false;

    if (prevKernelTypeClass_ == KernelTypeClass::MIX) {
        // Rule 1: After MIX, prefer MIX (prioritize different stream)
        for (auto* node : kernelNodes_) {
            if (IsMixKernelType(node->GetKernelType()) &&
                node->GetStreamIdxInGraph() != prevKernelStreamIdx_) {
                selectedNode = node;
                foundByRule = true;
                break;
            }
        }
    } else if (prevKernelTypeClass_ == KernelTypeClass::VEC) {
        // Rule 2: After VEC, prefer CUBE (prioritize different stream)
        for (auto* node : kernelNodes_) {
            if (IsCubeKernelType(node->GetKernelType()) &&
                node->GetStreamIdxInGraph() != prevKernelStreamIdx_) {
                selectedNode = node;
                foundByRule = true;
                break;
            }
        }
    } else if (prevKernelTypeClass_ == KernelTypeClass::CUBE) {
        // Rule 3: After CUBE, prefer VEC (prioritize different stream)
        for (auto* node : kernelNodes_) {
            if (IsVecKernelType(node->GetKernelType()) &&
                node->GetStreamIdxInGraph() != prevKernelStreamIdx_) {
                selectedNode = node;
                foundByRule = true;
                break;
            }
        }
    }

    // If no rule matched or previous is OTHER type:
    // Prefer node from different stream with smallest nodeId
    if (!foundByRule) {
        for (auto* node : kernelNodes_) {
            if (node->GetStreamIdxInGraph() != prevKernelStreamIdx_) {
                selectedNode = node;
                foundByRule = true;
                break;
            }
        }

        // If all nodes are on the same stream, choose smallest nodeId
        if (!foundByRule) {
            selectedNode = *kernelNodes_.begin();
        }
    }

    return selectedNode;
}

SuperKernelBaseNode* SkCandidateHeap::SelectNextNonKernelNode()
{
    if (nonKernelNodes_.empty()) {
        return nullptr;
    }

    // Select non-kernel node with smallest nodeId
    return *nonKernelNodes_.begin();
}

// Helper functions
bool SkCandidateHeap::IsMixKernelType(SkKernelType type)
{
    return type == SkKernelType::MIX_AIC_1_1 || type == SkKernelType::MIX_AIC_1_2;
}

bool SkCandidateHeap::IsVecKernelType(SkKernelType type)
{
    return type == SkKernelType::AIV_ONLY || type == SkKernelType::MIX_AIV_1_0;
}

bool SkCandidateHeap::IsCubeKernelType(SkKernelType type)
{
    return type == SkKernelType::AIC_ONLY || type == SkKernelType::MIX_AIC_1_0;
}

SkCandidateHeap::KernelTypeClass SkCandidateHeap::GetKernelTypeClass(SkKernelType type)
{
    if (IsMixKernelType(type)) {
        return KernelTypeClass::MIX;
    }
    if (IsVecKernelType(type)) {
        return KernelTypeClass::VEC;
    }
    if (IsCubeKernelType(type)) {
        return KernelTypeClass::CUBE;
    }
    return KernelTypeClass::OTHER;
}

bool SkCandidateHeap::CompareByNodeId(SuperKernelBaseNode* a, SuperKernelBaseNode* b)
{
    return a->GetNodeId() < b->GetNodeId();
}
