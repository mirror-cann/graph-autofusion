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
 * \file sk_lock_detector.h
 * \brief
 */
 
#pragma once

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>
#include "sk_types.h"
#include "sk_node.h"
#include "sk_graph.h"
#include "acl/acl.h"

/**
 * @class LockDetector
 * @brief 死锁检测器，用于检测SuperKernel融合时是否会引入死锁
 *
 * 该类通过分析节点的核心使用情况和依赖关系，判断是否可以将节点融合到SuperKernel中。
 */
class LockDetector {
private:
    // 存储已检测节点列表
    std::vector<uint64_t> nodes;
    uint32_t depOpCubeNum;
    uint32_t depOpVecNum;
    uint32_t superKernelCubeNum;
    uint32_t superKernelVecNum;
    uint32_t deviceRealCubeNum;
    uint32_t deviceRealVecNum;
    std::set<uint32_t> skStreamIds;
    bool isExistWaitFlag;

    void Init();

    std::pair<uint64_t, uint64_t> GetDeviceCores() const;

    std::pair<uint64_t, uint64_t> GetAvailableCores(bool isSuperKernel) const;

    std::pair<uint32_t, uint32_t> GetNodeCoreNum(const SuperKernelBaseNode& node);

    bool IsInSKStream(const SuperKernelBaseNode& node);

    bool HasDeadlock(SuperKernelBaseNode* curNode, SuperKernelGraph& graph);

    bool HasEnoughCores(const SuperKernelBaseNode* curNode, bool isSuperKernel);

public:
    LockDetector() {
        Init();
    }

    /**
     * @brief 重置检测器状态
     * @param curNode 当前节点
     */
    void Reset(SuperKernelGraph& graph);

    /**
     * @brief 判断节点是否可融合到SuperKernel中
     * @param curNode 待检测的节点
     * @param graph 图对象
     * @return true 可融合，false 不可融合（会导致死锁）
     */
    bool IsFusible(const SuperKernelBaseNode& curNode, SuperKernelGraph& graph);
};