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
 
#ifndef __SK_LOCK_DETECTOR_H__
#define __SK_LOCK_DETECTOR_H__


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

public:
    LockDetector() {
        Init();
    }


    static bool GetDeviceCores();

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
    bool IsFusible(SuperKernelBaseNode& curNode, SuperKernelGraph& graph);

private:
    
    void Init();

    std::pair<uint64_t, uint64_t> GetAvailableCores(bool isSuperKernel) const;

    std::pair<uint32_t, uint32_t> GetNodeCoreNum(const SuperKernelBaseNode& node);

    bool IsInSKStream(const SuperKernelBaseNode& node);

    void UpdateNodeInfo(const SuperKernelBaseNode& node);

    void UpdateSKRangeInStream(const SuperKernelBaseNode& node);

    bool IsBeforeSKRange(const SuperKernelBaseNode& node);

    bool IsAfterSKRange(const SuperKernelBaseNode& node);

    bool HasDeadlock(SuperKernelBaseNode* curNode, SuperKernelGraph& graph);

    bool HasEnoughCores(const SuperKernelBaseNode* curNode, bool isSuperKernel);

    
    std::vector<uint64_t> nodes; // visited nodes
    uint32_t depOpCubeNum; // visited op cube num outside superkernel
    uint32_t depOpVecNum;  // visited op vec num outside superkernel
    uint32_t superKernelCubeNum; // fused op cube num in superkernel
    uint32_t superKernelVecNum; // fused op vec num in superkernel
    static int64_t deviceRealCubeNum;
    static int64_t deviceRealVecNum;
    std::set<uint32_t> skStreamIds;
    bool isExistWaitFlag;
    uint32_t nodeNum;
    uint32_t kernelNodeNum;
    std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> skRangeInStream;
};

#endif // __SK_LOCK_DETECTOR_H__