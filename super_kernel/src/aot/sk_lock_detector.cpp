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
 * \file sk_lock_detector.cpp
 * \brief
 */
 
#include "sk_lock_detector.h"
#include "sk_log.h"

void LockDetector::Init() {
    static std::pair<uint64_t, uint64_t> coreInfos = GetDeviceCores();
    nodes.clear();
    depOpCubeNum = 0; // 依赖算子中 ---换名
    depOpVecNum = 0; // 依赖算子中
    superKernelCubeNum = 0; // sk
    superKernelVecNum = 0;
    deviceRealCubeNum = coreInfos.first;
    deviceRealVecNum = coreInfos.second;
    skStreamIds.clear();
    isExistWaitFlag = false;
}
std::pair<uint64_t, uint64_t> LockDetector::GetDeviceCores() const {
    // 获取deviceId
    int32_t deviceId;
    aclError ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("[lock detector] get device failed!!!");
    }
    // 获取CubeNum、VecNum
    int64_t cubeNum;
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &cubeNum);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("[lock detector] get cube num failed!!!");
    }
    int64_t vecNum;
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vecNum);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("[lock detector] get vector num failed!!!");
    }
    return {cubeNum, vecNum};
}

std::pair<uint64_t, uint64_t> LockDetector::GetAvailableCores(bool isSuperKernel) const {
    if (isSuperKernel) {
        return {deviceRealCubeNum - depOpCubeNum, deviceRealVecNum - depOpVecNum};
    } else {
        return {deviceRealCubeNum - superKernelCubeNum, deviceRealVecNum - superKernelVecNum};
    }
}

std::pair<uint32_t, uint32_t> LockDetector::GetNodeCoreNum(const SuperKernelBaseNode& node) {
    uint32_t numBlocks = node.GetNumBlocks();
    SkKernelType kernelType = node.GetKernelType();

    if (kernelType == SkKernelType::AIC_ONLY || kernelType == SkKernelType::MIX_AIC_1_0) {
        return {numBlocks, 0};
    } else if (kernelType == SkKernelType::AIV_ONLY || kernelType == SkKernelType::MIX_AIV_1_0) {
        return {0, numBlocks};
    } else if (kernelType == SkKernelType::MIX_AIC_1_1) {
        return {numBlocks, numBlocks};
    } else if (kernelType == SkKernelType::MIX_AIC_1_2) {
        return {numBlocks, numBlocks << 1};
    } else {
        // 需要加校验-报错
        SK_LOGE("[lock detector] Unsupported kernel type to compute core num: %u", kernelType);
    }
}

bool LockDetector::IsInSKStream(const SuperKernelBaseNode& node) {
    return std::find(skStreamIds.begin(), skStreamIds.end(), node.GetStreamIdxInGraph()) != skStreamIds.end();
}

bool LockDetector::HasDeadlock(SuperKernelBaseNode* curNode, SuperKernelGraph& graph) {
    if (curNode->GetPreNodeId() == INVALID_TASK_ID) {
        return false;
    }
    
    uint64_t preNodeId = curNode->GetPreNodeId();
    SuperKernelBaseNode* preNode = graph.GetNodeById(preNodeId); // 注意取的是在一条流上
    

    if (preNode->IsVisited()) {
        return false;
    }
    curNode->SetVisited(true);
    nodes.emplace_back(preNode->GetNodeId());
    if (preNode->GetNodeType() == SkNodeType::NODE_KERNEL) {
        if (HasEnoughCores(preNode, false)) {
            if (HasDeadlock(preNode, graph)) {
                SK_LOGI("Deadlock detected in kernel node, nodeId=%lu", preNode->GetNodeId());
                return true;
            }
        } else {
            SK_LOGI("Not enough cores for kernel node, nodeId=%lu, requiredCube=%u, requiredVec=%u",
                     preNode->GetNodeId(), GetNodeCoreNum(*preNode).first, GetNodeCoreNum(*preNode).second);
            return true;
        }
    } else if (preNode->GetNodeType() == SkNodeType::NODE_WAIT) {
        // 1. 获取notify
        uint64_t notifyId = preNode->GetNotifyNodeId();
        SuperKernelBaseNode* notifyNode = graph.GetNodeById(notifyId);
        if (HasDeadlock(preNode, graph)) {
            SK_LOGI("Deadlock detected in wait node pre-path, waitNodeId=%lu", preNode->GetNodeId());
            return true;
        }
        // 2. 是否在其他流上
        if (notifyNode->GetStreamIdxInGraph() != preNode->GetStreamIdxInGraph() && !IsInSKStream(*notifyNode)) {
            if (HasDeadlock(notifyNode, graph)) {
                SK_LOGI("Deadlock detected in wait node cross-stream path, waitNodeId=%lu, notifyNodeId=%lu",
                         preNode->GetNodeId(), notifyNode->GetNodeId());
                return true;
            }
        }
    } else if (preNode->GetNodeType() == SkNodeType::NODE_NOTIFY) {
        // 1. 获取wait
        uint64_t waitId = preNode->GetWaitNodeId();
        SuperKernelBaseNode* waitNode = graph.GetNodeById(waitId);
        if (!IsInSKStream(*waitNode)) {
            if (HasDeadlock(preNode, graph)) {
                SK_LOGI("Deadlock detected in notify node path, notifyNodeId=%lu", preNode->GetNodeId());
                return true;
            }
        }
    }

    return false;
}

bool LockDetector::HasEnoughCores(const SuperKernelBaseNode* curNode, bool isSuperKernel) {
    std::pair<uint32_t, uint32_t> coreNum = GetNodeCoreNum(*curNode);
    uint32_t curNodeCubeNum = coreNum.first;
    uint32_t curNodeVecNum = coreNum.second;

    if (isSuperKernel) {
        if(curNodeCubeNum <= superKernelCubeNum && curNodeVecNum <= superKernelVecNum){
            return true;
        }
    } else {
        if(curNodeCubeNum <= depOpCubeNum && curNodeVecNum <= depOpVecNum){
            return true;
        }
    }

    std::pair<uint32_t, uint32_t> availableCores = GetAvailableCores(isSuperKernel);
    if (isSuperKernel) {
        if (curNodeCubeNum <= availableCores.first && curNodeVecNum <= availableCores.second) {
            superKernelCubeNum = std::max(superKernelCubeNum, curNodeCubeNum);
            superKernelVecNum = std::max(superKernelVecNum, curNodeVecNum);
            return true;
        } else {
            return false;
        }
    } else {
        if (curNodeCubeNum <= availableCores.first && curNodeVecNum <= availableCores.second) {
            depOpCubeNum = std::max(depOpCubeNum, curNodeCubeNum);
            depOpVecNum = std::max(depOpVecNum, curNodeVecNum);
            return true;
        } else {
            return false;
        }
    }
}

void LockDetector::Reset(SuperKernelGraph& graph) {
    depOpCubeNum = 0;
    depOpVecNum = 0;
    superKernelCubeNum = 0;
    superKernelVecNum = 0;
    isExistWaitFlag = false;
    for (auto nodeId : nodes) {
        SuperKernelBaseNode* node = graph.GetNodeById(nodeId);
        node -> SetVisited(false);
    }
    nodes.clear();
    skStreamIds.clear();
}

bool LockDetector::IsFusible(const SuperKernelBaseNode& curNode, SuperKernelGraph& graph) {
    skStreamIds.insert(curNode.GetStreamIdxInGraph());
    if (curNode.GetNodeType() == SkNodeType::NODE_NOTIFY) {
        return true;
    } else if (curNode.GetNodeType() == SkNodeType::NODE_WAIT) {
        uint64_t notifyId = curNode.GetNotifyNodeId();
        SuperKernelBaseNode* notifyNode = graph.GetNodeById(notifyId);
        if (!isExistWaitFlag){
            isExistWaitFlag = true;
        }
        return !HasDeadlock(notifyNode, graph);
    } else if (curNode.GetNodeType() == SkNodeType::NODE_KERNEL) {
        // done 在wait后的kernel策略有改变
        // way1：在wait后，如果有足够的core，则可以fuse，否则不fuse
        // way2：在第一个wait后，不允许后续融合的kernel 核数大于当前sk的max core num
        std::pair<uint32_t, uint32_t> coreNum = GetNodeCoreNum(curNode);
        if (isExistWaitFlag){
            return coreNum.first <= superKernelCubeNum && coreNum.second <= superKernelVecNum;
        }
        if (HasEnoughCores(&curNode, true)) {
            superKernelCubeNum = std::max(superKernelCubeNum, coreNum.first);
            superKernelVecNum = std::max(superKernelVecNum, coreNum.second);
            return true;
        } else {
            return false;
        }
    } else {
        SK_LOGE("no support detector taskType!!! taskType=%u", curNode.GetNodeType());
        return false;
    }
}