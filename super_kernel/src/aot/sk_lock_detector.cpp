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

int64_t LockDetector::deviceRealCubeNum = 0;
int64_t LockDetector::deviceRealVecNum = 0;

void LockDetector::Init() {
    SK_LOGD("[lock detector] LockDetector::Init: Initializing lock detector");
    nodes.clear();
    depOpCubeNum = 0; // visited op cube num outside superkernel
    depOpVecNum = 0; // visited op vec num outside superkernel
    superKernelCubeNum = 0; // fused op cube num in superkernel
    superKernelVecNum = 0; // fused op vec num in superkernel
    nodeNum = 0; // current node num
    kernelNodeNum = 0; // kernel node num in scope
    skStreamIds.clear();
    isExistWaitFlag = false;

    // Initialize device core numbers if not already set
    if (deviceRealCubeNum == 0 && deviceRealVecNum == 0) {
        GetDeviceCores();
    } else {
        SK_LOGD("[lock detector] LockDetector::Init: Using cached device cores: cube=%lu, vec=%lu",
                deviceRealCubeNum, deviceRealVecNum);
    }
}

aclError LockDetector::GetDeviceCores() {
    // 获取deviceId
    int32_t deviceId;
    aclError ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("[lock detector] GetDeviceCores for deviceId failed, ret=%d", ret);
        return ret;
    }
    // 获取CubeNum、VecNum
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &LockDetector::deviceRealCubeNum);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("[lock detector] GetDeviceCores for cube num failed, ret=%d", ret);
        return ret;
    }
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &LockDetector::deviceRealVecNum);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("[lock detector] GetDeviceCores for vec num failed, ret=%d", ret);
        return ret;
    }
    SK_LOGI("[lock detector] GetDeviceCores success, cube num=%u, vec num=%u ", deviceRealCubeNum, deviceRealVecNum);
    return ret;
}

std::pair<uint64_t, uint64_t> LockDetector::GetAvailableCores(bool isSuperKernel) const {
    if (isSuperKernel) {
        return {deviceRealCubeNum - depOpCubeNum, deviceRealVecNum - depOpVecNum};
    } else {
        return {deviceRealCubeNum - superKernelCubeNum, deviceRealVecNum - superKernelVecNum};
    }
}

bool LockDetector::IsInSKStream(const SuperKernelBaseNode& node) {
    return std::find(skStreamIds.begin(), skStreamIds.end(), node.GetStreamIdxInGraph()) != skStreamIds.end();
}

bool LockDetector::HasDeadlock(SuperKernelBaseNode* curNode, SuperKernelGraph& graph) {
    curNode->SetVisited(true);
    nodes.emplace_back(curNode->GetNodeId());
    if (curNode->GetPreNodeId() == INVALID_TASK_ID) {
        return false;
    }
    
    uint64_t preNodeId = curNode->GetPreNodeId();
    SuperKernelBaseNode* preNode = graph.GetNodeById(preNodeId); // 注意取的是在一条流上
    

    if (preNode->IsVisited()) {
        return false;
    }
    if (preNode->GetNodeType() == SkNodeType::NODE_KERNEL) {
        if (HasEnoughCores(preNode, false)) {
            if (HasDeadlock(preNode, graph)) {
                SK_LOGI("Deadlock detected in kernel node, nodeId=%lu", preNode->GetNodeId());
                return true;
            }
        } else {
            SK_LOGI("Not enough cores for kernel node, nodeId=%lu, requiredCube=%u, requiredVec=%u",
                     preNode->GetNodeId(), preNode->GetCubeNum(), preNode->GetVecNum());
            return true;
        }
    } else if (preNode->GetNodeType() == SkNodeType::NODE_WAIT) {
        // 1. 获取notify
        uint64_t notifyId = preNode->GetCorrespondingNotifyNodeId();
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
        uint32_t cubeNum = curNode->GetCubeNum();
        uint32_t vecNum = curNode->GetVecNum();
        if (cubeNum > 0 || vecNum > 0) {
            if (!HasEnoughCores(preNode, false)) {
                SK_LOGI("Not enough cores for notify node, nodeId=%lu, requiredCube=%u, requiredVec=%u",
                         preNode->GetNodeId(), cubeNum, vecNum);
                return true;
            }
        }
        // 1. 获取所有wait节点（一对多关系）
        std::vector<uint64_t> waitIds = preNode->GetCorrespondingWaitNodeIds();
        for (uint64_t waitId : waitIds) {
            SuperKernelBaseNode* waitNode = graph.GetNodeById(waitId);
            if (IsAfterSKRange(*waitNode)){
                return true;
            }else if (!IsBeforeSKRange(*waitNode)) {
                if (HasDeadlock(preNode, graph)) {
                    SK_LOGI("Deadlock detected in notify node path, notifyNodeId=%lu", preNode->GetNodeId());
                    return true;
                }
            }
        }
    }

    return false;
}

bool LockDetector::HasEnoughCores(const SuperKernelBaseNode* curNode, bool isSuperKernel) {
    uint32_t curNodeCubeNum = curNode->GetCubeNum();
    uint32_t curNodeVecNum = curNode->GetVecNum();

    if (isSuperKernel) {
        if(curNodeCubeNum <= superKernelCubeNum && curNodeVecNum <= superKernelVecNum){
            SK_LOGD("[lock detector] Node %s: within current SK limits (cube %u<=%u, vec %u<=%u)",
                    curNode->FormatNodeInfo().c_str(), curNodeCubeNum, superKernelCubeNum,
                    curNodeVecNum, superKernelVecNum);
            return true;
        }
    } else {
        if(curNodeCubeNum <= depOpCubeNum && curNodeVecNum <= depOpVecNum){
            SK_LOGD("[lock detector] Node %s: within current depOp limits (cube %u<=%u, vec %u<=%u)",
                    curNode->FormatNodeInfo().c_str(), curNodeCubeNum, depOpCubeNum,
                    curNodeVecNum, depOpVecNum);
            return true;
        }
    }

    std::pair<uint32_t, uint32_t> availableCores = GetAvailableCores(isSuperKernel);
    if (isSuperKernel) {
        if (curNodeCubeNum <= availableCores.first && curNodeVecNum <= availableCores.second) {
            superKernelCubeNum = std::max(superKernelCubeNum, curNodeCubeNum);
            superKernelVecNum = std::max(superKernelVecNum, curNodeVecNum);
            SK_LOGD("[lock detector] Node %s: allocated from device (cube %u, vec %u), new SK limits: cube %u, vec %u",
                    curNode->FormatNodeInfo().c_str(), curNodeCubeNum, curNodeVecNum,
                    superKernelCubeNum, superKernelVecNum);
            return true;
        } else {
            SK_LOGD("[lock detector] Node %s: insufficient cores for SK (required: cube %u, vec %u, available: cube %u, vec %u)",
                    curNode->FormatNodeInfo().c_str(), curNodeCubeNum, curNodeVecNum,
                    availableCores.first, availableCores.second);
            return false;
        }
    } else {
        if (curNodeCubeNum <= availableCores.first && curNodeVecNum <= availableCores.second) {
            depOpCubeNum = std::max(depOpCubeNum, curNodeCubeNum);
            depOpVecNum = std::max(depOpVecNum, curNodeVecNum);
            SK_LOGD("[lock detector] Node %s: allocated from device (cube %u, vec %u), new depOp limits: cube %u, vec %u",
                    curNode->FormatNodeInfo().c_str(), curNodeCubeNum, curNodeVecNum,
                    depOpCubeNum, depOpVecNum);
            return true;
        } else {
            SK_LOGD("[lock detector] Node %s: insufficient cores for depOp (required: cube %u, vec %u, available: cube %u, vec %u)",
                    curNode->FormatNodeInfo().c_str(), curNodeCubeNum, curNodeVecNum,
                    availableCores.first, availableCores.second);
            return false;
        }
    }
}

void LockDetector::Reset(SuperKernelGraph& graph) {
    SK_LOGD("[lock detector] LockDetector::Reset: Resetting lock detector state");
    SK_LOGD("[lock detector] Previous state: depOpCubeNum=%u, depOpVecNum=%u, superKernelCubeNum=%u, superKernelVecNum=%u, nodeNum=%u, kernelNodeNum=%u, isExistWaitFlag=%d",
            depOpCubeNum, depOpVecNum, superKernelCubeNum, superKernelVecNum, nodeNum, kernelNodeNum, isExistWaitFlag);

    depOpCubeNum = 0;
    depOpVecNum = 0;
    superKernelCubeNum = 0;
    superKernelVecNum = 0;
    nodeNum = 0;
    kernelNodeNum = 0;
    isExistWaitFlag = false;

    uint32_t unvisitedCount = 0;
    for (auto nodeId : nodes) {
        SuperKernelBaseNode* node = graph.GetNodeById(nodeId);
        if (node != nullptr) {
            node->SetVisited(false);
            unvisitedCount++;
        }
    }
    SK_LOGD("[lock detector] Reset: Cleared visited flag for %u nodes", unvisitedCount);

    nodes.clear();
    skStreamIds.clear();
    skRangeInStream.clear();
    SK_LOGD("[lock detector] Reset: Completed, cleared %zu nodes, %zu stream IDs, %zu stream ranges",
            nodes.size(), skStreamIds.size(), skRangeInStream.size());
}

void LockDetector::UpdateSKRangeInStream(const SuperKernelBaseNode& curNode) {
    uint64_t nodeId = curNode.GetNodeIdxInStream();
    uint32_t streamId = curNode.GetStreamIdxInGraph();
    skRangeInStream[streamId].first = std::min(skRangeInStream[streamId].first, nodeId);
    skRangeInStream[streamId].second = std::max(skRangeInStream[streamId].second, nodeId);
}

bool LockDetector::IsBeforeSKRange(const SuperKernelBaseNode& curNode) {
    uint64_t nodeId = curNode.GetNodeIdxInStream();
    uint32_t streamId = curNode.GetStreamIdxInGraph();
    if (skRangeInStream.find(streamId) == skRangeInStream.end()) {
        return false;
    }
    return nodeId < skRangeInStream[streamId].first;
}

bool LockDetector::IsAfterSKRange(const SuperKernelBaseNode& curNode) {
    uint64_t nodeId = curNode.GetNodeIdxInStream();
    uint32_t streamId = curNode.GetStreamIdxInGraph();
    if (skRangeInStream.find(streamId) == skRangeInStream.end()) {
        return false;
    }
    return nodeId > skRangeInStream[streamId].second;
}

bool LockDetector::IsFusible(SuperKernelBaseNode& curNode, SuperKernelGraph& graph) {
    // If node already visited (already checked or fused), return true directly without modifying state
    if (curNode.IsVisited()) {
        SK_LOGD("[lock detector] Node %s: already visited, can fuse", curNode.FormatNodeInfo().c_str());
        return true;
    }

    SK_LOGI("[lock detector] IsFusible: Checking node %s, current state: nodeNum=%u, kernelNodeNum=%u, isExistWaitFlag=%d",
            curNode.FormatNodeInfo().c_str(), nodeNum, kernelNodeNum, isExistWaitFlag);

    // First check if node can be fused, without modifying state
    bool canFuse = false;
    bool shouldSetWaitFlag = false;

    if (curNode.GetNodeType() == SkNodeType::NODE_NOTIFY) {
        uint32_t cubeNum = curNode.GetCubeNum();
        uint32_t vecNum = curNode.GetVecNum();
        if (cubeNum == 0 && vecNum == 0) {
            canFuse = true;
            SK_LOGD("[lock detector] Notify node %s: not needed core resource, can fuse", curNode.FormatNodeInfo().c_str());
        } else {
            SK_LOGE("[lock detector] Notify node %s: in SK range with coreNum>0 (cube %u, vec %u), which not allowed",
                    curNode.FormatNodeInfo().c_str(), cubeNum, vecNum);
        }
    } else if (curNode.GetNodeType() == SkNodeType::NODE_WAIT) {
        uint64_t notifyId = curNode.GetCorrespondingNotifyNodeId();
        SuperKernelBaseNode* notifyNode = graph.GetNodeById(notifyId);
        if (notifyNode == nullptr) {
            SK_LOGE("[lock detector] Wait node %s: notify node %lu not found", curNode.FormatNodeInfo().c_str(), notifyId);
            canFuse = false;
        } else if (notifyNode->GetCubeNum() > 0 || notifyNode->GetVecNum() > 0) {
            if (HasEnoughCores(notifyNode, false)){
                SK_LOGD("[lock detector] Wait node %s: notify %s has enough cores, can fuse",
                        curNode.FormatNodeInfo().c_str(), notifyNode->FormatNodeInfo().c_str());
                canFuse = true;
            } else {
                SK_LOGD("[lock detector] Wait node %s: notify %s has not enough cores, cannot fuse",
                        curNode.FormatNodeInfo().c_str(), notifyNode->FormatNodeInfo().c_str());
                canFuse = false;
            }
        } else {
            if (IsAfterSKRange(*notifyNode)) {
                SK_LOGD("[lock detector] Wait node %s: notify %s is after SK range, cannot fuse",
                        curNode.FormatNodeInfo().c_str(), notifyNode->FormatNodeInfo().c_str());
                canFuse = false;
            } else if (IsBeforeSKRange(*notifyNode)) {
                SK_LOGD("[lock detector] Wait node %s: notify %s is before SK range, can fuse",
                        curNode.FormatNodeInfo().c_str(), notifyNode->FormatNodeInfo().c_str());
                canFuse = true;
            } else if (nodeNum == 0) {
                SK_LOGD("[lock detector] Wait node %s: first node in scope, can fuse", curNode.FormatNodeInfo().c_str());
                canFuse = true;
            } else {
                SK_LOGD("[lock detector] Wait node %s: checking shouldSetWaitFlag, nodeNum=%u, kernelNodeNum=%u, isExistWaitFlag=%d",
                        curNode.FormatNodeInfo().c_str(), nodeNum, kernelNodeNum, isExistWaitFlag);
                if (!isExistWaitFlag && nodeNum > 0 && kernelNodeNum > 0) {
                    shouldSetWaitFlag = true;
                    SK_LOGD("[lock detector] Wait node %s: will set isExistWaitFlag=true", curNode.FormatNodeInfo().c_str());
                }
                if (notifyNode->IsVisited()) {
                    SK_LOGD("[lock detector] Wait node %s: notify %s already visited, can fuse",
                            curNode.FormatNodeInfo().c_str(), notifyNode->FormatNodeInfo().c_str());
                    canFuse = true;
                } else {
                    bool hasDeadlock = HasDeadlock(notifyNode, graph);
                    SK_LOGD("[lock detector] Wait node %s: notify %s not visited, HasDeadlock=%d",
                            curNode.FormatNodeInfo().c_str(), notifyNode->FormatNodeInfo().c_str(), hasDeadlock);
                    canFuse = !hasDeadlock;
                }
            }
        }
    } else if (curNode.GetNodeType() == SkNodeType::NODE_KERNEL) {
        uint32_t cubeNum = curNode.GetCubeNum();
        uint32_t vecNum = curNode.GetVecNum();
        SK_LOGD("[lock detector] Kernel node %s: coreNum={%u, %u}, isExistWaitFlag=%d, superKernelCubeNum=%u, superKernelVecNum=%u",
                curNode.FormatNodeInfo().c_str(), cubeNum, vecNum, isExistWaitFlag, superKernelCubeNum, superKernelVecNum);
        if (isExistWaitFlag) {
            canFuse = (cubeNum <= superKernelCubeNum && vecNum <= superKernelVecNum);
            SK_LOGD("[lock detector] Kernel node %s: isExistWaitFlag=true, canFuse=%d (cube %u<=%u, vec %u<=%u)",
                    curNode.FormatNodeInfo().c_str(), canFuse, cubeNum, superKernelCubeNum, vecNum, superKernelVecNum);
        } else {
            canFuse = HasEnoughCores(&curNode, true);
            SK_LOGD("[lock detector] Kernel node %s: HasEnoughCores=%d", curNode.FormatNodeInfo().c_str(), canFuse);
        }
    } else {
        SK_LOGW("[lock detector] Node %s: unsupported taskType=%u", curNode.FormatNodeInfo().c_str(), curNode.GetNodeType());
        canFuse = false;
    }

    // Only modify state if node can be fused
    if (canFuse) {
        skStreamIds.insert(curNode.GetStreamIdxInGraph());
        UpdateSKRangeInStream(curNode);
        nodeNum++;
        curNode.SetVisited(true);
        nodes.emplace_back(curNode.GetNodeId());

        if (curNode.GetNodeType() == SkNodeType::NODE_WAIT) {
            if (shouldSetWaitFlag) {
                isExistWaitFlag = true;
                SK_LOGD("[lock detector] Wait node %s: set isExistWaitFlag=true", curNode.FormatNodeInfo().c_str());
            }
            SK_LOGD("[lock detector] Wait node %s: after fusion, nodeNum=%u, kernelNodeNum=%u, isExistWaitFlag=%d",
                    curNode.FormatNodeInfo().c_str(), nodeNum, kernelNodeNum, isExistWaitFlag);
        } else if (curNode.GetNodeType() == SkNodeType::NODE_KERNEL) {
            // HasEnoughCores already updated superKernelCubeNum/superKernelVecNum when isExistWaitFlag=false
            kernelNodeNum++;
            SK_LOGD("[lock detector] Kernel node %s: fused, kernelNodeNum now %u", curNode.FormatNodeInfo().c_str(), kernelNodeNum);
        }
    } else {
        SK_LOGI("[lock detector] Node %s: cannot be fused", curNode.FormatNodeInfo().c_str());
    }

    return canFuse;
}
