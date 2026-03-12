/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef __SK_TASK_BUILDER_H__
#define __SK_TASK_BUILDER_H__

#include "sk_node.h"
#include "sk_types.h"
#include "sk_options_manager.h"

#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

// 前向声明
class SuperKernelGraph;

// ========== 队列类型枚举（表示任务运行在哪个队列） ==========

enum class SkQueueType : uint8_t {
    AIC,     // 仅 AIC 队列
    AIV,     // 仅 AIV 队列
    MIX_1_1, // 同时在两个队列 (MIX_AIC_1_1)
    MIX_1_2, // 同时在两个队列 (MIX_AIC_1_2)
    UNKNOWN, // 未知/无效类型（用于调试同步等特殊场景）
};

inline const char* to_string(SkQueueType type) {
    switch (type) {
        case SkQueueType::AIC: return "AIC";
        case SkQueueType::AIV: return "AIV";
        case SkQueueType::MIX_1_1: return "MIX_1_1";
        case SkQueueType::MIX_1_2: return "MIX_1_2";
        case SkQueueType::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

// ========== 新的同步数据结构（节点导向，对齐Python实现） ==========

// 同步方向类型（对应Python的 "cub:vec", "vec:cub" 等）
enum class SyncDirection : uint8_t {
    NONE = 0,   // 无同步
    CUB_TO_CUB, // AIC -> AIC
    VEC_TO_VEC, // AIV -> AIV
    CUB_TO_VEC, // AIC -> AIV
    VEC_TO_CUB, // AIV -> AIC
    MIX_TO_MIX, // 双向同步（MIX -> MIX）
    ALL_SYNC,      // 全核同步
};

inline const char *to_string(SyncDirection dir) {
    switch (dir) {
    case SyncDirection::NONE:
        return "NONE";
    case SyncDirection::CUB_TO_CUB:
        return "CUB_TO_CUB";
    case SyncDirection::VEC_TO_VEC:
        return "VEC_TO_VEC";
    case SyncDirection::CUB_TO_VEC:
        return "CUB_TO_VEC";
    case SyncDirection::VEC_TO_CUB:
        return "VEC_TO_CUB";
    case SyncDirection::MIX_TO_MIX:
        return "MIX_TO_MIX";
    case SyncDirection::ALL_SYNC:
        return "ALL_SYNC";
    default:
        return "UNKNOWN";
    }
}

// 每个任务的同步信息
struct TaskSyncInfo {
    SkQueueType queueType;  // 任务运行在哪个队列：AIC/AIV/MIX
    
    // AIC队列（cub_op_list）的同步信息
    std::map<size_t, SyncDirection> cubSendInfo;  // 发送同步给哪些任务
    std::map<size_t, SyncDirection> cubRecvInfo;  // 从哪些任务接收同步
    
    // AIV队列（vec_op_list）的同步信息
    std::map<size_t, SyncDirection> vecSendInfo;  // 发送同步给哪些任务
    std::map<size_t, SyncDirection> vecRecvInfo;  // 从哪些任务接收同步

    // 核间同步方向: 0=CUBE(CUB_TO_CUB), 1=VEC(VEC_TO_VEC)
    std::map<size_t, SyncDirection> crossSyncInfo;
    
    TaskSyncInfo() : queueType(SkQueueType::UNKNOWN) {}
};

class SkTaskBuilder {
public:
    SkTaskBuilder(SuperKernelOptionsManager &opts, const SuperKernelGraph &graph)
        : opts(opts), graph_(graph) {}

    SkLaunchInfo Build(const std::vector<SuperKernelBaseNode *> &tasks,
                       const std::vector<SuperKernelBaseNode *> &customTasks);

private:
    SuperKernelOptionsManager &opts;
    const SuperKernelGraph &graph_;  // Graph对象引用

    // 新的同步信息存储：每个任务维护自己的send/recv信息
    std::vector<TaskSyncInfo> taskSyncInfos_;

    // 任务添加函数 - 按类型分离
    std::pair<int, int> GetPreFetchCnt(const ResolvedFunctionInfo &resolved);
    void AddSyncTask(SkTask &skTask, size_t nodeIndex, SkCoreSyncType syncType);
    void AddEventTask(SkTask &skTask, SuperKernelBaseNode *node, size_t nodeIndex, SkTaskType taskType);
    void AddFuncTask(SkTask &skTask, SuperKernelBaseNode *node, SkDfxInfo *dfxInfo,
                       size_t nodeIndex, int addrIndex, int binCount,
                       SkTaskType taskType, uint32_t numBlocks);

    void DispatchFuncTask(SkTask &skTaskCube, SkTask &skTaskVec,
                         SuperKernelBaseNode *node, SkDfxInfo *dfxInfo,
                         size_t nodeIndex, int binCount, SkTaskType taskType,
                         SkQueueType queueType);
    void DispatchEventTask(SkTask &skTaskCube, SkTask &skTaskVec,
                           SuperKernelBaseNode *node,
                           size_t nodeIndex, SkTaskType taskType,
                           SkQueueType queueType);

    void DispatchSyncTasks(SkTask &skTaskCube, SkTask &skTaskVec, size_t nodeIndex,
                           const std::map<size_t, SyncDirection> &syncInfo,
                           bool isSend, SkQueueType queueType);

    // ========== 新增：基于Graph拓扑的Sync提取方法 ==========

    // 初始化taskSyncInfos_
    void InitTaskSyncInfos(const std::vector<SuperKernelBaseNode *> &tasks);

    // 预计算sync关系（基于Graph拓扑）
    void PrecomputeSyncRelationsFromGraph(const std::vector<SuperKernelBaseNode *> &tasks);

    // 提取同流内的sync关系（基于GetNextNodeId）
    void ExtractIntraStreamSync(const std::vector<SuperKernelBaseNode *> &tasks);

    // 提取跨流的sync关系（基于eventToNodes）
    void ExtractInterStreamSync(const std::vector<SuperKernelBaseNode *> &tasks);

    // ========== 核心同步插入方法（对齐Python insert_sync_event） ==========
    
    // 插入同步事件：preOp -> currOp
    void InsertSyncEvent(size_t preIdx, size_t currIdx);

    // ========== 优化方法 ==========
    void OptimizeSyncRelations();
    void RemoveCrossedLineSync();
    void RemoveMultiSendSync();
    void RemoveMultiRecvSync();
    
    // 辅助方法：判断是否可以移除交叉同步
    bool JudgeRemoveCrossSync(size_t sendIdx, size_t recvIdx, bool isCubToVec);
    
    // 辅助方法：移除同步信息
    void RemoveSyncInfo(size_t sendIdx, size_t recvIdx, bool isRemoveRecv, SyncDirection dirToRemove);
    
    // 打印同步信息（调试用）
    void PrintSyncInfo(const char* stage);

    SkHostEntryInfo GenEntryInfo(SkTask &skTaskCube, SkTask &skTaskVec);
    DeviceArgsPtr GenEntryArgs(const SkTask &skTaskCube,
                                              const SkTask &skTaskVec,
                                              const SkDfxInfo *dfxInfos,
                                              uint32_t dfxCount);
};

#endif // __SK_TASK_BUILDER_H__
