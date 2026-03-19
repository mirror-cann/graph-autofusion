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
    AIC = 0b00,     // 仅 AIC 队列
    AIV = 0b01,     // 仅 AIV 队列
    MIX = 0b10,     // 同时在两个队列
    UNKNOWN = 0xFF, // 未知/无效类型
};

inline const char* to_string(SkQueueType type) {
    switch (type) {
        case SkQueueType::AIC: return "AIC";
        case SkQueueType::AIV: return "AIV";
        case SkQueueType::MIX: return "MIX";
        case SkQueueType::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

// ========== 新的同步数据结构（节点导向，对齐Python实现） ==========

// 同步方向类型（对应Python的 "cub:vec", "vec:cub" 等）
enum class SyncDirection : uint8_t {
    NONE = 0,      // 无同步
    CUB_TO_CUB,      // AIC -> AIC
    VEC_TO_VEC,      // AIV -> AIV
    CUB_TO_VEC,      // AIC -> AIV
    VEC_TO_CUB,      // AIV -> AIC
    BOTH,            // 双向同步（MIX -> MIX）
    DEBUG,           // 
};

inline const char* to_string(SyncDirection dir) {
    switch (dir) {
        case SyncDirection::NONE:      return "NONE";
        case SyncDirection::CUB_TO_CUB: return "CUB_TO_CUB";
        case SyncDirection::VEC_TO_VEC: return "VEC_TO_VEC";
        case SyncDirection::CUB_TO_VEC: return "CUB_TO_VEC";
        case SyncDirection::VEC_TO_CUB: return "VEC_TO_CUB";
        case SyncDirection::BOTH:       return "BOTH";
        case SyncDirection::DEBUG:      return "DEBUG";
        default: return "UNKNOWN";
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

    // 核间同步方向: 0=CUBE(CUB_TO_CUB), 1=VEC(VEC_TO_VEC), 默认NONE
    std::map<size_t, SyncDirection> crossSyncInfo;
    
    TaskSyncInfo() : queueType(SkQueueType::UNKNOWN) {}
};

class SkTaskBuilder {
public:
    SkTaskBuilder(SuperKernelOptionsManager &opts, const SuperKernelGraph &graph)
        : opts(opts), graph_(graph) {}

    SkLaunchInfo Build(const std::vector<SuperKernelBaseNode *> &tasks);

private:
    SuperKernelOptionsManager &opts;
    const SuperKernelGraph &graph_;  // Graph对象引用

    // 新的同步信息存储：每个任务维护自己的send/recv信息
    std::vector<TaskSyncInfo> taskSyncInfos_;
    std::pair<int, int> GetPreFetchCnt(const ResolvedFunctionInfo &resolved);
    void AddTask(SkTask &skTask, SkDfxInfo *dfxInfos, const std::vector<SuperKernelBaseNode *> &tasks, size_t index,
                SkKernelType originType, int customArg, int binCount, SkTaskType taskType,
                uint32_t syncFlag);

    void DispatchTask(SkTask &skTaskCube, SkTask &skTaskVec, SkDfxInfo *dfxInfos,
        const std::vector<SuperKernelBaseNode *> &tasks,
        size_t index, int binCount, SkTaskType taskType);
    void DispatchTaskSync(SkTask &skTaskCube, SkTask &skTaskVec, SkDfxInfo *dfxInfos,
        const std::vector<SuperKernelBaseNode *> &tasks, size_t index, int binCount,
        SkTaskType taskType, SkCoreSyncType syncFlag, bool addToAicQue, bool addToAivQue,
        SkQueueType prevType, SkQueueType nextType, const char *detail);
    
    // 批量处理同步任务（简化Build循环）
    void DispatchSyncTasks(SkTask &skTaskCube, SkTask &skTaskVec, SkDfxInfo *dfxInfos,
                           const std::vector<SuperKernelBaseNode *> &tasks,
                           const std::map<size_t, SyncDirection> &syncInfo,
                           size_t myIdx, int binCount, bool isSend);

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
    
    // 打印同步信息
    void PrintSyncInfo(const char* stage);

    SkHostEntryInfo GenEntryInfo(SkTask &skTaskCube, SkTask &skTaskVec);
    DeviceArgsPtr GenEntryArgs(const SkTask &skTaskCube,
                                              const SkTask &skTaskVec,
                                              const SkDfxInfo *dfxInfos,
                                              uint32_t dfxCount);
};

#endif // __SK_TASK_BUILDER_H__
