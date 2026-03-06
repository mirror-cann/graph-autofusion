/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_task_builder.h"
#include "sk_graph.h"
#include "sk_log.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include "securec.h"

static bool UsesAic(SkQueueType type)
{
    return type == SkQueueType::AIC || type == SkQueueType::MIX;
}

static bool UsesAiv(SkQueueType type)
{
    return type == SkQueueType::AIV || type == SkQueueType::MIX;
}

static const KernelInfos &GetKernelInfos(const SuperKernelBaseNode *node)
{
    return node->GetNodeInfos().kernelInfos;
}

// dump device entry args
static void DumpTaskQue(const TaskQue *que, const char *name)
{
    SK_LOGI("%s TaskQue: cap=%u, tasks=%u", name, que->cap, que->taskCnt);
    for (uint32_t i = 0; i < que->taskCnt; ++i)
    {
        const TaskInfo &ti = que->taskInfos[i];
        SK_LOGI("[%u] type=%s, idx=%u, blk=%u, kernel=%s, entries=%u, args=0x%llx",
               i, to_string(ti.type), ti.index, ti.numBlocks,
               to_string(ti.originType), ti.entryCnt,
               (unsigned long long)ti.args);
        for (uint32_t j = 0; j < ti.entryCnt; ++j)
        {
            SK_LOGI("      entry[%u]=0x%llx", j, (unsigned long long)ti.entry[j]);
        }
    }
}

static void DumpDeviceArgs(const SkDeviceEntryArgs *args)
{
    const uint8_t *base = (const uint8_t *)args;

    const SkHeaderInfo &hdr = args->skHeader;
    SK_LOGI("SkHeaderInfo: aicOff=%u, aivOff=%u, counterOff=%u, wsOff=%u, dfxOff=%u, nodeCnt=%u, totalSize=%lu",
           hdr.aicQueOffset, hdr.aivQueOffset, hdr.counterOffset,
           hdr.wsOffset, hdr.dfxOffset, hdr.nodeCnt, hdr.totalSize);

    DumpTaskQue((const TaskQue *)(base + args->skHeader.aicQueOffset), "AIC");
    DumpTaskQue((const TaskQue *)(base + args->skHeader.aivQueOffset), "AIV");

    const SkDfxInfo *dfx = (const SkDfxInfo *)(base + args->skHeader.dfxOffset);
    for (uint32_t i = 0; i < args->skHeader.nodeCnt; ++i)
    {
        SK_LOGI("dfx[%u]: bin=0x%llx, func=0x%llx, ori=0x%llx",
               i, (unsigned long long)dfx[i].binHdl,
               (unsigned long long)dfx[i].funcHdl,
               (unsigned long long)dfx[i].funcHdlOri);
    }
}

// ========== 新增：SkTaskBuilder静态方法实现 ==========

static const char *SyncDirectionName(SyncDirection dir)
{
    switch (dir)
    {
    case SyncDirection::CUB_TO_VEC:
        return "cub:vec";
    case SyncDirection::VEC_TO_CUB:
        return "vec:cub";
    case SyncDirection::BOTH:
        return "cub:vec;vec:cub";
    default:
        return "unknown";
    }
}

// 查找方向枚举
enum class SearchDirection : uint8_t
{
    PREV,
    NEXT
};
static const char *to_string(SearchDirection dir)
{
    return (dir == SearchDirection::PREV) ? "PREV" : "NEXT";
}

// 查找指定方向的 KERNEL 节点，未找到则抛出异常
static SuperKernelBaseNode *FindKernelNodeInDirection(
    size_t nodeIndex,
    SkNodeType nodeType,
    const SuperKernelBaseNode *startNode,
    const SuperKernelGraph &graph,
    SearchDirection direction,
    int maxHops = 5)
{
    const SuperKernelBaseNode *current = startNode;

    for (int hop = 0; hop < maxHops; ++hop)
    {
        uint32_t nextId = (direction == SearchDirection::PREV)
                              ? current->GetPreNodeId()
                              : current->GetNextNodeId();

        if (nextId == INVALID_TASK_ID)
        {
            SK_LOGE("%s node %zu has no %s-node.", to_string(nodeType), nodeIndex, to_string(direction));
            throw std::runtime_error("[sk error] FindKernelNodeInDirection failed.");
        }
        current = graph.GetNodeById(nextId);
        if (current->GetNodeType() == SkNodeType::NODE_KERNEL)
        {
            return const_cast<SuperKernelBaseNode *>(current);
        }
    }

    SK_LOGE("%s node %zu cannot find KERNEL within %d hops.", to_string(nodeType), nodeIndex, maxHops);
    throw std::runtime_error("[sk error] FindKernelNodeInDirection failed.");
}

static SyncDirection GenSyncDirection(SkQueueType preType, SkQueueType currType)
{
    // 对应Python的gen_sync_name函数
    switch (preType)
    {
    case SkQueueType::MIX:
        if (currType == SkQueueType::MIX)
        {
            return SyncDirection::BOTH;
        }
        return (currType == SkQueueType::AIC) ? SyncDirection::VEC_TO_CUB : SyncDirection::CUB_TO_VEC;
    case SkQueueType::AIC:
        switch (currType)
        {
        case SkQueueType::MIX:
            return SyncDirection::CUB_TO_VEC;
        case SkQueueType::AIC:
            return SyncDirection::CUB_TO_CUB;
        default:
            return SyncDirection::CUB_TO_VEC;
        }
    case SkQueueType::AIV:
        switch (currType)
        {
        case SkQueueType::MIX:
            return SyncDirection::VEC_TO_CUB;
        case SkQueueType::AIV:
            return SyncDirection::VEC_TO_VEC;
        default:
            return SyncDirection::VEC_TO_CUB;
        }
    default:
        return SyncDirection::VEC_TO_CUB;
    }
}

static SkQueueType ToQueueType(SkKernelType kernelType)
{
    switch (kernelType)
    {
    case SkKernelType::AIC_ONLY:
    case SkKernelType::MIX_AIC_1_0:
        return SkQueueType::AIC;
    case SkKernelType::AIV_ONLY:
    case SkKernelType::MIX_AIV_1_0:
        return SkQueueType::AIV;
    case SkKernelType::MIX_AIC_1_1:
    case SkKernelType::MIX_AIC_1_2:
        return SkQueueType::MIX;
    default:
        break;
    }

    SK_LOGE("unsupported kernel type %s for super kernel, aborting", to_string(kernelType));
    throw std::runtime_error("[sk error] Unsupported kernel type for super kernel.");
}

static TaskQuePtr InitTaskQuePtr(size_t cap)
{
    return TaskQuePtr(cap);
}

static TaskQuePtr ExtendTaskQuePtr(TaskQuePtr oldQue)
{
    oldQue.expand();
    return TaskQuePtr(std::move(oldQue));
}

// ========== 新增：任务过滤函数 ==========
static void FilterCancelledTasks(const std::vector<SuperKernelBaseNode *> &tasks,
                                 std::vector<SuperKernelBaseNode *> &filteredTasks)
{
    filteredTasks.clear();
    std::unordered_map<uint64_t, size_t> notifyEventIds;
    std::unordered_map<uint64_t, size_t> waitEventIds;

    // 第一次遍历：收集所有NOTIFY和WAIT节点的eventId映射
    for (size_t i = 0; i < tasks.size(); i++)
    {
        if (tasks[i]->GetNodeType() == SkNodeType::NODE_NOTIFY)
        {
            notifyEventIds[tasks[i]->GetEventId()] = i;
        }
        else if (tasks[i]->GetNodeType() == SkNodeType::NODE_WAIT)
        {
            waitEventIds[tasks[i]->GetEventId()] = i;
        }
    }

    // 第二次遍历：过滤对消的NOTIFY/WAIT节点
    for (size_t i = 0; i < tasks.size(); i++)
    {
        if (tasks[i]->GetNodeType() == SkNodeType::NODE_NOTIFY && waitEventIds.find(tasks[i]->GetEventId()) != waitEventIds.end())
        {
            SK_LOGI("Event[%lu] cancelled: NOTIFY task[%zu]",
                   tasks[i]->GetEventId(), i);
            continue;
        }
        else if (tasks[i]->GetNodeType() == SkNodeType::NODE_WAIT && notifyEventIds.find(tasks[i]->GetEventId()) != notifyEventIds.end())
        {
            SK_LOGI("Event[%lu] cancelled: WAIT task[%zu]",
                   tasks[i]->GetEventId(), i);
            continue;
        }
        else
        {
            filteredTasks.push_back(tasks[i]);
        }
    }

    SK_LOGI("Filtered tasks: %zu -> %zu (%zu cancelled)",
           tasks.size(), filteredTasks.size(), tasks.size() - filteredTasks.size());
}

// ========== 新增：同步信息初始化 ==========
void SkTaskBuilder::InitTaskSyncInfos(const std::vector<SuperKernelBaseNode *> &tasks)
{
    taskSyncInfos_.clear();
    taskSyncInfos_.resize(tasks.size());

    size_t kernelCount = 0;
    size_t notifyCount = 0;
    size_t waitCount = 0;

    for (size_t i = 0; i < tasks.size(); i++)
    {
        SkNodeType nodeType = tasks[i]->GetNodeType();

        switch (nodeType)
        {
        case SkNodeType::NODE_KERNEL:
        {
            // KERNEL 节点：根据 kernel 类型设置 queueType
            const KernelInfos &kernelInfo = GetKernelInfos(tasks[i]);
            taskSyncInfos_[i].queueType = ToQueueType(kernelInfo.kernelType);
            kernelCount++;
            break;
        }
        case SkNodeType::NODE_NOTIFY:
        {
            // NOTIFY 节点：继承前继KERNEL节点的类型
            auto *kernel = FindKernelNodeInDirection(i, nodeType, tasks[i], graph_, SearchDirection::PREV);
            taskSyncInfos_[i].queueType = (ToQueueType(GetKernelInfos(kernel).kernelType) == SkQueueType::AIC)
                                         ? SkQueueType::AIC
                                         : SkQueueType::AIV;
            notifyCount++;
            break;
        }
        case SkNodeType::NODE_WAIT:
        {
            // WAIT 节点：继承后续KERNEL节点的类型
            auto *kernel = FindKernelNodeInDirection(i, nodeType, tasks[i], graph_, SearchDirection::NEXT);
            taskSyncInfos_[i].queueType = (ToQueueType(GetKernelInfos(kernel).kernelType) == SkQueueType::AIC)
                                         ? SkQueueType::AIC
                                         : SkQueueType::AIV;
            waitCount++;
            break;
        }
        default:
            break;
        }
        // 生成核间同步方向: 0=CUB, 1=VEC
        switch (taskSyncInfos_[i].queueType)
        {
        case SkQueueType::AIC:
            taskSyncInfos_[i].crossSyncInfo[0] = SyncDirection::CUB_TO_CUB;
            break;
        case SkQueueType::AIV:
            taskSyncInfos_[i].crossSyncInfo[1] = SyncDirection::VEC_TO_VEC;
            break;
        case SkQueueType::MIX:
            taskSyncInfos_[i].crossSyncInfo[0] = SyncDirection::CUB_TO_CUB;
            taskSyncInfos_[i].crossSyncInfo[1] = SyncDirection::VEC_TO_VEC;
            break;
        default:
            SK_LOGW("unsupported kernel type for inter-sync, skipping sync insertion");
            break;
        }
    }

    SK_LOGI("Initialized TaskSyncInfos for %zu tasks (%zu KERNEL, %zu NOTIFY, %zu WAIT)",
           tasks.size(), kernelCount, notifyCount, waitCount);
}

// ========== 核心同步插入方法（对齐Python insert_sync_event） ==========

void SkTaskBuilder::InsertSyncEvent(size_t preIdx, size_t currIdx)
{
    SkQueueType preType = taskSyncInfos_[preIdx].queueType;
    SkQueueType currType = taskSyncInfos_[currIdx].queueType;
    // 生成核内同步方向
    SyncDirection dir = GenSyncDirection(preType, currType);

    SK_LOGI("  InsertSyncEvent: task[%zu](%s) -> task[%zu](%s), dir=%s",
           preIdx, to_string(preType), currIdx, to_string(currType), SyncDirectionName(dir));

    // 发送方（preIdx）：根据同步方向存储到对应队列
    // CUB_TO_VEC: 在 cub 队列发送 SET，在 vec 队列接收 WAIT
    // VEC_TO_CUB: 在 vec 队列发送 SET，在 cub 队列接收 WAIT
    // BOTH: 两个方向都需要
    if (dir == SyncDirection::CUB_TO_VEC || dir == SyncDirection::BOTH)
    {
        taskSyncInfos_[preIdx].cubSendInfo[currIdx] = SyncDirection::CUB_TO_VEC;
    }
    if (dir == SyncDirection::VEC_TO_CUB || dir == SyncDirection::BOTH)
    {
        taskSyncInfos_[preIdx].vecSendInfo[currIdx] = SyncDirection::VEC_TO_CUB;
    }

    // 接收方（currIdx）：根据同步方向存储到对应队列
    if (dir == SyncDirection::CUB_TO_VEC || dir == SyncDirection::BOTH)
    {
        taskSyncInfos_[currIdx].vecRecvInfo[preIdx] = SyncDirection::CUB_TO_VEC;
    }
    if (dir == SyncDirection::VEC_TO_CUB || dir == SyncDirection::BOTH)
    {
        taskSyncInfos_[currIdx].cubRecvInfo[preIdx] = SyncDirection::VEC_TO_CUB;
    }
}

// ========== 打印同步信息（调试用） ==========

void SkTaskBuilder::PrintSyncInfo(const char *stage)
{
    SK_LOGI("%s", stage);
    SK_LOGI("[VEC LIST OP]:");
    for (size_t i = 0; i < taskSyncInfos_.size(); i++)
    {
        auto &info = taskSyncInfos_[i];
        if (UsesAiv(info.queueType))
        {
            std::string vecSendStr = "vecSend={";
            for (auto &kv : info.vecSendInfo)
            {
                vecSendStr += std::to_string(kv.first) + ":" + SyncDirectionName(kv.second) + " ";
            }
            vecSendStr += "}, vecRecv={";
            for (auto &kv : info.vecRecvInfo)
            {
                vecSendStr += std::to_string(kv.first) + ":" + SyncDirectionName(kv.second) + " ";
            }
            vecSendStr += "}";
            SK_LOGI("  task[%zu](%s): %s", i, to_string(info.queueType), vecSendStr.c_str());
        }
    }
    SK_LOGI("[CUB LIST OP]:");
    for (size_t i = 0; i < taskSyncInfos_.size(); i++)
    {
        auto &info = taskSyncInfos_[i];
        if (UsesAic(info.queueType))
        {
            std::string cubSendStr = "cubSend={";
            for (auto &kv : info.cubSendInfo)
            {
                cubSendStr += std::to_string(kv.first) + ":" + SyncDirectionName(kv.second) + " ";
            }
            cubSendStr += "}, cubRecv={";
            for (auto &kv : info.cubRecvInfo)
            {
                cubSendStr += std::to_string(kv.first) + ":" + SyncDirectionName(kv.second) + " ";
            }
            cubSendStr += "}";
            SK_LOGI("  task[%zu](%s): %s", i, to_string(info.queueType), cubSendStr.c_str());
        }
    }
}

// ========== 同步关系提取方法 ==========

void SkTaskBuilder::PrecomputeSyncRelationsFromGraph(const std::vector<SuperKernelBaseNode *> &tasks)
{
    SK_LOGI("======================== Precomputing Sync Relations from Graph ========================");

    // 初始化taskSyncInfos_
    InitTaskSyncInfos(tasks);

    // 1. 提取同流内的sync关系
    SK_LOGI("[Sync by stream idx]");
    ExtractIntraStreamSync(tasks);
    // PrintSyncInfo("[After Sync by stream idx]");

    // 2. 提取跨流的sync关系（基于事件）
    SK_LOGI("[Sync by event]");
    ExtractInterStreamSync(tasks);
    // PrintSyncInfo("[After Sync by event]");
}

// label : success
void SkTaskBuilder::ExtractIntraStreamSync(const std::vector<SuperKernelBaseNode *> &tasks)
{
    SK_LOGI("ExtractIntraStreamSync: processing %zu tasks", tasks.size());

    // 按stream分组（对应Python的insert_sync_by_stream_idx）
    std::map<uint32_t, std::vector<size_t>> streamOps;
    for (size_t i = 0; i < tasks.size(); i++)
    {
        uint32_t streamIdx = tasks[i]->GetStreamIdxInGraph();
        streamOps[streamIdx].push_back(i);
    }

    auto streamfusionOption = opts.GetOption(aclskOtionType::STREAM_FUSION);
    uint32_t streamFusionValue = 0;
    if (streamfusionOption != nullptr)
    {
        streamFusionValue = streamfusionOption->GetIntValue();
    }
    if (streamFusionValue == 0 && streamOps.size() > 1) {
        SK_LOGW("Multi stream fusion is triggered with %zu streams detected, "
            "but aclskStreamFusionOption is off (value=%u). "
            "To explicitly enable multi stream fusion, set aclskStreamFusionOption to 1. "
            "Please confirm whether this fusion behavior meets your expectations.",
            streamOps.size(), streamFusionValue);
    }

    // 对每个stream内的连续任务插入同步
    for (auto &streamPair : streamOps)
    {
        auto &opList = streamPair.second;
        for (size_t j = 1; j < opList.size(); j++)
        {
            size_t preIdx = opList[j - 1];
            size_t currIdx = opList[j];
            InsertSyncEvent(preIdx, currIdx);
        }
    }
}

void SkTaskBuilder::ExtractInterStreamSync(const std::vector<SuperKernelBaseNode *> &tasks)
{
    SK_LOGI("ExtractInterStreamSync: processing event-based sync");

    // 对齐 Python 的 insert_sync_by_event 逻辑
    // C++ 中 NOTIFY/WAIT 是独立的任务节点，会生成 NotifyFunc/WaitFunc
    // 同步关系是：NOTIFY 节点 -> WAIT 节点

    std::unordered_map<uint64_t, size_t> eventSendIdx;               // eventId -> NOTIFY 节点的 taskIdx
    std::unordered_map<uint64_t, std::vector<size_t>> eventRecvIdxs; // eventId -> WAIT 节点的 taskIdx 列表

    // 遍历 tasks，从 NOTIFY/WAIT 节点提取事件关系
    for (size_t i = 0; i < tasks.size(); i++)
    {
        SuperKernelBaseNode *node = tasks[i];

        if (node->GetNodeType() == SkNodeType::NODE_NOTIFY)
        {
            // NOTIFY 节点是发送方，记录其 eventId
            uint64_t eventId = node->GetEventId();
            eventSendIdx[eventId] = i;
            SK_LOGI("  Event[%lu] NOTIFY at task[%zu]",
                   eventId, i);
        }
        else if (node->GetNodeType() == SkNodeType::NODE_WAIT)
        {
            // WAIT 节点是接收方，记录其 eventId
            uint64_t eventId = node->GetEventId();
            eventRecvIdxs[eventId].push_back(i);
            SK_LOGI("  Event[%lu] WAIT at task[%zu]",
                   eventId, i);
        }
    }

    SK_LOGI("  Found %zu events with NOTIFY, %zu events with WAIT",
           eventSendIdx.size(), eventRecvIdxs.size());

    // 建立同步关系：NOTIFY 节点 -> WAIT 节点
    for (const auto &eventPair : eventSendIdx)
    {
        uint64_t eventId = eventPair.first;
        size_t notifyIdx = eventPair.second;
        auto recvIt = eventRecvIdxs.find(eventId);
        if (recvIt != eventRecvIdxs.end())
        {
            for (size_t waitIdx : recvIt->second)
            {
                bool crossStream = (tasks[notifyIdx]->GetStreamIdxInGraph() !=
                                    tasks[waitIdx]->GetStreamIdxInGraph());

                SK_LOGI("  Insert sync: NOTIFY task[%zu] -> WAIT task[%zu] (event=%lu, crossStream=%d)",
                       notifyIdx, waitIdx, eventId, crossStream);

                InsertSyncEvent(notifyIdx, waitIdx);
            }
        }
    }
}

// ========== 优化方法（对齐Python实现） ==========

void SkTaskBuilder::OptimizeSyncRelations()
{
    SK_LOGI("======================== Optimizing Sync Relations ========================");
    // PrintSyncInfo("[INIT STATE]");

    RemoveCrossedLineSync();
    // PrintSyncInfo("[AFTER REMOVE CROSSED LINE SYNC]");

    RemoveMultiSendSync();
    RemoveMultiRecvSync();
    // PrintSyncInfo("[AFTER REMOVE MULTI EVENT SYNC]");
}

bool SkTaskBuilder::JudgeRemoveCrossSync(size_t sendIdx, size_t recvIdx, bool isCubToVec)
{
    // 对应Python的judge_remove函数
    // 检查是否存在交叉的同步线
    // Python逻辑：遍历所有索引 < recvIdx 的接收方，检查是否有交叉
    // 交叉条件：存在另一条同步线 (send_idx1 -> recv_idx1)
    //          满足 recv_idx1 < recv_idx 且 send_idx1 > send_idx

    if (isCubToVec)
    {
        // cub:vec 方向
        // 对应Python: for sub_op in self.vec_op_list[0:recv_idx][::-1]
        for (size_t otherRecvIdx = 0; otherRecvIdx < recvIdx; otherRecvIdx++)
        {
            // 检查这个接收方的接收信息
            for (auto &kv : taskSyncInfos_[otherRecvIdx].vecRecvInfo)
            {
                size_t otherSendIdx = kv.first;
                SyncDirection dir = kv.second;
                if (dir == SyncDirection::CUB_TO_VEC || dir == SyncDirection::BOTH)
                {
                    // otherRecvIdx < recvIdx 已经由循环条件保证
                    // 只需检查 otherSendIdx > sendIdx
                    if (otherSendIdx > sendIdx)
                    {
                        SK_LOGD("  Found crossed sync: task[%zu]->task[%zu] crosses with task[%zu]->task[%zu]",
                               otherSendIdx, otherRecvIdx, sendIdx, recvIdx);
                        return true;
                    }
                }
            }
        }
    }
    else
    {
        // vec:cub 方向
        // 对应Python: for sub_op in self.cub_op_list[0:recv_idx][::-1]
        for (size_t otherRecvIdx = 0; otherRecvIdx < recvIdx; otherRecvIdx++)
        {
            // 检查这个接收方的接收信息
            for (auto &kv : taskSyncInfos_[otherRecvIdx].cubRecvInfo)
            {
                size_t otherSendIdx = kv.first;
                SyncDirection dir = kv.second;
                if (dir == SyncDirection::VEC_TO_CUB || dir == SyncDirection::BOTH)
                {
                    // otherRecvIdx < recvIdx 已经由循环条件保证
                    // 只需检查 otherSendIdx > sendIdx
                    if (otherSendIdx > sendIdx)
                    {
                        SK_LOGD("  Found crossed sync: task[%zu]->task[%zu] crosses with task[%zu]->task[%zu]",
                               otherSendIdx, otherRecvIdx, sendIdx, recvIdx);
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

void SkTaskBuilder::RemoveSyncInfo(size_t sendIdx, size_t recvIdx, bool isRemoveRecv, SyncDirection dirToRemove)
{
    // 对应Python的remove_info_by_name函数
    auto removeDirection = [](std::map<size_t, SyncDirection> &info, size_t key, SyncDirection toRemove)
    {
        auto it = info.find(key);
        if (it != info.end() && it->second == toRemove)
        {
            info.erase(it);
        }
    };

    if (isRemoveRecv)
    {
        removeDirection(taskSyncInfos_[sendIdx].vecRecvInfo, recvIdx, dirToRemove);
        removeDirection(taskSyncInfos_[sendIdx].cubRecvInfo, recvIdx, dirToRemove);
    }
    else
    {
        removeDirection(taskSyncInfos_[sendIdx].vecSendInfo, recvIdx, dirToRemove);
        removeDirection(taskSyncInfos_[sendIdx].cubSendInfo, recvIdx, dirToRemove);
    }
}

void SkTaskBuilder::RemoveCrossedLineSync()
{
    SK_LOGI("RemoveCrossedLineSync: checking crossed sync relations");

    // 检查cub队列的交叉同步（CUB_TO_VEC 方向）
    for (size_t i = 0; i < taskSyncInfos_.size(); i++)
    {
        auto &info = taskSyncInfos_[i];

        std::vector<size_t> toRemove;
        for (auto &kv : info.cubSendInfo)
        {
            if (kv.second == SyncDirection::CUB_TO_VEC && JudgeRemoveCrossSync(i, kv.first, true))
            {
                toRemove.push_back(kv.first);
            }
        }
        for (size_t recvIdx : toRemove)
        {
            SK_LOGI("  Remove crossed cub:vec sync: task[%zu] -> task[%zu]", i, recvIdx);
            RemoveSyncInfo(i, recvIdx, false, SyncDirection::CUB_TO_VEC);
            RemoveSyncInfo(recvIdx, i, true, SyncDirection::CUB_TO_VEC);
        }
    }

    // 检查vec队列的交叉同步（VEC_TO_CUB 方向）
    for (size_t i = 0; i < taskSyncInfos_.size(); i++)
    {
        auto &info = taskSyncInfos_[i];

        std::vector<size_t> toRemove;
        for (auto &kv : info.vecSendInfo)
        {
            if (kv.second == SyncDirection::VEC_TO_CUB && JudgeRemoveCrossSync(i, kv.first, false))
            {
                toRemove.push_back(kv.first);
            }
        }
        for (size_t recvIdx : toRemove)
        {
            SK_LOGI("  Remove crossed vec:cub sync: task[%zu] -> task[%zu]", i, recvIdx);
            RemoveSyncInfo(i, recvIdx, false, SyncDirection::VEC_TO_CUB);
            RemoveSyncInfo(recvIdx, i, true, SyncDirection::VEC_TO_CUB);
        }
    }
}

void SkTaskBuilder::RemoveMultiSendSync()
{
    SK_LOGI("RemoveMultiSendSync: removing redundant send sync");

    // 检查vec队列（VEC_TO_CUB 方向）
    for (size_t i = 0; i < taskSyncInfos_.size(); i++)
    {
        auto &info = taskSyncInfos_[i];
        if (info.vecSendInfo.size() > 1)
        {
            std::vector<size_t> vecSendList;
            for (auto &kv : info.vecSendInfo)
            {
                if (kv.second == SyncDirection::VEC_TO_CUB)
                {
                    vecSendList.push_back(kv.first);
                }
            }
            if (vecSendList.size() > 1)
            {
                std::sort(vecSendList.begin(), vecSendList.end());
                for (size_t j = 1; j < vecSendList.size(); j++)
                {
                    size_t recvIdx = vecSendList[j];
                    SK_LOGI("  Remove multi vec:cub send sync: task[%zu] -> task[%zu]", i, recvIdx);
                    RemoveSyncInfo(i, recvIdx, false, SyncDirection::VEC_TO_CUB);
                    RemoveSyncInfo(recvIdx, i, true, SyncDirection::VEC_TO_CUB);
                }
            }
        }
    }

    // 检查cub队列（CUB_TO_VEC 方向）
    for (size_t i = 0; i < taskSyncInfos_.size(); i++)
    {
        auto &info = taskSyncInfos_[i];
        if (info.cubSendInfo.size() > 1)
        {
            std::vector<size_t> cubSendList;
            for (auto &kv : info.cubSendInfo)
            {
                if (kv.second == SyncDirection::CUB_TO_VEC)
                {
                    cubSendList.push_back(kv.first);
                }
            }
            if (cubSendList.size() > 1)
            {
                std::sort(cubSendList.begin(), cubSendList.end());
                for (size_t j = 1; j < cubSendList.size(); j++)
                {
                    size_t recvIdx = cubSendList[j];
                    SK_LOGI("  Remove multi cub:vec send sync: task[%zu] -> task[%zu]", i, recvIdx);
                    RemoveSyncInfo(i, recvIdx, false, SyncDirection::CUB_TO_VEC);
                    RemoveSyncInfo(recvIdx, i, true, SyncDirection::CUB_TO_VEC);
                }
            }
        }
    }
}

void SkTaskBuilder::RemoveMultiRecvSync()
{
    SK_LOGI("RemoveMultiRecvSync: removing redundant recv sync");

    // 检查vec队列（CUB_TO_VEC 方向）
    for (size_t i = 0; i < taskSyncInfos_.size(); i++)
    {
        auto &info = taskSyncInfos_[i];
        if (info.vecRecvInfo.size() > 1)
        {
            std::vector<size_t> vecRecvList;
            for (auto &kv : info.vecRecvInfo)
            {
                if (kv.second == SyncDirection::CUB_TO_VEC)
                {
                    vecRecvList.push_back(kv.first);
                }
            }
            if (vecRecvList.size() > 1)
            {
                // 按索引倒序排序，保留最后一个（最后一个接收）
                std::sort(vecRecvList.begin(), vecRecvList.end(), std::greater<>());
                for (size_t j = 1; j < vecRecvList.size(); j++)
                {
                    size_t sendIdx = vecRecvList[j];
                    SK_LOGI("  Remove multi cub:vec recv sync: task[%zu] <- task[%zu]", i, sendIdx);
                    RemoveSyncInfo(i, sendIdx, true, SyncDirection::CUB_TO_VEC);
                    RemoveSyncInfo(sendIdx, i, false, SyncDirection::CUB_TO_VEC);
                }
            }
        }
    }

    // 检查cub队列（VEC_TO_CUB 方向）
    for (size_t i = 0; i < taskSyncInfos_.size(); i++)
    {
        auto &info = taskSyncInfos_[i];
        if (info.cubRecvInfo.size() > 1)
        {
            std::vector<size_t> cubRecvList;
            for (auto &kv : info.cubRecvInfo)
            {
                if (kv.second == SyncDirection::VEC_TO_CUB)
                {
                    cubRecvList.push_back(kv.first);
                }
            }
            if (cubRecvList.size() > 1)
            {
                std::sort(cubRecvList.begin(), cubRecvList.end(), std::greater<>());
                for (size_t j = 1; j < cubRecvList.size(); j++)
                {
                    size_t sendIdx = cubRecvList[j];
                    SK_LOGI("  Remove multi vec:cub recv sync: task[%zu] <- task[%zu]", i, sendIdx);
                    RemoveSyncInfo(i, sendIdx, true, SyncDirection::VEC_TO_CUB);
                    RemoveSyncInfo(sendIdx, i, false, SyncDirection::VEC_TO_CUB);
                }
            }
        }
    }
}

// ========== 同步任务插入方法 ==========

static SkCoreSyncType GetSyncTypesForDirection(SyncDirection dir, bool isSend)
{
    SkCoreSyncType syncType = SkCoreSyncType::SYNC_NONE;

    // 注意：现在只会收到 CUB_TO_VEC 或 VEC_TO_CUB，不会再有 BOTH
    switch (dir)
    {
    case SyncDirection::CUB_TO_VEC:
        // AIC -> AIV
        syncType = isSend ? SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV : SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV;
        break;
    case SyncDirection::VEC_TO_CUB:
        // AIV -> AIC
        syncType = isSend ? SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC : SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC;
        break;
    case SyncDirection::CUB_TO_CUB:
        // AIC -> AIC
        syncType = SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC;
        break;
    case SyncDirection::VEC_TO_VEC:
        // AIV -> AIV
        syncType = SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV;
        break;
    case SyncDirection::BOTH:
        // 不应该再出现，因为 InsertSyncEvent 已经拆分
        syncType = SkCoreSyncType::SYNC_NONE;
        SK_LOGW("GetSyncTypesForDirection: unexpected BOTH direction");
        break;
    case SyncDirection::NONE:
        syncType = SkCoreSyncType::SYNC_NONE;
        break;
    case SyncDirection::DEBUG:
        syncType = SkCoreSyncType::ALL_SYNC_DEBUG;
        break;
    default:
        syncType = SkCoreSyncType::SYNC_NONE;
        SK_LOGW("GetSyncTypesForDirection: unknown direction %s", to_string(dir));
        break;
    }
    return syncType;
}

// ========== 原有方法（保持不变） ==========

DeviceArgsPtr SkTaskBuilder::GenEntryArgs(const SkTask &skTaskCube,
                                                const SkTask &skTaskVec,
                                                const SkDfxInfo *dfxInfos,
                                                uint32_t dfxCount)
{
    size_t header_size = sizeof(SkHeaderInfo);
    size_t aic_que_size = GetTaskQueSize(skTaskCube.taskQue.get());
    size_t aiv_que_size = GetTaskQueSize(skTaskVec.taskQue.get());
    size_t counter_size = DEFAULT_COUNTER_COUNT * sizeof(SkCounterInfo);
    size_t ws_size = sizeof(SkWorkSpace);
    size_t dfx_size = dfxCount * sizeof(SkDfxInfo);

    size_t aic_que_offset = header_size;
    size_t aiv_que_offset = aic_que_offset + aic_que_size;
    size_t counter_offset = aiv_que_offset + aiv_que_size;
    size_t ws_offset = counter_offset + counter_size;
    size_t dfx_offset = ws_offset + ws_size;

    uint64_t total_size = dfx_offset + dfx_size;
    DeviceArgsPtr args(total_size);
    if (args.get() == nullptr)
    {
        return DeviceArgsPtr();
    }
    args.get()->skHeader.aicQueOffset = aic_que_offset;
    args.get()->skHeader.aivQueOffset = aiv_que_offset;
    args.get()->skHeader.counterOffset = counter_offset;
    args.get()->skHeader.wsOffset = ws_offset;
    args.get()->skHeader.dfxOffset = dfx_offset;
    args.get()->skHeader.totalSize = total_size;

    uint8_t *base = (uint8_t *)args.get();
    TaskQue *dst_aic = (TaskQue *)(base + aic_que_offset);
    errno_t err = memcpy_s(dst_aic, aic_que_size, skTaskCube.taskQue.get(), aic_que_size);
    if (err != 0)
    {
        SK_LOGE("GenEntryArgs memcpy_s AIC queue failed, ret=%d", static_cast<int>(err));
        return DeviceArgsPtr();
    }
    TaskQue *dst_aiv = (TaskQue *)(base + aiv_que_offset);
    err = memcpy_s(dst_aiv, aiv_que_size, skTaskVec.taskQue.get(), aiv_que_size);
    if (err != 0)
    {
        SK_LOGE("GenEntryArgs memcpy_s AIV queue failed, ret=%d", static_cast<int>(err));
        return DeviceArgsPtr();
    }
    if (counter_size > 0)
    {
        err = memset_s(base + counter_offset, counter_size, 0, counter_size);
        if (err != 0)
        {
            SK_LOGE("GenEntryArgs memset_s counter failed, ret=%d", static_cast<int>(err));
            return DeviceArgsPtr();
        }
    }
    if (ws_size > 0)
    {
        err = memset_s(base + ws_offset, ws_size, 0, ws_size);
        if (err != 0)
        {
            SK_LOGE("GenEntryArgs memset_s workspace failed, ret=%d", static_cast<int>(err));
            return DeviceArgsPtr();
        }
    }
    if (dfx_size > 0 && dfxInfos != nullptr)
    {
        err = memcpy_s((uint8_t *)args.get() + args.get()->skHeader.dfxOffset, dfx_size, dfxInfos, dfxCount * sizeof(SkDfxInfo));
        if (err != 0)
        {
            SK_LOGE("GenEntryArgs memcpy_s dfx failed, ret=%d", static_cast<int>(err));
            return DeviceArgsPtr();
        }
    }
    args.get()->skHeader.nodeCnt = dfxCount;
    return args;
}

std::pair<int, int> SkTaskBuilder::GetPreFetchCnt(const ResolvedFunctionInfo &resolved)
{
    std::pair<int, int> preFetchCntValue = std::make_pair(resolved.prefetchCnt[0], resolved.prefetchCnt[1]);
    auto preLoadOptions = opts.GetOption(aclskOtionType::PRELOAD_CODE);
    
    // default: preLoadValue == 1, use func size to prefetch
    uint32_t preLoadValue = 1;
    
    if (preLoadOptions != nullptr) {
        preLoadValue = preLoadOptions->GetIntValue();
    }
    
    if (preLoadValue == 0) {
        // use max size to prefetch
        preFetchCntValue.first = 16;  // cube's max icache size is 32K(16 * 2)
        preFetchCntValue.second = 8;  // vec's max icache size is 16K(8 * 2)
    } else if (preLoadValue == 2) {
        // no preload
        preFetchCntValue.first = 0;
        preFetchCntValue.second = 0;
    }
    // preLoadValue == 1: use func size (default from resolved.prefetchCnt)

    return preFetchCntValue;
}

void SkTaskBuilder::AddTask(SkTask &skTask, SkDfxInfo *dfxInfos, const std::vector<SuperKernelBaseNode *> &tasks, size_t index,
                            SkKernelType kernelType, int customArg, int binCount, SkTaskType taskType,
                            uint32_t syncFlag = (uint32_t)SkCoreSyncType::ALL_SYNC_DEBUG)
{
    if (index >= tasks.size()) {
        return;
    }

    TaskQue *taskQue = skTask.taskQue.get();
    if (taskQue->taskCnt >= taskQue->cap) {
        skTask.taskQue = ExtendTaskQuePtr(std::move(skTask.taskQue));
    }

    auto syncAllOptions = opts.GetOption(aclskOtionType::DEBUG_SYNC_ALL);
    uint32_t debugSyncAll = 0;
    if (syncAllOptions != nullptr) {
        debugSyncAll = syncAllOptions->GetIntValue();
    }
    auto disableDcciOptions = opts.GetOption(aclskOtionType::DEBUG_DCCI_DISABLE_ON_KERNEL);
    std::vector<std::string> disableDcciList;
    if (disableDcciOptions != nullptr) {
        disableDcciList = disableDcciOptions->GetStringListValue();
    }

    TaskInfo &taskInfo = taskQue->taskInfos[taskQue->taskCnt];
    taskInfo.index = index;

    if (taskType == SkTaskType::TYPE_SYNC) {
        taskInfo.type = taskType;
        taskInfo.args = syncFlag;
        if (opts.EnableDebug() && debugSyncAll == 1) {
            taskInfo.debugOptions |= 0x2;
        }
    } else if (taskType == SkTaskType::TYPE_EVENT_NOTIFY || taskType == SkTaskType::TYPE_EVENT_WAIT) {
        if (tasks[index]->GetNodeType() != SkNodeType::NODE_NOTIFY &&
            tasks[index]->GetNodeType() != SkNodeType::NODE_WAIT) {
            throw std::runtime_error("[sk error] unsupported node type for EVENT_NOTIFY/EVENT_WAIT task");
        }
        taskInfo.type = taskType;
        taskInfo.args = (uint64_t)tasks[index]->GetNodeInfos().syncInfos.addrValue;
    } else if (taskType == SkTaskType::TYPE_PRELOAD || taskType == SkTaskType::TYPE_FUNC) {
        if (tasks[index]->GetNodeType() != SkNodeType::NODE_KERNEL) {
            throw std::runtime_error("[sk error] unsupported node type for PRELOAD/FUNC task");
        }
        const KernelInfos &kernelInfo = tasks[index]->GetNodeInfos().kernelInfos;
        taskInfo.type = taskType;
        taskInfo.originType = kernelInfo.kernelType;
        if (kernelInfo.kernelType == SkKernelType::MIX_AIC_1_2 && customArg == 1) {
            taskInfo.numBlocks = kernelInfo.numBlocks * 2;
        } else {
            taskInfo.numBlocks = kernelInfo.numBlocks;
        }

        for (int i = 0; i < binCount; i++) {
            const ResolvedFunctionInfo &resolved = kernelInfo.resolvedFuncs[i];
            if (taskType == SkTaskType::TYPE_PRELOAD) {
                std::pair<int,int> prefetchCntValue = GetPreFetchCnt(resolved);
                if (i == 0) {
                    SK_LOGI("kernel name: %s, prefetch count: %d %d",
                        kernelInfo.funcName.c_str(), prefetchCntValue.first, prefetchCntValue.second);
                }
                taskInfo.args = customArg == 0 ? prefetchCntValue.first : prefetchCntValue.second;
            }
            uint64_t addr = resolved.funcAddr[customArg];
            if (addr == 0) {
                throw std::runtime_error("[sk error] unresolved function address");
            }

            taskInfo.entryCnt += 1;
            taskInfo.entry[i] = addr;
            if (taskType == SkTaskType::TYPE_FUNC && dfxInfos) {
                dfxInfos[index].binHdl = (uint64_t)kernelInfo.binHdl;
                dfxInfos[index].funcHdl = (uint64_t)resolved.funcHdl;
                dfxInfos[index].funcHdlOri = (uint64_t)kernelInfo.funcHdl;
            }
        }

        if (taskType == SkTaskType::TYPE_FUNC) {
            taskInfo.args = (uint64_t)kernelInfo.devArgs;
            skTask.numBlocks = std::max(skTask.numBlocks, (uint32_t)taskInfo.numBlocks);
            skTask.funcCnt++;
            if (!disableDcciList.empty() && !kernelInfo.funcName.empty()) {
                bool disable = opts.JudgeDisableKernelDcci(disableDcciList, kernelInfo.funcName);
                if (disable) {
                    taskInfo.debugOptions |= 0x1;
                }
            }
        }
    } else {
        throw std::runtime_error("[sk error] unsupported task type for AddTask");
    }
    taskQue->taskCnt++;
}

void SkTaskBuilder::DispatchTask(SkTask &skTaskCube, SkTask &skTaskVec, SkDfxInfo *dfxInfos,
                                 const std::vector<SuperKernelBaseNode *> &tasks, size_t index,
                                 int binCount, SkTaskType taskType)
{
    auto calcBlocks = [&](const KernelInfos &kernelInfo, int customArg)
    {
        if (kernelInfo.kernelType == SkKernelType::MIX_AIC_1_2 && customArg == 1)
        {
            return kernelInfo.numBlocks * 2;
        }
        return kernelInfo.numBlocks;
    };
    if (taskType == SkTaskType::TYPE_FUNC || taskType == SkTaskType::TYPE_PRELOAD)
    {
        if (tasks[index]->GetNodeType() != SkNodeType::NODE_KERNEL)
        {
            throw std::runtime_error("[sk error] unsupported node type for FUNC/PRELOAD task");
        }
        auto kernelInfo = tasks[index]->GetNodeInfos().kernelInfos;
        switch (kernelInfo.kernelType)
        {
        case SkKernelType::AIV_ONLY:
        case SkKernelType::MIX_AIV_1_0:
            AddTask(skTaskVec, dfxInfos, tasks, index, SkKernelType::AIV_ONLY, 1, binCount, taskType);
            SK_LOGI("task insert: stask %zu, queue=aiv, type=%s, kernel=%s, numBlocks=%u",
                   index, to_string(taskType), to_string(kernelInfo.kernelType),
                   calcBlocks(kernelInfo, 1));
            break;
        case SkKernelType::AIC_ONLY:
        case SkKernelType::MIX_AIC_1_0:
            AddTask(skTaskCube, dfxInfos, tasks, index, SkKernelType::AIC_ONLY, 0, binCount, taskType);
            SK_LOGI("task insert: stask %zu, queue=aic, type=%s, kernel=%s, numBlocks=%u",
                   index, to_string(taskType), to_string(kernelInfo.kernelType),
                   calcBlocks(kernelInfo, 0));
            break;
        case SkKernelType::MIX_AIC_1_1:
            AddTask(skTaskCube, dfxInfos, tasks, index, SkKernelType::MIX_AIC_1_1, 0, binCount, taskType);
            AddTask(skTaskVec, dfxInfos, tasks, index, SkKernelType::MIX_AIC_1_1, 1, binCount, taskType);
            SK_LOGI("task insert: stask %zu, queue=aic, type=%s, kernel=%s, numBlocks=%u",
                   index, to_string(taskType), to_string(kernelInfo.kernelType),
                   calcBlocks(kernelInfo, 0));
            SK_LOGI("task insert: stask %zu, queue=aiv, type=%s, kernel=%s, numBlocks=%u",
                   index, to_string(taskType), to_string(kernelInfo.kernelType),
                   calcBlocks(kernelInfo, 1));
            break;
        case SkKernelType::MIX_AIC_1_2:
            AddTask(skTaskCube, dfxInfos, tasks, index, SkKernelType::MIX_AIC_1_2, 0, binCount, taskType);
            AddTask(skTaskVec, dfxInfos, tasks, index, SkKernelType::MIX_AIC_1_2, 1, binCount, taskType);
            skTaskVec.nodeType = SkKernelType::MIX_AIC_1_2;
            skTaskCube.nodeType = SkKernelType::MIX_AIC_1_2;
            SK_LOGI(" task insert: stask %zu, queue=aic, type=%s, kernel=%s, numBlocks=%u",
                   index, to_string(taskType), to_string(kernelInfo.kernelType),
                   calcBlocks(kernelInfo, 0));
            SK_LOGI(" task insert: stask %zu, queue=aiv, type=%s, kernel=%s, numBlocks=%u",
                   index, to_string(taskType), to_string(kernelInfo.kernelType),
                   calcBlocks(kernelInfo, 1));
            break;
        default:
            SK_LOGE("unsupported kernel type %s for super kernel, aborting", to_string(kernelInfo.kernelType));
            throw std::runtime_error("[sk error] Unsupported kernel type for super kernel.");
        }
    }
    else if (taskType == SkTaskType::TYPE_EVENT_NOTIFY || taskType == SkTaskType::TYPE_EVENT_WAIT)
    {
        if (tasks[index]->GetNodeType() != SkNodeType::NODE_NOTIFY &&
            tasks[index]->GetNodeType() != SkNodeType::NODE_WAIT)
        {
            throw std::runtime_error("[sk error] unsupported node type for EVENT_NOTIFY/EVENT_WAIT task");
        }
        auto queueType = taskSyncInfos_[index].queueType;
        SkTask *targetTask = nullptr;
        switch (queueType)
        {
        case SkQueueType::AIC:
            targetTask = &skTaskCube;
            break;
        case SkQueueType::AIV:
            targetTask = &skTaskVec;
            break;
        default:
            throw std::runtime_error("[sk error] unsupported kernel type for event notify/wait task");
        }
        AddTask(*targetTask, nullptr, tasks, index, SkKernelType::DEFAULT, 0, 0, taskType);
        const auto &syncInfos = tasks[index]->GetNodeInfos().syncInfos;
        uint64_t nodeId = (taskType == SkTaskType::TYPE_EVENT_NOTIFY) ? syncInfos.notifyNodeId : syncInfos.waitNodeId;
        const char *nodeIdName = (taskType == SkTaskType::TYPE_EVENT_NOTIFY) ? "notifyNodeId" : "waitNodeId";
        SK_LOGI("task insert: stask %zu, queue=%s, type=%s, eventId=%lu, %s=%lu",
               index, to_string(queueType), to_string(taskType), syncInfos.eventId, nodeIdName, nodeId);
    }
    else
    {
        throw std::runtime_error("[sk error] unsupported task type for DispatchTask");
    }
}

void SkTaskBuilder::DispatchTaskSync(SkTask &skTaskCube, SkTask &skTaskVec, SkDfxInfo *dfxInfos,
                                     const std::vector<SuperKernelBaseNode *> &tasks, size_t index, int binCount,
                                     SkTaskType taskType, SkCoreSyncType syncFlag, bool addToAicQue, bool addToAivQue,
                                     SkQueueType prevType, SkQueueType nextType, const char *detail)
{
    if (addToAicQue)
    {
        AddTask(skTaskCube, dfxInfos, tasks, index, SkKernelType::DEFAULT, 0, binCount, taskType,
                static_cast<uint32_t>(syncFlag));
        SK_LOGI("sync insert: stask %zu, queue=aic, type=%s, flag=%s, prev=%s, next=%s%s%s",
               index, to_string(taskType), to_string(syncFlag),
               to_string(prevType), to_string(nextType),
               detail ? ", detail=" : "", detail ? detail : "");
    }
    if (addToAivQue)
    {
        AddTask(skTaskVec, dfxInfos, tasks, index, SkKernelType::DEFAULT, 0, binCount, taskType,
                static_cast<uint32_t>(syncFlag));
        SK_LOGI("sync insert: stask %zu, queue=aiv, type=%s, flag=%s, prev=%s, next=%s%s%s",
               index, to_string(taskType), to_string(syncFlag),
               to_string(prevType), to_string(nextType),
               detail ? ", detail=" : "", detail ? detail : "");
    }
}

// ========== 辅助函数：批量处理同步任务 ==========

void SkTaskBuilder::DispatchSyncTasks(SkTask &skTaskCube, SkTask &skTaskVec, SkDfxInfo *dfxInfos,
                                      const std::vector<SuperKernelBaseNode *> &tasks,
                                      const std::map<size_t, SyncDirection> &syncInfo,
                                      size_t myIdx, int binCount, bool isSend)
{
    SkQueueType myType = taskSyncInfos_[myIdx].queueType;

    for (const auto &kv : syncInfo)
    {
        size_t peerIdx = kv.first;
        SyncDirection dir = kv.second;
        SkQueueType peerType = taskSyncInfos_[peerIdx].queueType;
        auto syncType = GetSyncTypesForDirection(dir, isSend);

        // 根据同步类型确定分发目标
        bool addToAicQue = false;
        bool addToAivQue = false;
        SkQueueType prevType = SkQueueType::UNKNOWN;
        SkQueueType nextType = SkQueueType::UNKNOWN;
        switch (syncType)
        {
        case SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC:
            addToAicQue = true;
            addToAivQue = false;
            prevType = myType;
            nextType = SkQueueType::UNKNOWN;
            break;
        case SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV:
            addToAicQue = false;
            addToAivQue = true;
            prevType = myType;
            nextType = SkQueueType::UNKNOWN;
            break;
        case SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC:
            // AIV SET信号：插入AIV队列
            addToAicQue = false;
            addToAivQue = true;
            prevType = myType;
            nextType = peerType;
            break;
        case SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC:
            // AIV->AIC 的 WAIT：插入AIC队列
            addToAicQue = true;
            addToAivQue = false;
            prevType = peerType;
            nextType = myType;
            break;
        case SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV:
            // AIC SET信号：插入AIC队列
            addToAicQue = true;
            addToAivQue = false;
            prevType = myType;
            nextType = peerType;
            break;
        case SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV:
            // AIC->AIV 的 WAIT：插入AIV队列
            addToAicQue = false;
            addToAivQue = true;
            prevType = peerType;
            nextType = myType;
            break;
        case SkCoreSyncType::ALL_SYNC_DEBUG:
            addToAicQue = true;
            addToAivQue = true;
            prevType = SkQueueType::UNKNOWN;
            nextType = SkQueueType::UNKNOWN; // special case for debug sync
            break;
        default:
        {
            SK_LOGE("unknown sync type %s, skipping", to_string(syncType));
            continue;
        }
        }

        DispatchTaskSync(skTaskCube, skTaskVec, dfxInfos, tasks, myIdx, binCount,
                         SkTaskType::TYPE_SYNC, syncType, addToAicQue, addToAivQue,
                         prevType, nextType, to_string(syncType));
    }
}

SkHostEntryInfo SkTaskBuilder::GenEntryInfo(SkTask &skTaskCube, SkTask &skTaskVec)
{
    SkHostEntryInfo entryInfo;
    bool enableDebug = opts.EnableDebug();
    if (skTaskCube.funcCnt == 0 && skTaskVec.funcCnt > 0)
    {
        entryInfo.skEntryFuncName = enableDebug == false ? "sk_entry_aiv" : "sk_entry_aiv_debug";
        entryInfo.numBlocks = skTaskVec.numBlocks;
        skTaskCube.numBlocks = 0;
    }
    else if (skTaskCube.funcCnt > 0 && skTaskVec.funcCnt == 0)
    {
        entryInfo.skEntryFuncName = enableDebug == false ? "sk_entry_aic" : "sk_entry_aic_debug";
        entryInfo.numBlocks = skTaskCube.numBlocks;
        skTaskVec.numBlocks = 0;
    }
    else if (skTaskCube.funcCnt > 0 && skTaskVec.funcCnt > 0)
    {
        uint32_t mix_1_2_aiv_numBlocks = (skTaskVec.numBlocks + 1) / 2;
        if (skTaskCube.nodeType == SkKernelType::MIX_AIC_1_2 && skTaskVec.nodeType == SkKernelType::MIX_AIC_1_2)
        {
            entryInfo.skEntryFuncName = enableDebug == false ? "sk_entry_mix12" : "sk_entry_mix12_debug";
            entryInfo.numBlocks = std::max(skTaskCube.numBlocks, mix_1_2_aiv_numBlocks);
            skTaskCube.numBlocks = entryInfo.numBlocks;
            skTaskVec.numBlocks = entryInfo.numBlocks * 2;
        }
        else if (skTaskVec.numBlocks <= skTaskCube.numBlocks)
        {
            entryInfo.skEntryFuncName = enableDebug == false ? "sk_entry_mix11" : "sk_entry_mix11_debug";
            entryInfo.numBlocks = skTaskCube.numBlocks;
            skTaskVec.numBlocks = skTaskCube.numBlocks;
        }
        else
        {
            entryInfo.skEntryFuncName = enableDebug == false ? "sk_entry_mix12" : "sk_entry_mix12_debug";
            entryInfo.numBlocks = std::max(skTaskCube.numBlocks, mix_1_2_aiv_numBlocks);
            skTaskCube.numBlocks = entryInfo.numBlocks;
            skTaskVec.numBlocks = entryInfo.numBlocks * 2;
        }
    }
    else
    {
        SK_LOGE("both skTaskCube and skTaskVec have no task, aborting");
        throw std::runtime_error("No task for super kernel");
    }

    auto *taskQue = skTaskVec.taskQue.get();
    if (entryInfo.skEntryFuncName == "sk_entry_mix12" || entryInfo.skEntryFuncName == "sk_entry_mix12_debug")
    {
        for (auto i = 0; i < taskQue->taskCnt; i++)
        {
            TaskInfo &taskInfo = taskQue->taskInfos[i];
            if (taskInfo.type != SkTaskType::TYPE_FUNC)
            {
                continue;
            }
            if (taskInfo.originType == SkKernelType::MIX_AIC_1_1)
            {
                taskInfo.numBlocks = taskInfo.numBlocks * 2;
            }
        }
    }
    return entryInfo;
}

// ========== Build方法（使用新的同步逻辑） ==========

SkLaunchInfo SkTaskBuilder::Build(const std::vector<SuperKernelBaseNode *> &tasks)
{
    SkLaunchInfo launchInfo;

    // 原始任务为空，直接返回
    if (tasks.empty())
    {
        SK_LOGW("no task to build for super kernel");
        return launchInfo;
    }

    // ========== [新增] 过滤对消的NOTIFY/WAIT节点 ==========
    std::vector<SuperKernelBaseNode *> filteredTasks;
    FilterCancelledTasks(tasks, filteredTasks);

    if (filteredTasks.empty())
    {
        SK_LOGW("no task to build for super kernel after filtering");
        return launchInfo;
    }

    const size_t taskCount = filteredTasks.size();

    size_t cap = taskCount * (uint8_t)SkTaskType::TYPE_MAX;
    SkTask aicTask;
    SkTask aivTask;
    aicTask.taskQue = InitTaskQuePtr(cap);
    aivTask.taskQue = InitTaskQuePtr(cap);

    const auto *debugSyncOpt = opts.GetOption(aclskOtionType::DEBUG_SYNC_ALL);
    const bool debugSyncAll = (debugSyncOpt != nullptr && debugSyncOpt->GetIntValue() != 0);

    if (taskCount > (std::numeric_limits<size_t>::max() / sizeof(SkDfxInfo)))
    {
        SK_LOGE("invalid dfxInfos alloc size, taskCount=%zu", taskCount);
        return launchInfo;
    }

    std::unique_ptr<SkDfxInfo[]> dfxInfos = std::make_unique<SkDfxInfo[]>(taskCount);
    if (dfxInfos == nullptr)
    {
        SK_LOGE("malloc dfxInfos failed");
        return launchInfo;
    }

    int splitBinCount = 4;
    auto splitOptions = opts.GetOption(aclskOtionType::SPLIT_MODE);
    if (splitOptions != nullptr)
    {
        splitBinCount = splitOptions->GetIntValue();
    }

    // ========== 阶段1：预计算所有sync关系（基于Graph拓扑） ==========
    PrecomputeSyncRelationsFromGraph(filteredTasks);

    // ========== 阶段2：后处理优化sync关系 ==========
    OptimizeSyncRelations();

    // [DEBUG] 下输出最终的同步关系
    if (opts.EnableDebug() && debugSyncAll)
    {
        for (int i = 0; i < static_cast<int>(taskCount); i++)
        {
            // 清空所有的同步flag信息，强行重置为DEBUG状态
            taskSyncInfos_[i].vecRecvInfo.clear();
            taskSyncInfos_[i].cubRecvInfo.clear();
            taskSyncInfos_[i].vecSendInfo.clear();
            taskSyncInfos_[i].cubSendInfo.clear();
            taskSyncInfos_[i].crossSyncInfo.clear();
            taskSyncInfos_[i].crossSyncInfo[0] = SyncDirection::DEBUG;
        }
    }
    // ========== 阶段3：构建任务队列 ==========
    int preloadCount = 1;
    for (int i = 0; i < preloadCount && i < static_cast<int>(taskCount); i++)
    {
        if (filteredTasks[i]->GetNodeType() == SkNodeType::NODE_KERNEL)
        {
            DispatchTask(aicTask, aivTask, dfxInfos.get(), filteredTasks, i, splitBinCount, SkTaskType::TYPE_PRELOAD);
        }
    }

    for (int i = 0; i < static_cast<int>(taskCount); i++)
    {
        SK_LOGI("======================== start process %d, nodeType=%s ========================", i,
                 to_string(filteredTasks[i]->GetNodeType()));

        auto &info = taskSyncInfos_[i];

        // 1. 插入核内同步WAIT信号（在任务启动前）
        DispatchSyncTasks(aicTask, aivTask, dfxInfos.get(), filteredTasks, info.vecRecvInfo, i, splitBinCount, false);
        DispatchSyncTasks(aicTask, aivTask, dfxInfos.get(), filteredTasks, info.cubRecvInfo, i, splitBinCount, false);

        // 2. 添加功能任务
        switch (filteredTasks[i]->GetNodeType())
        {
        case SkNodeType::NODE_KERNEL:
            DispatchTask(aicTask, aivTask, dfxInfos.get(), filteredTasks, i, splitBinCount, SkTaskType::TYPE_FUNC);
            break;
        case SkNodeType::NODE_NOTIFY:
            DispatchTask(aicTask, aivTask, dfxInfos.get(), filteredTasks, i, splitBinCount, SkTaskType::TYPE_EVENT_NOTIFY);
            break;
        case SkNodeType::NODE_WAIT:
            DispatchTask(aicTask, aivTask, dfxInfos.get(), filteredTasks, i, splitBinCount, SkTaskType::TYPE_EVENT_WAIT);
            break;
        default:
            SK_LOGW("process task %d: unsupported node type %s, skipping (supported types: KERNEL, NOTIFY, WAIT)",
                    i, to_string(filteredTasks[i]->GetNodeType()));
            break;
        }

        // 3. 添加预加载任务
        if (preloadCount > 0 && i + preloadCount < static_cast<int>(taskCount) && filteredTasks[i + preloadCount]->GetNodeType() == SkNodeType::NODE_KERNEL)
        {
            DispatchTask(aicTask, aivTask, dfxInfos.get(), filteredTasks, i + preloadCount, splitBinCount, SkTaskType::TYPE_PRELOAD);
        }

        // 4. 插入核间同步信号 or 强行为DEBUG信号
        DispatchSyncTasks(aicTask, aivTask, dfxInfos.get(), filteredTasks, info.crossSyncInfo, i, splitBinCount, true);

        // 5. 插入核内同步SET信号（在任务结束后）
        DispatchSyncTasks(aicTask, aivTask, dfxInfos.get(), filteredTasks, info.vecSendInfo, i, splitBinCount, true);
        DispatchSyncTasks(aicTask, aivTask, dfxInfos.get(), filteredTasks, info.cubSendInfo, i, splitBinCount, true);
    }

    launchInfo.entryInfo = GenEntryInfo(aicTask, aivTask);
    launchInfo.entryInfo.nodeCnt = static_cast<uint32_t>(taskCount);

    launchInfo.devArgs = GenEntryArgs(aicTask, aivTask,
                                      dfxInfos.get(), static_cast<uint32_t>(taskCount));

    if (opts.EnableDebug())
    {
        DumpDeviceArgs(launchInfo.devArgs.get());
    }
    return launchInfo;
}
