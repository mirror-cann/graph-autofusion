/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>
#include "sk_optimizer.h"
#include "sk_scope_split.h"
#include "kernel_launcher.h"
#include "sk_task_builder.h"


namespace {

bool FileExists(const std::string &path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

aclrtFuncHandle ResolveSkEntryFunc(const char *funcName) {
    std::string skEntryPath = "./sk_oop/kernel/sk_entry.o";
    if (!FileExists(skEntryPath)) {
        printf("[sk error] sk_entry.o not found, tried path: %s\n", skEntryPath.c_str());
        return nullptr;
    }

    aclrtBinHandle bhdl = nullptr;
    CHECK_ACL(aclrtBinaryLoadFromFile(skEntryPath.c_str(), nullptr, &bhdl));

    aclrtFuncHandle fhdl = nullptr;
    CHECK_ACL(aclrtBinaryGetFunction(bhdl, funcName, &fhdl));
    if (fhdl == nullptr) {
        printf("[sk error] failed to resolve entry func handle, source: %s\n", skEntryPath.c_str());
        return nullptr;
    }
    return fhdl;
}

} // namespace

std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> SuperKernelOptimizer::SplitGraph(const SuperKernelGraph &graph) const {
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> result;

    // 判断节点是否可融合到 SuperKernel
    auto IsFusible = [](const SuperKernelBaseNode *node) -> bool {
        if (node == nullptr) {
            return false;
        }
        // 只融合 KERNEL 类型的节点
        if (node->GetNodeType() != SkNodeType::NODE_KERNEL) {
            return false;
        }
        // KERNEL 节点必须满足 kernelType 不是 DEFAULT
        auto *kernelNode = dynamic_cast<const SuperKernelKernelNode *>(node);
        if (kernelNode == nullptr) {
            return false;
        }
        SkKernelType type = kernelNode->GetKernelType();
        return type != SkKernelType::DEFAULT;
    };

    // step: graph.GetHeadNodes() ----> nodes[已处理完]  | --nodes[未处理]
    // nodes---dfs--->nodes[已处理完]  | --nodes[未处理]


    // 遍历每个 head 节点，开始切分
    for (uint64_t headId : graph.GetHeadNodes()) {
        std::vector<uint64_t> currentSubGraphHeads;
        std::vector<uint64_t> currentSubGraphTails;
        bool inSubGraph = false;

        uint64_t nodeId = headId;
        uint64_t prevNodeId = INVALID_TASK_ID;

        while (nodeId != INVALID_TASK_ID) {
            SuperKernelBaseNode *node = graph.GetNodeById(nodeId);
            if (node == nullptr) {
                break;
            }

            // 如果当前节点可融合
            if (IsFusible(node)) {
                // 记录当前子图的起始节点
                if (!inSubGraph) {
                    inSubGraph = true;
                    currentSubGraphHeads.push_back(nodeId);
                }
                prevNodeId = nodeId;
            } else {
                // 当前节点不可融合，需要断开
                // 如果之前有可融合的子图，保存它（前提是 prevNodeId 有效）
                if (inSubGraph && prevNodeId != INVALID_TASK_ID) {
                    currentSubGraphTails.push_back(prevNodeId);
                    result.push_back({currentSubGraphHeads, currentSubGraphTails});
                    // 重置当前子图
                    currentSubGraphHeads.clear();
                    currentSubGraphTails.clear();
                }
                inSubGraph = false;
                prevNodeId = INVALID_TASK_ID;
            }

            nodeId = node->GetNextNodeId();
        }

        // 处理最后一个子图
        if (inSubGraph && prevNodeId != INVALID_TASK_ID) {
            currentSubGraphTails.push_back(prevNodeId);
            result.push_back({currentSubGraphHeads, currentSubGraphTails});
        }
    }

    // 如果没有任何子图，说明所有节点都不可融合，返回空结果
    printf("[sk info] SplitGraph: split into %zu subgraphs\n", result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        printf("[sk info]   SubGraph[%zu]: heads=%zu, tails=%zu\n",
               i, result[i].first.size(), result[i].second.size());
    }

    return result;
}

std::vector<SuperKernelBaseNode*> SuperKernelOptimizer::SerializeSubGraph(const SuperKernelGraph &graph,
                                                                     const std::vector<uint64_t> &headIds,
                                                                     const std::vector<uint64_t> &tailIds) const {
    std::vector<SuperKernelBaseNode*> taskNodes;

    // 遍历指定的头节点集合，收集节点直到遇到尾节点
    for (uint64_t headId : headIds) {
        uint64_t nodeId = headId;
        while (nodeId != INVALID_TASK_ID) {
            SuperKernelBaseNode *node = graph.GetNodeById(nodeId);
            if (node == nullptr) {
                break;
            }

            // 收集所有类型的节点（包括 KERNEL, NOTIFY, WAIT 等）
            // NOTIFY/WAIT 节点用于建立事件同步关系
            taskNodes.push_back(node);

            // 如果当前节点是尾节点之一，停止遍历
            if (std::find(tailIds.begin(), tailIds.end(), nodeId) != tailIds.end()) {
                break;
            }

            nodeId = node->GetNextNodeId();
        }
    }

    printf("[sk info] SerializeSubGraph: %zu heads, %zu tails, %zu total nodes (all types)\n",
           headIds.size(), tailIds.size(), taskNodes.size());
    return taskNodes;
}

// 调度执行任务流节点
void SuperKernelOptimizer::Schedule(const SuperKernelScopeInfo &scopeInfo,
                                    const SuperKernelGraph &graph) {
    auto taskNodes = scopeInfo.nodes;
    uint32_t streamIdx = scopeInfo.scopeStreamInfos[0].streamIdx;
    if (taskNodes.empty()) {
        printf("[sk warning] no task for super kernel optimization\n");
        return;
    }
    printf("[sk info] total task count : %zu, streamIdx=%u\n", taskNodes.size(), streamIdx);

    SkTaskBuilder builder(opts, graph);
    BuildResult build = builder.Build(taskNodes);

    DumpLaunchDebug(build.entryInfo, build.devArgs);
    KernelLauncher launcher;
    aclrtFuncHandle entryFunc = ResolveSkEntryFunc(build.entryInfo.funcName);
    if (entryFunc == nullptr) {
        free(build.devArgs);
        printf("[sk error] failed to resolve sk entry function\n");
        return;
    }
    aclrtStream stream = graph.GetStreamByIndex(streamIdx);
    if (stream == nullptr) {
        printf("[sk error] invalid streamIdx=%u, failed to get stream\n", streamIdx);
        free(build.devArgs);
        return;
    }

    printf("launch super kernel\n");
    printf("[sk info] sk entry func name : [%s] \n", build.entryInfo.funcName);
    printf("[sk info] sk entry block dim : [%u] \n", build.entryInfo.blockDim);

    void *addr[2] = {0};
    CHECK_ACL(aclrtGetFunctionAddr(entryFunc, addr, addr + 1));
    printf("[sk info] sk entry aic addr : pcAddr=%p \n", addr[0]);
    printf("[sk info] sk entry aiv addr : pcAddr=%p \n", addr[1]);

    launcher.Launch(build.entryInfo, build.devArgs, stream, entryFunc);
    free(build.devArgs);
    CHECK_ACL(aclrtSynchronizeStreamWithTimeout(stream, 1000));
}

void SuperKernelOptimizer::DumpLaunchDebug(const SkHostEntryInfo &entryInfo, const SkDeviceEntryArgs *skEntryArgs) {
    TaskQue *aicQue = (TaskQue *)((uint8_t *)skEntryArgs + skEntryArgs->skHeader.aicQueOffset);
    printf("[sk info] AIC TaskQue cap=%u task_cnt=%u\n", aicQue->cap, aicQue->taskCnt);
    for (uint32_t i = 0; i < aicQue->taskCnt; ++i) {
        TaskInfo &ti = aicQue->taskInfos[i];
        printf("[sk info] AIC task[%u] type=%u index=%u blocks=%u entry_cnt=%u args=0x%llx origin_type=%u\n",
               i, (unsigned)ti.type, (unsigned)ti.index, (unsigned)ti.blocks, (unsigned)ti.entryCnt,
               (unsigned long long)ti.args, (unsigned)ti.originType);
        for (uint32_t j = 0; j < ti.entryCnt; ++j) {
            printf("    AIC task[%u] entry[%u]=0x%llx\n", i, j, (unsigned long long)ti.entry[j]);
        }
    }

    TaskQue *aivQue = (TaskQue *)((uint8_t *)skEntryArgs + skEntryArgs->skHeader.aivQueOffset);
    printf("[sk info] AIV TaskQue cap=%u task_cnt=%u\n", aivQue->cap, aivQue->taskCnt);
    for (uint32_t i = 0; i < aivQue->taskCnt; ++i) {
        TaskInfo &ti = aivQue->taskInfos[i];
        printf("[sk info] AIV task[%u] type=%u index=%u blocks=%u entry_cnt=%u args=0x%llx origin_type=%u\n",
               i, (unsigned)ti.type, (unsigned)ti.index, (unsigned)ti.blocks, (unsigned)ti.entryCnt,
               (unsigned long long)ti.args, (unsigned)ti.originType);
        for (uint32_t j = 0; j < ti.entryCnt; ++j) {
            printf("    AIV task[%u] entry[%u]=0x%llx\n", i, j, (unsigned long long)ti.entry[j]);
        }
    }

    SkDfxInfo *dfxBase = (SkDfxInfo *)((uint8_t *)skEntryArgs + skEntryArgs->skHeader.dfxOffset);
    for (uint32_t i = 0; i < (uint32_t)skEntryArgs->skHeader.nodeCnt; ++i) {
        printf("[sk info] dfx[%u] bin_hdl=0x%llx func_hdl=0x%llx func_hdl_ori=0x%llx\n",
               i,
               (unsigned long long)dfxBase[i].binHdl,
               (unsigned long long)dfxBase[i].funcHdl,
               (unsigned long long)dfxBase[i].funcHdlOri);
    }

    printf("[sk info] skEntryArgs total_size=%zu node_cnt=%u\n",
           (size_t)skEntryArgs->skHeader.totalSize,
           (unsigned)skEntryArgs->skHeader.nodeCnt);

    printf("[sk info] offsets: aic=%zu aiv=%zu counter=%zu ws=%zu dfx=%zu\n",
           (size_t)skEntryArgs->skHeader.aicQueOffset,
           (size_t)skEntryArgs->skHeader.aivQueOffset,
           (size_t)skEntryArgs->skHeader.counterOffset,
           (size_t)skEntryArgs->skHeader.wsOffset,
           (size_t)skEntryArgs->skHeader.dfxOffset);
}

void SuperKernelOptimizer::Process(const SuperKernelGraph &graph) {
    // 切分图为多个子图
    SuperKernelScopeSplitter splitter(graph);
    if(splitter.SplitSingleStreamGraph()) {
        SK_LOGI("graph split into %zu scopes\n", splitter.GetScopeInfos().size());
    } else {
        SK_LOGW("graph split failed or no scopes found\n");
        return;
    }

    auto scopeInfos = splitter.GetScopeInfos();

    // 逐个处理每个子图
    for (auto scopeInfo : scopeInfos) {
        Schedule(scopeInfo, graph);
    }

    // graph.Update(); // 同步最新的执行状态到Graph
}
