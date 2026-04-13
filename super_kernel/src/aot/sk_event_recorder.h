/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SK_EVENT_RECORDER_H
#define SK_EVENT_RECORDER_H

#include <atomic>
#include <pthread.h>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include "sk_types.h"
#include "sk_common.h"
#include "sk_file_guard.h"
#include "sk_scope_postprocess.h"

// ==================== Node 信息结构体 ====================
struct SkNodeInfo {
    std::string nodeName;
    uint32_t numBlocks{0};
};

// ==================== 常量定义 ====================
constexpr uint32_t SK_EVENT_MAX_DEVICE_NUM = 16;
constexpr uint32_t SK_EVENT_CORE_NUM = 75;           // 每个 device 的 core 数
constexpr uint32_t SK_EVENT_DEFAULT_CORE_SIZE = 1024 * 1024; // 默认每个 core 1MB
#if defined(NPU_ARCH) && NPU_ARCH == 310
constexpr uint32_t TICK_US_MULTIPLER = 1000;
#else
constexpr uint32_t TICK_US_MULTIPLER = 50;
#endif
constexpr uint32_t SHAPE_MAX_TENSOR_NUM = 800;

// 打点环境变量名称
static const char* ENV_SK_EVENT_RECORD = "ASCEND_PROF_SK_ON";
constexpr uint32_t  SPRINT_LEN_BUFFER = 1024;

// ==================== Device 上下文 ====================
class SkEventRecorder;
struct SkEventDeviceCtx {
    std::atomic_int active{0};                                  // 默认不激活
    void* gmAddr = nullptr;                                    // GM 内存地址
    std::unique_ptr<uint8_t[]> hostBuf;                        // Host 缓冲区 (RAII)
    uint32_t deviceId = 0;                                     // Device ID
    uint32_t totalSize = 0;                                    // 总大小
    std::string outputDir;                                     // 每个device的profiling输出目录路径
    FileGuard outputFp;                                        // 小算子的时间信息文件的输出文件句柄 (RAII)
    uint64_t lastOffset[SK_EVENT_CORE_NUM]{};                  // 每个core的上次读取位置
    SkEventRecorder* recorder = nullptr;                        // 回调指针
};


// ==================== 全局管理器 ====================
class SkEventRecorder {
public:
    static SkEventRecorder& Instance();
    
    // 初始化（检查环境变量，如果开启则初始化）
    bool Init();
    
    // 为指定 device 分配 GM 空间并返回地址
    void* GetGmAddrForDevice(uint32_t deviceId);

    // 关闭所有资源
    void SkProfilingShutdown();
    
    // 是否启用
    bool IsEnabled() const { return enabled; }

    static uint32_t GetCoreSize() { return coreSize_; }
    static uint32_t GetTotalSize() { return totalSize_; }

    // 添加 modelRI -> skId -> nodeId -> (nodeName, numBlocks) 映射（线程安全）
    void AddNodeInfoMapping(uint64_t modelRI, uint32_t skId, uint32_t nodeId,
                            const std::string& nodeName, uint32_t numBlocks);

    // 获取 NodeInfo（线程安全）
    SkNodeInfo GetNodeInfo(uint64_t modelRI, uint32_t skId, uint32_t nodeId) const;

    // 添加 skName 映射（线程安全）
    void AddSkNameMapping(uint64_t modelRI, uint32_t skId, const std::string& skName);

    // 获取 skName（线程安全）
    std::string GetSkName(uint64_t modelRI, uint32_t skId) const;

private:
    SkEventRecorder() = default;
    ~SkEventRecorder();

    // 创建输出目录 sk_meta/<pid>，返回目录路径
    static std::string CreateOutputDir();

    // 根据环境变量计算并设置 coreSize 和 totalSize，返回是否启用
    static bool ParseEnvAndSetSize();

    // 创建 device 上下文（带加锁）
    SkEventDeviceCtx* CreateDeviceCtx(uint32_t deviceId);

    // 后台线程入口
    static void* DumpThreadFunc(void* arg);

    void DumpModelData(SkEventRecorder* recorder);
    
    // 拷贝并解析 GM 数据
    void DumpDeviceData(SkEventDeviceCtx* ctx);

    // 写入单个节点事件到 JSON 文件
    bool WriteNodeEventToJson(SkEventDeviceCtx* ctx, const SkKernelEventRecord* record,
                              uint32_t core, const SkNodeInfo& nodeInfo);

    // 输出 sk 级别统计事件到 JSON 文件
    bool WriteSkEventToJson(SkEventDeviceCtx* ctx, const SkKernelEventRecord* record, uint32_t core);

private:
    std::atomic_bool enabled{false}; // 打点是否执行
    std::atomic_bool globalRunning{false};  // 全局打点解析线程运行状态
    pthread_t dumpThread;                   // 全局打点解析后台线程
    std::once_flag initFlag_;               // 保证 Init() 创建后台线程只执行一次
    std::mutex mutex;
    SkEventDeviceCtx deviceCtxs; // device上下文列表

    static uint32_t coreSize_;   // 每个 core 的profiling 记录的gm缓冲区大小（字节），由环境变量决定
    static uint32_t totalSize_;  // 总缓冲区大小 = SK_EVENT_CORE_NUM * coreSize_

    // NodeInfo 映射表：modelRI -> skId -> nodeId -> NodeInfo
    mutable std::mutex nodeInfoMapMutex;
    std::unordered_map<uint64_t,
                       std::unordered_map<uint32_t,
                                          std::unordered_map<uint32_t, SkNodeInfo>>> nodeInfoMap;

    // SkName 映射表：modelRI -> skId -> skName
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, std::string>> skNameMap;
};

// ==================== sk profiling 性能分析相关函数 ====================
// 根据内核类型获取入口函数名
const char* GetEntryFuncNameByOpType(SkKernelType& opType);

// 性能分析处理
bool SkProfiling(const SuperKernelScopeInfo& scopeInfo, SkLaunchInfo& launchInfo, SuperKernelGraph& graph);

// 详细性能分析数据输出
bool DumpProfilingDetail(const std::vector<SuperKernelBaseNode*>& taskNodes, SkLaunchInfo& launchInfo,
                         const SuperKernelScopeInfo& scopeInfo, aclmdlRI modelRI);
#endif // SK_EVENT_RECORDER_H
